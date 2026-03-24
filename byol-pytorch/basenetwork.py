import torch
import torch.nn as nn

class Transformer1D(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, batch_first=True):
        super(Transformer1D, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, d_model)  # Output layer to match the BYOL requirement
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(-1).float()  # Add channel dimension and change to float
        x = self.embedding(x)
        embedding = self.transformer_encoder(x)
        out = self.avgpool(embedding.permute(1, 2, 0)).squeeze(2)
        return embedding, self.output_layer(out)

import torch
import torch.nn as nn
from tst.encoder import Encoder
from tst.utils import generate_original_PE, generate_regular_PE
import math


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim, device):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim).to(device) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor
    
class Transformer_for_byol(nn.Module):
    """
    Trasnformer model which follows the sturecture of original paper:
    https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    """
    def __init__(self,
                 class_num: int,
                 N_seq: int,
                 d_input: int,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = 5,
                 use_bottleneck=True,
                 bottleneck_dim=128):

        super(Transformer_for_byol, self).__init__()
        self._d_model = d_model
        self.N_seq = 60
        self._embedding = nn.Linear(d_input, d_model)
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                          q,
                          v,
                          h,
                          attention_size=attention_size,
                          dropout=dropout,
                          chunk_mode=chunk_mode) for _ in range(N)])

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None

        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features = self._d_model * N_seq, out_features = 128),
                nn.Dropout(),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(in_features = 128, out_features = 128)
            )
        self.classifier.apply(init_weights)

        self.use_bottleneck = use_bottleneck

        if self.use_bottleneck:
            self.bottleneck = nn.Linear(128, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(128, class_num)
            self.fc.apply(init_weights)
            self.__in_features = 128

        self.output_layer = nn.Linear(N_seq, d_model)  # Output layer to match the BYOL requirement
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def get_parameters(self):
        if self.use_bottleneck:
            parameter_list = [{"params":self._embedding.parameters(), "lr_mult":1, 'decay_mult':2},
                            {"params":self.layers_encoding.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self._embedding.parameters(), "lr_mult":1, 'decay_mult':2},
                            {"params":self.layers_encoding.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        return parameter_list

    def forward(self, input_data):
        K = input_data.shape[1]
        # unsqueeze to add channel dimension
        input_data.to(torch.float32)
        if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(2)

        # print(input_data.dtype)

        # Embedding module
        encoding = self._embedding(input_data)

        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Classification on source domain data
        x = self.classifier(encoding)
        if self.use_bottleneck:
            x = self.bottleneck(x)

        y = self.fc(x)

        # z = encoding.permute(1,2,0)
        z = self.avgpool(encoding)
        z = z.squeeze(2)
        z = self.output_layer(z)

        return x, y, z
        # x (N_batch, N_cycles, d_model) is features (in CDAN), embeddings (BYOL)
        # y (N_batch, class_num) is logits of last layer for classification
        # z (N_batch, N_seq) is result of avgpooling for BYOL

    def output_num(self):
        return self.__in_features