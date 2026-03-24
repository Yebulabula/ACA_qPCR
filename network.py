from random import random
import numpy as np
import torch
import torch.nn as nn
import math
from tst.encoder import Encoder
from tst.utils import generate_original_PE, generate_regular_PE

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float32(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

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


class Transformer(nn.Module):
    """
    Trasnformer model which follows the sturecture of original paper:
    https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    """
    def __init__(self, 
                 class_num: int, 
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
                 
        super(Transformer, self).__init__()
        self._d_model = d_model
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                          q,
                          v,
                          h,
                          attention_size=attention_size,
                          dropout=dropout,
                          chunk_mode=chunk_mode) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None

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

        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features = self._d_model * 60, out_features = 128),
                nn.Dropout(),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(in_features = 128, out_features = 128)
            )
        self.classifier.apply(init_weights)

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
        if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(2)

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
            
        # x = torch.GELU(encoding)
        
        # Classification on source domain data
        x = self.classifier(encoding)

        if self.use_bottleneck:
            x = self.bottleneck(x)

        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork obtained from official CDAN repository:
    https://github.com/thuml/CDAN/blob/master/pytorch/network.py
    """
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.GELU1 = nn.GELU()
        self.GELU2 = nn.GELU()
        self.GELU3 = nn.GELU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.GELU1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.GELU2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1
        
    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":15, 'decay_mult':2}]
