# os.chdir(r'C:\Users\louis_kreitmann\DEEP_ACA_7plex_cluster\CDAN')

import argparse
import warnings
import os
import sys
import os.path as osp
import numpy as np
import pandas as pd
from math import floor
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import lr_schedule
import loss
import torch.optim as optim
from train import train, train_2, train_CDAN
from torch.autograd import Variable
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from data_loader import generate_data_loader
import network
from network import Transformer
from utils import plot_cm, visualize, proxy_a_distance, plot_learning_curves
import warnings 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from utils import split_data_labels, df_to_tensor, str_to_int
import torch
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend
import matplotlib.pyplot as plt
plt.ion()
from pathlib import Path
from tst.encoder import Encoder
from tst.utils import generate_original_PE, generate_regular_PE


dir_data = '7_plex'
dir_out = '7_plex_output'

# parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
# parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
# parser.add_argument('--num_iterations', type=int, default=4000)
# parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
# parser.add_argument('--dir_data', type=str, default=dir_data_2025, help="directory of data")
# parser.add_argument('--dir_out', type=str, default=dir_out, help="output directory of our model (in ../snapshot directory)")
# parser.add_argument('--lr', type=float, default= 1e-3, help="learning rate")
# parser.add_argument('--random', type=bool, default= False, help="whether use random projection")
# # args = parser.parse_args()
# args, unknown = parser.parse_known_args()
# print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= CDAN Framework Setup ================ #
# config = {
#     "method": args.method,
#     "num_iterations": args.num_iterations,
#     "test_interval": args.test_interval,
#     "output_path": args.dir_out,
#     "loss": {"trade_off": 1.0, "random": args.random, "random_dim": 512},
#     "optimizer": {"type":optim.Adam, "optim_params":{'lr':args.lr, \
#                         "weight_decay":0.0001}, "lr_type":"inv", \
#                         "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}},
#     "data": {"dir_data": args.dir_data, 
#                 "source":{"name": ['df_dPCR_GB_2025.csv'], "batch_size":256}, 
#                 "target":{"name": "df_dPCR_SP_2025.csv", "batch_size":256},
#                 "test":{"name": "df_dPCR_SP_2025.csv", "batch_size":256}},
#     "class_num": 7
# }

config = {
    "method": 'CDAN+E',
    "num_iterations": 40,
    "test_interval": 20,
    "output_path": dir_out,
    "loss": {"trade_off": 1.0, "random": False, "random_dim": 512},
    "optimizer": {"type":optim.Adam, "optim_params":{'lr':1e-3, \
                        "weight_decay":0.0001}, "lr_type":"inv", \
                        "lr_param":{"lr":1e-3, "gamma":0.001, "power":0.75}},
    "data": {"dir_data": dir_data, 
             "source":{"name": ["df_dPCR_GB_2025.csv"], "batch_size":128}, 
             "target":{"name": "df_dPCR_SP_2025.csv", "batch_size":128},
             "test":{"name": "df_dPCR_SP_2025.csv", "batch_size":128}},
    "class_num": 7
}

# ================= Feature Extractor Setup ================ #
F_config = {
    'd_input': 1,
    'd_model': 512, # Lattent dim
    'q': 16, # Query size
    'v': 16,  # Value size
    'h': 4, # Number of self-attention heads
    'N': 4, # Number of encoder to stack
    'attention_size': 20,  # Attention window size
    'dropout': 0.5, # drop out rate
    'chunk_mode': None,
    'pe': "regular",  # Positional encoding metric
    # 'batch_first': True,
    'pe_period': 5
}

# ================= Data Preparation ================ #
data_loaders = {}
data_config = config["data"]
train_bs = data_config["source"]["batch_size"]
train_bt = data_config["target"]["batch_size"]
test_b = data_config["test"]["batch_size"]
dir_data = data_config["dir_data"]
source_data_name = data_config["source"]["name"]
target_data_name = data_config["target"]["name"]
test_data_name = data_config["test"]["name"]

torch.manual_seed(0) 

source_X_df, source_Y_df, target_X_df, target_Y_df, test_X_df, test_Y_df, \
    data_loaders = generate_data_loader(dir_data, train_bs, train_bt, test_b, \
                                        source_data_name, target_data_name, test_data_name, \
                                            device, use_normalize = 'min_max') 


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

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
                 bottleneck_dim=256):
                 
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
                nn.ReLU(inplace=True),
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
        print('K:', K)

        # unsqueeze to add channel dimension
        if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(2)
        print('input_data.shape:', input_data.shape)

        # Embedding module
        encoding = self._embedding(input_data)
        print('encoding.shape:', encoding.shape)
        
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)
        
        # Encoding
        for layer in self.layers_encoding:
            encoding = layer(encoding)
            print('MHA')
        
        print('encoding.shape:', encoding.shape)
            
        x = torch.relu(encoding)
        
        # Classification on source domain data
        x = self.classifier(x)
        print('x.shape:', x.shape)

        if self.use_bottleneck:
            x = self.bottleneck(x)

        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features                                              

Generator = Transformer(config['class_num'], **F_config).to(device)

# Initialize domain classifier.
if config["loss"]["random"]:
    random_layer = network.RandomLayer([Generator.output_num(), config['class_num']], config["loss"]["random_dim"], device).to(device)
    ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 256).to(device)
else:
    random_layer = None
    ad_net = network.AdversarialNetwork(Generator.output_num() * config['class_num'], 256).to(device)

base_network = Generator

parameter_list = base_network.get_parameters() + ad_net.get_parameters()
optimizer_config = config["optimizer"]
optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
                
schedule_param = optimizer_config["lr_param"]
lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

## train   
len_train_source = len(data_loaders["source"])
len_train_target = len(data_loaders["target"])
best_acc = 0.0
best_model = None

base_network.train(True)

classifier_loss_iter = []
transfer_loss_iter = []
iters = 0

for i in range(config["num_iterations"]):

    iters += 1

    # train one iter.
    # base_network.train(True)
    # ad_net.train(True)

    loss_params = config["loss"]  
    optimizer = lr_scheduler(optimizer, i, **schedule_param)
    optimizer.zero_grad()

    if i % len_train_source == 0:
        iter_source = iter(data_loaders["source"])

    if i % len_train_target == 0:
        iter_target = iter(data_loaders["target"])
    
    inputs_source, labels_source = next(iter_source)
    inputs_target, _= next(iter_target)
    inputs_source, inputs_target, labels_source = inputs_source.to(device), inputs_target.to(device), labels_source.to(device)
    
    # inputs_source, labels_source = inputs_source.to(device), labels_source.to(device)
    features_source, outputs_source = base_network(inputs_source)
    features_target, outputs_target = base_network(inputs_target)
    features = torch.cat((features_source, features_target), dim=0)
    outputs = torch.cat((outputs_source, outputs_target), dim=0)
    softmax_out = nn.Softmax(dim=1)(outputs)

    if config['method'] == 'CDAN+E':           
        entropy = loss.Entropy(softmax_out)
        _, transfer_loss = loss.CDAN([features, softmax_out], ad_net, device, entropy, network.calc_coeff(i), random_layer)
    elif config['method']  == 'CDAN':
        _, transfer_loss = loss.CDAN([features, softmax_out], ad_net, device, None, None, random_layer)
    elif config['method']  == 'DANN':
        _, transfer_loss = loss.DANN(features, ad_net)
    else:
        raise ValueError('Method cannot be recognized.')

    classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
    total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
    # total_loss = classifier_loss
    total_loss.backward()
    optimizer.step()



    # ================= CL  ================ #
    learner = BYOL(
        transformer_model,
        image_size = 60,
        hidden_layer = 'avgpool'
    )
