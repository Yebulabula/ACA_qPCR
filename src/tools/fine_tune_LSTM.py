import argparse
import copy
import itertools
import pandas as pd
import warnings
import os
import os.path as osp
import numpy as np
import pandas as pd
from math import floor
import lr_schedule
import sys
import torch
import torch.optim as optim
from train import train, train_2, train_CDAN
from torch.autograd import Variable
import torch.nn as nn
from test import test
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

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  
            batch_first=True,
            bidirectional=bidirectional
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            features = h_n[-1,:,:]

        output = self.fc(features)
        return features, output

# LSTM Configuration
LSTM_config = {
    'input_dim': 1,
    'hidden_dim': 512,
    'num_layers': 4,
    'num_classes': 7,
    'dropout': 0.5,
    'bidirectional': False
}

def train_tmp(config, base_network, data_loaders, device):
    parameter_list = base_network.parameters()  # Fixed: use parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))

    best_acc = 0.0
    best_model = None

    base_network.train(True)

    for i in range(config["num_iterations"]):
        optimizer.zero_grad()

        iter_source = iter(data_loaders["source"])
        inputs_source, labels_source = next(iter_source)
        inputs_source, labels_source = inputs_source.to(device), labels_source.to(device)

        if len(inputs_source.shape) == 2:  # Ensure correct shape
            inputs_source = inputs_source.unsqueeze(-1)

        _, outputs_source = base_network(inputs_source)

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
        classifier_loss.backward()
        optimizer.step()

    return best_acc, best_model

def grid_search_hyperparams(base_config, base_LSTM_config, data_loaders, device):
    param_grid = {
        "lr":         [1e-4, 1e-3, 1e-2],
        "momentum":   [0.8, 0.9, 0.99],
        "num_layers": [2, 4, 6],
        "hidden_dim": [256, 512, 1024],
        "dropout":    [0.1, 0.3, 0.5]
    }

    results = []

    for (lr, mom, num_layers, hidden_dim, dp) in itertools.product(
        param_grid["lr"],
        param_grid["momentum"],
        param_grid["num_layers"],
        param_grid["hidden_dim"],
        param_grid["dropout"]
    ):
        run_config = copy.deepcopy(base_config)
        run_LSTM_config = copy.deepcopy(base_LSTM_config)

        run_config["optimizer"]["optim_params"]["lr"] = lr
        run_LSTM_config["num_layers"] = num_layers
        run_LSTM_config["hidden_dim"] = hidden_dim
        run_LSTM_config["dropout"] = dp

        Generator = LSTMClassifier(
            input_dim=run_LSTM_config["input_dim"],
            hidden_dim=run_LSTM_config["hidden_dim"],
            num_layers=run_LSTM_config["num_layers"],
            num_classes=run_config['class_num'],
            dropout=run_LSTM_config["dropout"]
        ).to(device)

        best_acc, best_model = train_tmp(run_config, Generator, data_loaders, device)

        results.append({
            "lr":       lr,
            "momentum": mom,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "dropout":  dp,
            "best_acc":  best_acc
        })

    df_results = pd.DataFrame(results)
    return df_results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device for training: {device}')

    config = {
        "method": 'CDAN+E',
        "num_iterations": 4001,
        "test_interval": 2000,
        "loss": {"trade_off": 1.0, "random": False, "random_dim": 512},
        "optimizer": {
            "type": optim.Adam, 
            "optim_params": {'lr': 1e-3, "weight_decay": 0.0001}, 
            "lr_type": "inv",
            "lr_param": {"lr": 1e-3, "gamma": 0.001, "power": 0.75}
        },
        "class_num": 7
    }

    torch.manual_seed(0) 

    df_search = grid_search_hyperparams(
        base_config=config, 
        base_LSTM_config=LSTM_config, 
        data_loaders=data_loaders, 
        device=device
    )

    print("Grid Search Results:")
    print(df_search)
    df_search.to_csv("grid_search_results_lstm.csv", index=False)