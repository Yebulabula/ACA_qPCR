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


dir_data_2025 = '7_plex'
dir_out = '7_plex_output'

def eval_best_model(best_model, data_loaders, Activities, save_name, mode = 'test'): 
    """
        This method is to evaluate the best-performing model on target domain data. 

        INPUT:
        best_model: best transformer-based generator.
        dset_loaders: A collection data loaders with batach size (148). 
        Activities: The categories of targets required to be classified. 
        save_name: The name of the model for evaluating.
    """
    # Curve-level performance evaluation.
    print("-------Curve-level performance evaluation-------")
    best_model.eval()
    best_model.to(device)
    pred_list = torch.tensor([]).to(device)
    label_list = torch.tensor([]).to(device)
    # sample_list = torch.tensor([]).to(device)
    count = 0
    for data, label in data_loaders[mode]: 
        data, label = data.to(device), label.to(device)
        # note on volatile: https://stackoverflow.com/questions/49837638/what-is-volatile-variable-in-pytorch
        data, label = Variable(data, volatile=True), Variable(label)
        _, output = best_model(data) 

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
       
        pred = torch.flatten(pred)
        label = torch.flatten(label)
        
        pred_list = torch.cat((pred_list, pred), 0).to(device)
        label_list = torch.cat((label_list, label), 0).to(device)
    
    print(classification_report(label_list.cpu().numpy(), pred_list.cpu().numpy(), digits = 5, target_names = Activities))

    y_true = label_list.cpu().numpy()
    y_pred = pred_list.cpu().numpy()
    
    # Plot confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Activities)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {save_name}')
    plt.savefig(os.path.join(dir_out, "confusion_matrix.png"))  # Save plot to file

def train_tmp(config, base_network, data_loaders, device):
    """
    Train the model and compute training and testing accuracy curves.

    INPUT:
    config: framework configuration.
    base_network: Feature extractor model.
    ad_net: Domain classifier network.
    random_layer: Random projectaion layer for calculating A_distance.
    data_loaders: Data loaders for source, target, and test datasets.
    device: Device to run the model on ('cuda' or 'cpu').

    RETURN:
    best_acc: Best testing accuracy achieved.
    best_model: Model corresponding to the best testing accuracy.
    training_curve: Dictionary containing training and testing accuracy curves.
    """
    parameter_list = base_network.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    # dir_out = config["output_path"]

    len_train_source = len(data_loaders["source"])
    best_acc = 0.0
    best_model = None

    # Initialize lists to store training and testing accuracy
    training_accuracy = []
    testing_accuracy = []

    base_network.train(True)

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            
            # Evaluate on test data
            test_target = test(base_network, data_loaders["test"], i, device)
            temp_acc = test_target['accuracy %']
            temp_model = nn.Sequential(base_network)
            
            # Save best model if this iteration has better accuracy
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model

            # Log testing accuracy
            testing_accuracy.append((i + 1, temp_acc))
            
            # Print testing info
            log_str = "[iter: {:04d} / all {:05d}], Testing Accuracy: {:.5f}".format(i + 1, config["num_iterations"], temp_acc)
            # config["out_file"].write(log_str + "\n")
            # config["out_file"].flush()
            print(f'\n{log_str}')

        # Train one iteration
        loss_params = config["loss"]
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(data_loaders["source"])

        inputs_source, labels_source = next(iter_source)
        inputs_source, labels_source = inputs_source.to(device), labels_source.to(device)
        features_source, outputs_source = base_network(inputs_source)

        # Calculate classification loss
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
        total_loss = classifier_loss
        total_loss.backward()
        optimizer.step()

        # Compute training accuracy for the current batch
        preds = outputs_source.argmax(dim=1)
        train_acc = (preds == labels_source).float().mean().item()
        training_accuracy.append((i + 1, train_acc))

        # Print training info
        sys.stdout.write('\r[iter: %d / all %d], Classification loss: %f, Training Accuracy: %f, Testing Accuracy: %f, Best Testing Accuracy: %f' %
                         (i + 1, config["num_iterations"], classifier_loss.item(), train_acc, temp_acc if i % config["test_interval"] == config["test_interval"] - 1 else 0, best_acc))
        sys.stdout.flush()

    # Save best model
    best_model = nn.Sequential(base_network)

    return train_acc, best_acc, best_model

def grid_search_hyperparams(base_config, base_F_config, data_loaders, device):
    
    # Define hyperparameter grid
    param_grid = {
        "lr":         [1e-3, 1e-2],
        "momentum":   [0.9, 0.99],
        "num_heads":  [4, 8, 16],
        "num_layers": [2, 4, 6],
        "dropout":    [0.1, 0.3, 0.5]
    }

    results = []  # to store each run's result

    # Iterate over *all* possible combinations
    for (lr, mom, h, N, dp) in itertools.product(
        param_grid["lr"],
        param_grid["momentum"],
        param_grid["num_heads"],
        param_grid["num_layers"],
        param_grid["dropout"]
    ):
        # 1) Copy config & F_config so we don't mutate them
        run_config   = copy.deepcopy(base_config)
        run_F_config = copy.deepcopy(base_F_config)

        # 2) Overwrite relevant params
        run_config["optimizer"]["optim_params"]["lr"] = lr
        # If using Adam, 'momentum' may not apply. If using SGD, do:
        # run_config["optimizer"]["optim_params"]["momentum"] = mom

        run_F_config["h"] = h
        run_F_config["N"] = N
        run_F_config["dropout"] = dp

        print(f"Running with lr={lr}, momentum={mom}, num_heads={h}, num_layers={N}, dropout={dp}")

        # 3) Instantiate new model
        # run_config['class_num'] must be set in base_config
        Generator = Transformer(run_config['class_num'], **run_F_config).to(device)

        # 4) Train the model with your training function
        #    If train_tmp returns (train_acc, best_acc, best_model), do that; 
        #    otherwise adjust accordingly. 
        train_acc, best_acc, best_model = train_tmp(run_config, Generator, data_loaders, device)

        # 5) Evaluate best_model (if you want to do it here)
        Activities = ['Adeno', 'COVID', 'Cov_229E', 'Cov_HKU1', 'Cov_NL63', 'Cov_OC43', 'MERS']
        best_model.eval()
        eval_best_model(best_model, data_loaders, Activities, save_name="CDAN", mode="test")

        # 6) Store results
        results.append({
            "lr":       lr,
            "momentum": mom,
            "num_heads": h,
            "num_layers": N,
            "dropout":  dp,
            "train_acc": train_acc,
            "best_acc":  best_acc
        })
        df_results_tmp = pd.DataFrame(results)
        df_results_tmp.to_csv(os.path.join(dir_out, "df_results.csv"), index=False)

    # 7) Build DataFrame outside the loop
    df_results = pd.DataFrame(results)
    return df_results

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device for training: {device}')

    results = []
    
    # ================= CDAN Framework Setup ================ #
    config = {
    "method": 'CDAN+E',
    "num_iterations": 4001,
    "test_interval": 2000,
    # "output_path": "CDAN_ACA/" + 'log',
    "loss": {"trade_off": 1.0, "random": False, "random_dim": 512},
    "optimizer": {
        "type": optim.Adam, 
        "optim_params": {'lr': 1e-3, "weight_decay": 0.0001}, 
        "lr_type": "inv",
        "lr_param": {"lr": 1e-3, "gamma": 0.001, "power": 0.75}
    },
    "data": {"dir_data": dir_data_2025, 
                 "source":{"name": ['df_dPCR_GB_2025.csv'], "batch_size":256}, 
                 "target":{"name": "df_dPCR_SP_2025.csv", "batch_size":256},
                 "test":{"name": "df_dPCR_SP_2025.csv", "batch_size":256}},
    "class_num": 7
}

    # ================= Feature Extractor Setup ================ #
    F_config = {
        'd_input': 1,
        'd_model': 512, # Lattent dim
        'q': 16, # Query size
        'v': 16,  # Value size
        'h': 8, # Number of self-attention heads
        'N': 4, # Number of encoder to stack
        'attention_size': 20,  # Attention window size
        'dropout': 0.5, # drop out rate
        'chunk_mode': None,
        'pe': "regular",  # Positional encoding metric
        # 'batch_first': True,
        'pe_period': 20
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
                                            
    df_search = grid_search_hyperparams(
        base_config=config, 
        base_F_config=F_config, 
        data_loaders=data_loaders, 
        device=device
    )
    print("Grid Search Results:")
    print(df_search)

    # Optionally save the results to CSV:
    df_search.to_csv(os.path.join(dir_out, "grid_search_results.csv"), index=False)
