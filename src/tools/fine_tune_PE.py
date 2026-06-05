import argparse
import warnings
import os
import os.path as osp
from test import test

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from data_loader import generate_data_loader
from network import Transformer
import lr_schedule
import sys
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn  # Add this line
import torch.optim as optim
from torch.autograd import Variable
from utils import split_data_labels

# Define directories
dir_data_2025 = '7_plex'
dir_out = '7_plex_output'

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Activities for classification
Activities = ['Adeno', 'COVID', 'Cov_229E', 'Cov_HKU1', 'Cov_NL63', 'Cov_OC43', 'MERS']

def evaluate_model_accuracy(best_model, data_loader, mode='test'):
    """
    Evaluates the classification accuracy of the model on a given dataset.
    """
    best_model.eval()
    pred_list = []
    label_list = []

    with torch.no_grad():
        for data, label in data_loader[mode]:
            data, label = data.to(device), label.to(device)
            _, output = best_model(data)
            pred = output.argmax(dim=1).cpu().numpy()
            label = label.cpu().numpy()
            pred_list.extend(pred)
            label_list.extend(label)

    report = classification_report(label_list, pred_list, target_names=Activities, output_dict=True)
    return report['accuracy']

def train(config, base_network, data_loaders, device):
    
    parameter_list = base_network.get_parameters() 
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    dir_out = config["output_path"]

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
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
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

    # Save training and testing curves
    training_curve = {"training_accuracy": training_accuracy, "testing_accuracy": testing_accuracy}
    # Extract training and testing accuracy data
    train_iters, train_acc = zip(*training_curve["training_accuracy"])
    test_iters, test_acc = zip(*training_curve["testing_accuracy"])

    # Convert training accuracy to percentages
    train_acc_percentage = [x * 100 for x in train_acc]

    # Plot training and testing accuracy curves
    plt.figure()
    plt.plot(train_iters, train_acc_percentage, label="Training Accuracy (%)")
    plt.plot(test_iters, test_acc, label="Testing Accuracy (%)")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Testing Accuracy Curves")
    plt.legend()
    plt.savefig(os.path.join(dir_out, "training_curve.png"))  # Save plot to file

    # Save the training curve to dir_out
    output_path = os.path.join(config["output_path"], "training_curve.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(training_curve, f)
    print(f"Training curve saved to {output_path}")

    return best_acc, best_model, training_curve

def main():
    # Training configuration
    config = {
        "method": 'CDAN+E',
        "num_iterations": 5000,
        "test_interval": 1000,
        "output_path": dir_out,
        "loss": {"trade_off": 1.0, "random": False, "random_dim": 512},
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-3, "weight_decay": 0.0001}, "lr_type": "inv", "lr_param": {"lr": 1e-3, "gamma": 0.001, "power": 0.75}},
        "data": {"dir_data": dir_data_2025, "source": {"name": ['df_simulated_ACs_norm.csv'], "batch_size": 128}, 
                                        "target": {"name": 'df_qPCR_SP_2025.csv', "batch_size": 128}, 
                                        "test": {"name": 'df_qPCR_GB_2025_conc3_clean.csv', "batch_size": 32}},
        "class_num": 7
    }

    F_config = {
        'd_input': 1,
        'd_model': 128,
        'q': 32,
        'v': 32,
        'h': 4,
        'N': 4,
        'attention_size': 5,
        'dropout': 0.5,
        'chunk_mode': None,
        'pe': "regular",
        'pe_period': None  # To be tuned
    }

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+ config["output_path"])
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    # Save framework and model information.
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    config["out_file"].write("CDAN framework for ACA configuration" + str(config) + "\n")
    config["out_file"].flush()
    config["out_file"].write("Transformer-based feature extractor configuration" + str(F_config))
    config["out_file"].flush()

    # Generate data loaders
    data_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    train_bt = data_config["target"]["batch_size"]
    test_b = data_config["test"]["batch_size"]
    dir_data = data_config["dir_data"]
    source_data_name = data_config["source"]["name"]
    target_data_name = data_config["target"]["name"]
    test_data_name = data_config["test"]["name"]

    source_X_df, source_Y_df, target_X_df, target_Y_df, test_X_df, test_Y_df, data_loaders = generate_data_loader(
        dir_data, train_bs, train_bt, test_b, source_data_name, target_data_name, test_data_name, device, use_normalize='min_max'
    )

    # Iterate over pe_period values
    pe_period_values = range(0, 61, 5)
    results = []

    for pe_period in pe_period_values:
        print(f"Tuning pe_period: {pe_period}")

        # Update pe_period in the config
        F_config['pe_period'] = pe_period

        # Initialize model
        Generator = Transformer(config['class_num'], **F_config).to(device)

        # Train the model
        _, best_model, _ = train(config, Generator, data_loaders, device)

        # Evaluate the model
        accuracy = evaluate_model_accuracy(best_model, data_loaders)
        results.append((pe_period, accuracy))
        print(f"pe_period: {pe_period}, Accuracy: {accuracy:.5f}")

    # Print final results
    print("\nFinal Results:")
    for pe_period, accuracy in results:
        print(f"pe_period: {pe_period}, Accuracy: {accuracy:.5f}")

if __name__ == "__main__":
    main()
