import argparse
import datetime
import logging
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join('byol-pytorch'))
import warnings
import tqdm
from torch.utils.data import DataLoader
import lr_schedule
import matplotlib
import network
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from basenetwork import Transformer_for_byol
from byol_pytorch import BYOL
from tools.data_loader import generate_data_loader
from loss import CDAN, DANN, Entropy
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import Dataset

matplotlib.use('Agg')  # Use non-GUI backend suitable for headless environments
import matplotlib.pyplot as plt

plt.ion()



# Directories
dir_data_2025 = '../7_plex_data'
dir_out = '7_plex_output'
PRETRAINED_CL_CHECKPOINT = '../pretrained_model_CL_299.pth'

# Ensure output directory exists
os.makedirs(dir_out, exist_ok=True)

# Create a timestamp for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(dir_out, f'training_log_{timestamp}.txt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Log the start of execution
logging.info("Starting BYOL_CDAN training. Logs saved to: %s", log_file)

def eval_best_model(best_model, data_loaders, activities, save_name, device, mode='test'):
    logging.info("-------Curve-level performance evaluation-------")
    best_model.eval()
    best_model.to(device)
    pred_list = torch.tensor([], device=device)
    label_list = torch.tensor([], device=device)

    with torch.no_grad():
        for data, label in data_loaders[mode]:
            data, label = data.to(device), label.to(device)
            _, output, _ = best_model(data)
            pred = output.data.max(1, keepdim=True)[1]
            pred_list = torch.cat((pred_list, torch.flatten(pred)), 0)
            label_list = torch.cat((label_list, torch.flatten(label)), 0)

    report = classification_report(
        label_list.cpu().numpy(),
        pred_list.cpu().numpy(),
        digits=5,
        target_names=activities,
    )
    logging.info("\n%s", report)

    y_true = label_list.cpu().numpy()
    y_pred = pred_list.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=activities)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {save_name}')
    cm_path = os.path.join(dir_out, f"confusion_matrix_{save_name}_{timestamp}.png")
    plt.savefig(cm_path)
    logging.info("Confusion matrix saved to: %s", cm_path)

    return 100.0 * (y_pred == y_true).sum() / len(y_true)


def sigmoid5_un(x, Fm, Fb, Sc, Cs, As):
    """5-parameter sigmoid model for PCR amplification curves."""
    return Fm / (1. + np.exp(-(x - Cs) * Sc)) ** As + Fb

def test(model, data_loader, iter, device):
    model.eval()
    test_loss = 0
    correct_class = 0

    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)
            _, output, _ = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct_class += pred.eq(label.data.view_as(pred)).cpu().sum()

    test_loss = test_loss / len(data_loader.dataset)
    accuracy = 100.0 * correct_class / len(data_loader.dataset)
    logging.info(f"Test iteration {iter}: Accuracy = {accuracy:.2f}%")

    return {
        "iter": iter,
        "average_loss": test_loss,
        "correct_class": correct_class,
        "total_elems": len(data_loader.dataset),
        "accuracy %": accuracy,
    }


def augment_fct(params_df, idx, sigmoid_fn=sigmoid5_un):
    param_cols = ['Fm', 'Fb', 'Sc', 'Cs', 'As']
    params = params_df.loc[idx, param_cols]
    param_to_change = str('Cs')
    params_aug = params.copy()
    
    mean = 24.835403682791412
    sigma = 8.098645591811588
    dist = np.random.normal(loc=mean, scale=sigma, size=len(params_df))
    params_aug[param_to_change] = np.random.choice(dist)
    x = np.arange(1, 60 + 1)
    y_aug = sigmoid_fn(x, *params_aug)
    return y_aug

import matplotlib.pyplot as plt
class PCRDataset_2(Dataset):
    def __init__(self, df_CL, dir_data, params_df):
        self.df_CL = df_CL
        self.params_df = pd.read_csv(params_df, index_col=0).dropna()
        self.dir_data = dir_data

    def __len__(self):
        return len(self.df_CL)

    def __getitem__(self, idx):
        # Return the normalized curves
        c1 = augment_fct(self.params_df, idx, sigmoid_fn=sigmoid5_un)
        c2 = augment_fct(self.params_df, idx, sigmoid_fn=sigmoid5_un)
        
        # baseline = self.df_CL.iloc[idx]
        # # draw the curves for visualization
        # plt.figure()
        # plt.plot(c1, label='Augmented Curve 1')
        # plt.plot(c2, label='Augmented Curve 2')
        # plt.plot(baseline, label='Original Curve', linestyle='dashed')
        # plt.title(f'Augmented Curves for Sample {idx}')
        # plt.xlabel('Cycle')
        # plt.ylabel('Fluorescence')
        # plt.legend()
        # curve_path = os.path.join(self.dir_data, f"augmented_curves_sample_{idx}.png")
        # plt.savefig(curve_path)
        return c1, c2
    
    def get_original_data(self, idx):
        """Return original non-normalized data"""
        return self.df_CL.iloc[idx]
    
def train_baseline(config, base_network, data_loaders, device):
    
    parameter_list = base_network.get_parameters() 
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    len_train_source = len(data_loaders["source"])
    best_acc = 0.0
    best_model = None
    training_accuracy = []
    testing_accuracy = []
    classifier_loss_iter = []
    iters = 0

    for i in range(config["num_iterations"]):
        base_network.train(True)
        iters += 1
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
                model_path = os.path.join(dir_out, f'baseline_best_model_{timestamp}.pth')
                torch.save(best_model.state_dict(), model_path)
                logging.info(f"New best baseline model saved with accuracy: {temp_acc:.2f}%")

            # Log testing accuracy
            testing_accuracy.append((i + 1, temp_acc))

        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(data_loaders["source"])

        inputs_source, labels_source = next(iter_source)
        inputs_source, labels_source = inputs_source.to(device), labels_source.to(device)
        _, outputs_source, _ = base_network(inputs_source)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
        
        classifier_loss.backward()
        optimizer.step()
        classifier_loss_iter.append(classifier_loss.item())

        # Compute training accuracy for the current batch
        preds = outputs_source.argmax(dim=1)
        train_acc = (preds == labels_source).float().mean().item()
        training_accuracy.append((i + 1, train_acc))

        # Log every 10 iterations to avoid too many logs
        if (i+1) % 500 == 0:
            logging.info(f'[Iter: {i+1}/{config["num_iterations"]}] Classification loss: {classifier_loss.item():.4f}')
    
    # Plot training and testing accuracy curves
    logging.info("Baseline training complete! Plotting training curves...")
    train_iters, train_acc = zip(*training_accuracy)
    test_iters, test_acc = zip(*testing_accuracy)
    train_acc_percentage = [x * 100 for x in train_acc]
    plt.figure()
    plt.plot(train_iters, train_acc_percentage, label="Training Accuracy (%)")
    plt.plot(test_iters, test_acc, label="Testing Accuracy (%)")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Testing Accuracy Curves - Baseline")
    plt.legend()
    training_curve_path = os.path.join(dir_out, f"training_curve_baseline_model_{timestamp}.png")
    plt.savefig(training_curve_path)
    logging.info(f"Training curve saved to: {training_curve_path}")

    return best_model

def train_CDAN(config, CDAN_model, ad_net, random_layer, data_loaders, device):
    logging.info("Starting CDAN training...")
    parameter_list = CDAN_model.get_parameters() + ad_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train   
    len_train_source = len(data_loaders["source"])
    len_train_target = len(data_loaders["target"])
    best_acc = 0.0
    best_model = None
    training_accuracy = []
    testing_accuracy = []
    classifier_loss_iter = []
    transfer_loss_iter = []
    total_loss_iter = []
    iters = 0

    for i in range(config["num_iterations"]):
        CDAN_model.train(True)
        iters += 1
        if i % config["test_interval"] == config["test_interval"] - 1:
            CDAN_model.train(False)
            
            # Evaluate on test data
            test_target = test(CDAN_model, data_loaders["test"], i, device)
            temp_acc = test_target['accuracy %']
            temp_model = nn.Sequential(CDAN_model)
            
            # Save best model if this iteration has better accuracy
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
                model_path = os.path.join(dir_out, f'best_model_CDAN_{timestamp}.pth')
                torch.save(best_model.state_dict(), model_path)
                logging.info(f"New best CDAN model saved with accuracy: {temp_acc:.2f}%")

            # Log testing accuracy
            testing_accuracy.append((i + 1, temp_acc))

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
        
        features_source, outputs_source, _ = CDAN_model(inputs_source)
        features_target, outputs_target, _ = CDAN_model(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        if config['method'] == 'CDAN+E':           
            entropy = Entropy(softmax_out)
            _, transfer_loss = CDAN([features, softmax_out], ad_net, device, entropy, network.calc_coeff(i), random_layer)
        elif config['method']  == 'CDAN':
            _, transfer_loss = CDAN([features, softmax_out], ad_net, device, None, None, random_layer)
        elif config['method']  == 'DANN':
            _, transfer_loss = DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source.long())
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()

        classifier_loss_iter.append(classifier_loss.item())
        transfer_loss_iter.append(transfer_loss.item())
        total_loss_iter.append(total_loss.item())

        # Compute training accuracy for current batch
        preds = outputs_source.argmax(dim=1)
        train_acc = (preds == labels_source).float().mean().item()
        training_accuracy.append((i + 1, train_acc))

        # Log every 10 iterations to avoid too many logs
        if (i+1) % 200 == 0:
            logging.info(f'[Iter: {i+1}/{config["num_iterations"]}] Classification loss: {classifier_loss.item():.4f}, ' +
                         f'CDAN loss: {transfer_loss.item():.4f}, Total loss: {total_loss.item():.4f}')
    
    # Plot training and testing accuracy curves
    logging.info("CDAN training complete! Plotting curves...")
    
    # Plot accuracy curves
    train_iters, train_acc = zip(*training_accuracy)
    test_iters, test_acc = zip(*testing_accuracy)
    train_acc_percentage = [x * 100 for x in train_acc]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_iters, train_acc_percentage, label="Training Accuracy (%)")
    plt.plot(test_iters, test_acc, label="Testing Accuracy (%)")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Testing Accuracy - CDAN")
    plt.legend()
    
    # Plot loss curves
    plt.subplot(1, 2, 2)
    plt.plot(range(1, iters+1), classifier_loss_iter, label="Classifier Loss")
    plt.plot(range(1, iters+1), transfer_loss_iter, label="Transfer Loss")
    plt.plot(range(1, iters+1), total_loss_iter, label="Total Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("CDAN Training Losses")
    plt.legend()
    
    plt.tight_layout()
    curves_path = os.path.join(dir_out, f"training_curves_CDAN_{timestamp}.png")
    plt.savefig(curves_path)
    logging.info(f"CDAN training curves saved to: {curves_path}")
    
    return best_model

def build_domain_adversary(model, config, device):
    if config["loss"]["random"]:
        random_layer = network.RandomLayer(
            [model.output_num(), config["class_num"]],
            config["loss"]["random_dim"],
            device,
        ).to(device)
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 256).to(device)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(
            model.output_num() * config["class_num"], 256
        ).to(device)
    return random_layer, ad_net


def load_matching_state_dict(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    model_state.update(filtered_state)
    model.load_state_dict(model_state)
    logging.info("Loaded %s matching tensors from %s", len(filtered_state), checkpoint_path)


def train_BYOL_CDAN(config, transformer_model, device):
    """
    First train with BYOL contrastive learning, then fine-tune with CDAN
    """
    # Step 1: Prepare data for contrastive learning
    # logging.info("Preparing data for contrastive learning...")
    df_curves_CL = pd.read_csv(os.path.join(config['data']['dir_data'], config['data']['CL']['name'][0])).iloc[:,3:63]
    
    dataset = PCRDataset_2(df_curves_CL, config['data']['dir_data'], '/mnt/new_drive/Documents/for_Ye/7_plex_data/param_df_5_20250305_2248.csv')
    train_loader_CL = DataLoader(dataset, batch_size=config['data']['CL']['batch_size'], shuffle=True)
    logging.info(f"Prepared contrastive learning dataset with {len(dataset)} samples")

    # Step 2: Setup BYOL learner
    logging.info("Setting up BYOL learner...")
    learner = BYOL(
        transformer_model,
        image_size=60,
        hidden_layer='avgpool',
        config=config,
        F_config=F_config
    ).to(device)

    # Step 3: Train with BYOL (contrastive learning)
    logging.info("Starting contrastive learning training...")
    opt = torch.optim.Adam(learner.parameters(), lr=3e-5, weight_decay=0.0001)
    best_loss = np.inf
    cl_log = {'train_loss': []}
    pretrained_model = None

    for epoch in range(config['epochs_CL']):
        avg_loss = 0
        for i, (x, y) in enumerate(tqdm.tqdm(train_loader_CL)):
            x, y = x.to(device), y.to(device)
            loss = learner(x, y)
            cl_log['train_loss'].append(loss.item())
            avg_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
        avg_loss /= (i + 1)
        logging.info(f'BYOL Epoch {epoch}, Avg Loss: {avg_loss:.6f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            pretrained_model = learner.online_encoder.net
            model_path = os.path.join(dir_out, f'pretrained_model_CL_{timestamp}.pth')
            torch.save(pretrained_model.state_dict(), model_path)
            logging.info(f'BYOL Epoch {epoch}: Model saved to {model_path} with loss {avg_loss:.6f}')
            
        if (epoch + 1) % 50 == 0:
            logging.info(f'BYOL Epoch {epoch}, Avg Loss: {avg_loss:.6f}, Best Loss: {best_loss:.6f}')
            pretrained_model = learner.online_encoder.net
            model_path = os.path.join(dir_out, f'pretrained_model_CL_{timestamp}.pth')
            torch.save(pretrained_model.state_dict(), model_path)
            logging.info(f'BYOL Epoch {epoch}: Model saved to {model_path} with loss {avg_loss:.6f}')

    # # Plot contrastive learning loss curve
    # plt.figure()
    # plt.plot(cl_log['train_loss'])
    # plt.xlabel('Iterations')
    # plt.ylabel('BYOL Loss')
    # plt.title('Contrastive Learning Loss Curve')
    # cl_plot_path = os.path.join(dir_out, f'byol_loss_curve_{timestamp}.png')
    # plt.savefig(cl_plot_path)
    # logging.info(f"BYOL loss curve saved to: {cl_plot_path}")
    # logging.info("Contrastive learning training complete!")

    # # Step 4: Use the pretrained model for CDAN training
    # logging.info("Setting up CDAN with pretrained model...")
    pretrained_model = learner.online_encoder.net
    
    # /mnt/new_drive/Documents/for_Ye/CDAN/output/20250309_160040/pretrained_model_CL_299.pth
    pretrained_dict = torch.load('/mnt/new_drive/Documents/for_Ye/pretrained_model_CL_299.pth')
    model_dict = pretrained_model.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    # Load matching parameters
    model_dict.update(pretrained_dict)
    pretrained_model.load_state_dict(model_dict)
    CDAN_model = pretrained_model.to(device)
    CDAN_model = CDAN_model.train(True)

    # # Freeze encoder layers, only train classifier
    # for param in CDAN_model.parameters():
    #     param.requires_grad = True
    # for param in CDAN_model.classifier.parameters():
    #     param.requires_grad = False
    # for name, param in CDAN_model.named_parameters():
    #     if 'bottleneck' in name :
    #         param.requires_grad = False
    # for name, param in CDAN_model.named_parameters():
    #     if 'linear' in name :
    #         param.requires_grad = True
    # for param in CDAN_model.fc.parameters():
    #     param.requires_grad = False

    # Log which parameters are trainable
    logging.info("Trainable parameters:")
    for name, param in CDAN_model.named_parameters():
        if param.requires_grad:
            logging.info(f"Trainable: {name}")
    
    # Initialize domain classifier
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([CDAN_model.output_num(), config['class_num']], config["loss"]["random_dim"], device).to(device)
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 256).to(device)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(CDAN_model.output_num() * config['class_num'], 256).to(device)
    
    # Step 5: Train with CDAN
    best_model = train_CDAN(config, CDAN_model, ad_net, random_layer, data_loaders, device)
    return best_model


# def train_BYOL_CDAN(config, transformer_model, data_loaders, device):
#     """Initialize the encoder with BYOL weights, then fine-tune with CDAN."""
#     logging.info("Preparing BYOL+CDAN training...")
#     logging.info("Setting up BYOL learner...")
#     learner = BYOL(
#         transformer_model,
#         image_size=60,
#         hidden_layer='avgpool',
#         config=config,
#         F_config=F_config,
#     ).to(device)

#     pretrained_model = learner.online_encoder.net
#     load_matching_state_dict(pretrained_model, PRETRAINED_CL_CHECKPOINT, device)

#     cdan_model = pretrained_model.to(device).train(True)
#     logging.info("Trainable parameters:")
#     for name, param in cdan_model.named_parameters():
#         if param.requires_grad:
#             logging.info(f"Trainable: {name}")

#     random_layer, ad_net = build_domain_adversary(cdan_model, config, device)
#     return train_CDAN(config, cdan_model, ad_net, random_layer, data_loaders, device)


if __name__ == "__main__":
    # Parse command line arguments
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--dir_data', type=str, default=dir_data_2025, help="directory of data")
    parser.add_argument('--dir_out', type=str, default=dir_out, help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--run_baseline', type=bool, default=False, help="whether to run baseline model training")
    parser.add_argument('--run_cdan', type=bool, default=False, help="whether to run CDAN model training")
    parser.add_argument('--run_byol_cdan', type=bool, default=True, help="whether to run BYOL+CDAN model training")
    args, _ = parser.parse_known_args()
    
    logging.info(f"Arguments: {args}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device for training: {device}')
    
    # ================= Configuration Setup ================ #
    config = {
        "method": args.method,
        "N_seq": 60,
        "num_iterations": args.num_iterations,
        "epochs_CL": 100,
        "test_interval": args.test_interval,
        "dir_out": args.dir_out,
        "loss": {"trade_off": 1.0, "random": args.random, "random_dim": 512},
        "optimizer": {"type":optim.Adam, "optim_params":{'lr':args.lr, \
                            "weight_decay":0.0001}, "lr_type":"inv", \
                            "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}},
        "data": {"dir_data": args.dir_data,
                "source":{"name": ['df_dPCR_GB_2025.csv'], "batch_size":128},
                "target":{"name": "df_dPCR_SP_2025.csv", "batch_size":128},
                "test":{"name": "df_dPCR_SP_2025.csv", "batch_size":128},
                "CL": {"name": ["df_dPCR_SP_2025.csv"], "batch_size":2048}, 
                "normalize": "min_max"},
        "class_num": 7
    }

    # ================= Transformer Setup ================ #
    F_config = {
        'd_input': 1,
        'd_model': 128, # Lattent dim
        'q': 8, # Query size
        'v': 8,  # Value size
        'h': 4, # Number of self-attention heads
        'N': 4, # Number of encoder to stack
        'attention_size': None,  # Attention window size
        'dropout': 0.3, # drop out rate
        'chunk_mode': None,
        'pe': "regular",  # Positional encoding metric
        'pe_period': 20,
        'use_bottleneck': True,
        "bottleneck_dim": 256
    }

    # Log configurations
    logging.info("CDAN Configuration: " + str(config))
    logging.info("Transformer Configuration: " + str(F_config))

    torch.manual_seed(42)

    # ================= Data Preparation for CDAN ================ #
    logging.info("Loading data...")
    data_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    train_bt = data_config["target"]["batch_size"]
    test_b = data_config["test"]["batch_size"]
    dir_data = data_config["dir_data"]
    source_data_name = data_config["source"]["name"]
    target_data_name = data_config["target"]["name"]
    test_data_name = data_config["test"]["name"]

    data_loaders = generate_data_loader(dir_data, train_bs, train_bt, test_b, \
                                        source_data_name, target_data_name, test_data_name, \
                                            device, use_normalize = 'None')
    logging.info("Data loading complete!")

    # Define the activities for evaluation
    activities = ['Adeno', 'COVID', 'Cov_229E', 'Cov_HKU1', 'Cov_NL63', 'Cov_OC43', 'MERS']
    
    # Dictionary to store final results for comparison
    final_results = {}

    # ================= 1. Train Baseline Model ================ #
    if args.run_baseline:
        logging.info("\n\n=========== STARTING BASELINE MODEL TRAINING ===========")
        # Initialize a fresh transformer model for baseline
        baseline_transformer = Transformer_for_byol(config['class_num'], config['N_seq'], **F_config).to(device)
        
        # Train the baseline model (supervised learning on source domain only)
        baseline_best_model = train_baseline(config, baseline_transformer, data_loaders, device)
        
        # Evaluate the baseline model
        baseline_acc = eval_best_model(baseline_best_model, data_loaders, activities, 'Baseline', device, 'test')
        final_results['Baseline'] = baseline_acc
        logging.info(f"Baseline Model Final Accuracy: {baseline_acc:.2f}%")
    
    # ================= 2. Train CDAN Model ================ #
    if args.run_cdan:
        logging.info("\n\n=========== STARTING CDAN MODEL TRAINING ===========")
        # Initialize a fresh transformer model for CDAN
        cdan_transformer = Transformer_for_byol(config['class_num'], config['N_seq'], **F_config).to(device)
        
        # Initialize domain classifier
        random_layer, ad_net = build_domain_adversary(cdan_transformer, config, device)
        
        # Train the CDAN model
        cdan_best_model = train_CDAN(config, cdan_transformer, ad_net, random_layer, data_loaders, device)
        
        # Evaluate the CDAN model
        cdan_acc = eval_best_model(cdan_best_model, data_loaders, activities, 'CDAN', device, 'test')
        final_results['CDAN'] = cdan_acc
        logging.info(f"CDAN Model Final Accuracy: {cdan_acc:.2f}%")
    
    # ================= 3. Train BYOL+CDAN Model ================ #
    if args.run_byol_cdan:
        logging.info("\n\n=========== STARTING BYOL+CDAN MODEL TRAINING ===========")
        # Initialize a fresh transformer model for BYOL+CDAN
        byol_cdan_transformer = Transformer_for_byol(config['class_num'], config['N_seq'], **F_config).to(device)
        
        # Train with BYOL and then fine-tune with CDAN
        byol_cdan_best_model = train_BYOL_CDAN(config, byol_cdan_transformer, device)
        
        # Evaluate the BYOL+CDAN model
        byol_cdan_acc = eval_best_model(byol_cdan_best_model, data_loaders, activities, 'BYOL_CDAN', device, 'test')
        final_results['BYOL+CDAN'] = byol_cdan_acc
        logging.info(f"BYOL+CDAN Model Final Accuracy: {byol_cdan_acc:.2f}%")
    
    # ================= Final Comparison ================ #
    logging.info("\n\n=========== FINAL COMPARISON ===========")
    for method, accuracy in final_results.items():
        logging.info(f"{method} Accuracy: {accuracy:.2f}%")
    
    # Plot final comparison
    if len(final_results) > 1:
        plt.figure(figsize=(10, 6))
        plt.bar(final_results.keys(), final_results.values())
        plt.xlabel('Method')
        plt.ylabel('Accuracy (%)')
        plt.title('Comparison of Different Training Methods')
        plt.ylim([0, 100])
        for i, v in enumerate(final_results.values()):
            plt.text(i, v + 1, f"{v:.2f}%", ha='center')
        compare_path = os.path.join(dir_out, f"method_comparison_{timestamp}.png")
        plt.savefig(compare_path)
        logging.info(f"Comparison chart saved to: {compare_path}")
    
    logging.info("All training and evaluation complete!")