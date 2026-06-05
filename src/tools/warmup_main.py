import argparse
import warnings
import os
import os.path as osp
import numpy as np
import pandas as pd
from math import floor
import torch
import torch.optim as optim
from pretrain import train
from torch.autograd import Variable
from sklearn.metrics import classification_report
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

# class dataset(Dataset):
#     """
#         This class is used to encapsulate data and labels into a dataset.
#     """
#     def __init__(self, data, labels):
#         self.data = data
#         self.data = torch.unsqueeze(self.data, -1)
#         self.labels = labels
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, index):
#         return self.data[index], self.labels[index]
    
def eval_best_model(best_model, data_loaders, Activities, save_name): 
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

    for data, label in data_loaders['test']: 
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

    # Plot confusion matrix.
    # plot_cm(label_list.cpu().numpy(), pred_list.cpu().numpy(), Activities, save_name)
    # visualize(best_model, 'Transformer', dset_loaders["source"], dset_loaders["target"], save_name)
    print(classification_report(label_list.cpu().numpy(), pred_list.cpu().numpy(), digits = 5, target_names = Activities))

    # # Sample-level performance evaluation.
    # print("-------Sample-level performance evaluation-------")
    # length = floor(len(dset_loaders['target'].dataset) / len(dset_loaders['target'])) *  len(dset_loaders['target'])
    # df = pd.DataFrame(np.empty((length,3)), columns= ['true', 'pred', 'sampleid'])
    # df['true'] = label_list.cpu().numpy()
    # df['pred'] = pred_list.cpu().numpy()
    # df['sampleid'] = sample_list.cpu().numpy()
    # df = df.groupby(["sampleid", "true", "pred"])["pred"].count().reset_index(name="count")

    # sample_level_true_list, sample_level_pred_list = [], []
    # for i in df['sampleid'].unique():
    #     gdf = df[(df.sampleid == i)].sort_values('count', ascending =False)
    #     true, pred = gdf.iloc[0,1:3].values
    #     sample_level_true_list.append(true)
    #     sample_level_pred_list.append(pred)

    # # plot_cm(np.array(true_list), np.array(pred_list), Activities, save_name + '_sample_level')
    # # Print classification report, including accuracy, f1 score, precision, etc.
    # print(classification_report(np.array(sample_level_true_list), np.array(sample_level_pred_list), digits = 5, target_names = Activities))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--num_iterations', type=int, default=30)
    parser.add_argument('--test_interval', type=int, default=20, help="interval of two continuous test phase")
    parser.add_argument('--dir_data', type=str, default='7_plex', help="directory of data")
    parser.add_argument('--output_dir', type=str, default='log', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default= 1e-3, help="learning rate")
    parser.add_argument('--random', type=bool, default= False, help="whether use random projection")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    # device = 'cpu'
    print(f'Device for training: {device}')
    
    # ================= CDAN Framework Setup ================ #
    config = {
        "method": args.method,
        "num_iterations": args.num_iterations,
        "test_interval": args.test_interval,
        "output_path": "CDAN_ACA/" + args.output_dir,
        "loss": {"trade_off": 1.0, "random": args.random, "random_dim": 512},
        "optimizer": {"type":optim.Adam, "optim_params":{'lr':args.lr, \
                            "weight_decay":0.0001}, "lr_type":"inv", \
                            "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}},
        "data": {"dir_data": args.dir_data, 
                 "source":{"name": ["df_dPCR_GB_bs_man_norm.csv", 'df_qPCR_GB_bs_man_norm.csv'], "batch_size":128}, 
                 "target":{"name": "df_dPCR_SP_bs_man_norm.csv", "batch_size":128},
                 "test":{"name": "df_qPCR_SP_bs_man_norm.csv", "batch_size":32}},
        "class_num": 7
    }

    # config = {
    #     "method": 'CDAN+E',
    #     "num_iterations": 20,
    #     "test_interval": 20,
    #     "output_path": "CDAN_ACA/" + 'log',
    #     "loss": {"trade_off": 1.0, "random": False, "random_dim": 512},
    #     "optimizer": {"type":optim.Adam, "optim_params":{'lr':1e-3, \
    #                         "weight_decay":0.0001}, "lr_type":"inv", \
    #                         "lr_param":{"lr":1e-3, "gamma":0.001, "power":0.75}},
    #     "data": {"dir_data": dir_data, 
    #              "source":{"name": "df_dPCR_GB_bs_man_norm.csv", "batch_size":128}, 
    #              "target":{"name": "df_dPCR_SP_bs_man_norm.csv", "batch_size":128},
    #              "test":{"name": "df_qPCR_SP_bs_man_norm.csv", "batch_size":32}},
    #     "class_num": 7
    # }

    # ================= Feature Extractor Setup ================ #
    F_config = {
        'd_input': 1,
        'd_model': 16, # Lattent dim
        'q': 8, # Query size
        'v': 8,  # Value size
        'h': 4, # Number of self-attention heads
        'N': 4, # Nuember of encoder to stack
        'attention_size': 15,  # Attention wsindow size
        'dropout': 0.5, # drop out rate
        'chunk_mode': None,
        'pe': "regular",  # Positional encoding metric
        # 'batch_first': True,
        'pe_period': 45
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
                                                # 60, True, 
                                                device)    
    
    # print(source_X_df.shape[0], target_X_df.shape[0])

    # Initialize feature extractor (generator) model.
    Generator = Transformer(config['class_num'], **F_config).to(device)

    
    # ================= Training Procedure ================ #
    _, best_model = train(config, Generator, data_loaders, device)    

    # ================= Evaluating Procedure ================ # 
    best_model = torch.load('{0}/TST_ACA_DA_Generator.pth'.format('CDAN_ACA/model'), map_location=torch.device(device)) 
    best_model.eval() # switch to test mode.
    
    # # imp' -> 0, 'ndm' -> 1, 'kpc' -> 2
    # Activities =  [r'$bla_{IMP}$',r'$bla_{NDM}$', r'$bla_{OXA48}$']
    Activities = ['Adeno', 'Cov_229E', 'Cov_HKU1', 'Cov_NL63', 'Cov_OC43', 'MERS', 'COVID']

    eval_best_model(best_model, data_loaders, Activities, 'CDAN')

    
