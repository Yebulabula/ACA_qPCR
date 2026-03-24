# Load general libraries
import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
# import seaborn as sns
import math
import warnings
import matplotlib.pyplot as plt
import re
from itertools import product
import itertools
import time
import matplotlib as mpl
import scipy.optimize as opt
import tqdm
from cycler import cycler

import tst
from tst.encoder import Encoder
from tst.utils import generate_original_PE, generate_regular_PE
from sklearn.metrics import roc_auc_score, roc_curve, auc
mpl.rcParams["axes.prop_cycle"] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

## Load ML libraries
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, StratifiedGroupKFold, GroupKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.decomposition import PCA
from sklearn import preprocessing as preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

## Load PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

## Set directories
# os.chdir('../DEEP_ACA_7plex')
dir_data = '../data_2025'
dir_out = '../THESIS_2025'
dir_scr = '../7plex_RESPI_ANI/DEEP_ACA_7plex/scr'
targets = ['Adeno', 'Cov_229E', 'Cov_HKU1', 'Cov_NL63', 'Cov_OC43', 'MERS', 'COVID', 'ctrl_1', 'ctrl_2']
targets_7plex = ['Adeno', 'Cov_229E', 'Cov_HKU1', 'Cov_NL63', 'Cov_OC43', 'MERS', 'COVID']
color_num = pd.Series(targets_7plex)


### functions ofs models
##########################################################################

# dict_model_in = f_create_dict_DL_models_in()
# dict_data = {}
# df_dPCR = pd.read_csv(os.path.join(dir_data, 'unfiltered_dpcr.csv'))
# df_dPCR = prepare_dataset(df_dPCR)
# df_dPCR = removeNTCs(df_dPCR)
# df_dPCR = df_dPCR.sample(frac=0.01)
# dict_data['df_dPCR'] = df_dPCR
# dict_data['df_tmp_DL'] = df_tmp_DL.sample(frac=0.01)
# dict_data_in = dict_data

def flatten_list_of_tensors(lst):
    flat_lst = []
    for tsr in lst:
        flat_lst.append(tsr.tolist())
    flat_lst = list(itertools.chain.from_iterable(flat_lst))
    return flat_lst

def apply_min_max_normalization_TR_TE(X_train: np.ndarray, X_test: np.ndarray, t_min: float = 0, t_max: float = 1):
    scaler = MinMaxScaler(feature_range=(t_min,t_max))
    scaler.fit(X_train)
    X_train_ts = scaler.transform(X_train)
    X_test_ts = scaler.transform(X_test)
    return X_train_ts, X_test_ts
                                      
def apply_standard_scaler_normalization_TR_TE(X_train: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_ts = scaler.transform(X_train)
    X_test_ts = scaler.transform(X_test)
    return X_train_ts, X_test_ts

def expand_grid(dictionary):
    if len([row for row in product(*dictionary.values())]) == 1:
        return pd.DataFrame([dictionary.values()], 
                        columns = dictionary.keys())
    else:
        return pd.DataFrame([row for row in product(*dictionary.values())], 
                        columns = dictionary.keys())

def f_extract_score(clf, X_test):
    if hasattr(clf, 'predict_proba'):
        score = clf.predict_proba(X_test)
    elif hasattr(clf, 'decision_function'):
        score = clf.decision_function(X_test)
    return(score)

def f_CV_accuracy(df):
    CV_fold = []; accuracy = []
    for i, df_ in df.groupby('group_CV'):
        CV_fold.append(i)
        acc_tmp = np.divide(sum(df_.pred == df_.truth), df_.shape[0])
        accuracy.append(acc_tmp)
    res = pd.DataFrame({'CV_fold':CV_fold, 'accuracy': accuracy})
    return(res)

def f_data_loader(dict_x):
    X = dict_x['X']
    Y = dict_x['Y']
    groups = dict_x['groups']
    return(X, Y, groups)


class MLP_3(nn.Module):

    def __init__(self, input_size, hidden_size, n_hidden_layers, output_size, batch_norm=True, dropout_prob=0.3, learning_rate=0.01):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout_prob = dropout_prob
        if dropout_prob>0:
            self.Dropout = nn.Dropout(p=dropout_prob)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.a2 = nn.ReLU()
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_hidden_layers-1)])
        self.out_l = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate
        # self.a3 = nn.ReLU()
    
    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, input):
        tmp = self.l1(input)
        if self.batch_norm: 
            tmp = self.bn1(tmp)
        tmp = self.a1(tmp)
        tmp = self.l2(tmp)
        if self.dropout_prob>0:
            tmp = self.Dropout(tmp)
        if self.batch_norm:
            tmp = self.bn2(tmp)
        tmp = self.a2(tmp)
        for hidden_layer in self.hidden_layers:
            tmp = torch.relu(hidden_layer(tmp))
        output = self.out_l(tmp)
        # output = self.a3(tmp)
        # output = nn.Softmax(dim = 1)(tmp)
        return output

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1,
                 batch_norm=True, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add seq_len=1 for LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take output at last time step
        if self.batch_norm:
            out = self.bn(out)
        out = self.dropout(out)
        return self.fc(out)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1,
                 batch_norm=True, dropout_prob=0.3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add seq_len=1 for GRU
        out, _ = self.gru(x)
        out = out[:, -1, :]
        if self.batch_norm:
            out = self.bn(out)
        out = self.dropout(out)
        return self.fc(out)


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
                 chunk_mode: str = None,
                 pe: str = None,
                 pe_period: int = 20,
                 use_bottleneck=True, 
                 bottleneck_dim=256):
                 
        super(Transformer, self).__init__()
        self._d_model = d_model
        # self._batch_first = batch_first 
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
            
        x = torch.relu(encoding)
        
        # Classification on source domain data
        x = self.classifier(x)

        if self.use_bottleneck:
            x = self.bottleneck(x)

        y = self.fc(x)
        return y

    def output_num(self):
        return self.__in_features


def f_create_dict_DL_models_in():

    dict_models_in = {}
    
    dict_models_in['MLP'] = {
        'clf_factory': lambda hyperparam: MLP_3(input_size=hyperparam['input_size'], 
                                                hidden_size=hyperparam['hidden_size'], 
                                                n_hidden_layers=hyperparam['n_hidden_layers'],
                                                output_size=hyperparam['output_size'], 
                                                batch_norm=hyperparam['batch_norm'],
                                                dropout_prob=hyperparam['dropout_prob'],
                                                learning_rate=hyperparam['learning_rate']),
        
        'hyperparam': {
            'input_size': [60],
            'hidden_size': [64, 128],
            'n_hidden_layers': [2, 4, 8],
            'output_size': [7],
            'batch_norm': [True, False],
            'dropout_prob': [0, 0.3],
            'learning_rate': [0.01, 0.001],
        }
    }

    dict_models_in['LSTM'] = {
        'clf_factory': lambda hyperparam: LSTMModel(
            input_dim=int(hyperparam['input_dim']),
            hidden_dim=int(hyperparam['hidden_dim']),
            output_dim=int(hyperparam['output_dim']),
            num_layers=int(hyperparam.get('num_layers', 1)),
            dropout_prob=hyperparam.get('dropout_prob', 0.3)
        ),
        'hyperparam': {
            'input_dim': [60],
            'hidden_dim': [64, 128],
            'output_dim': [7],
            'num_layers': [1, 2],
            'dropout_prob': [0, 0.3],
            'learning_rate': [0.01, 0.001],
            'batch_norm': [True, False]
        }
    }

    dict_models_in['GRU'] = {
        'clf_factory': lambda hyperparam: GRUModel(
            input_dim=int(hyperparam['input_dim']),
            hidden_dim=int(hyperparam['hidden_dim']),
            output_dim=int(hyperparam['output_dim']),
            num_layers=int(hyperparam['num_layers']),
            dropout_prob=hyperparam['dropout_prob'],
            batch_norm=hyperparam['batch_norm']
        ),
        'hyperparam': {
            'input_dim': [60],
            'hidden_dim': [64, 128],
            'output_dim': [7],
            'num_layers': [1, 2],
            'dropout_prob': [0, 0.3],
            'learning_rate': [0.01, 0.001],
            'batch_norm': [True, False]
        }
    }

    dict_models_in['Transformer'] = {
        'clf_factory': lambda hyperparam: Transformer(
            class_num=hyperparam['output_dim'],
            d_input=hyperparam['input_dim'],
            d_model=hyperparam['d_model'],
            q=hyperparam['q'],
            v=hyperparam['v'],
            h=hyperparam['h'],
            N=hyperparam['N'],
            dropout=hyperparam['dropout'],
            pe=hyperparam['pe'],
            use_bottleneck=hyperparam['use_bottleneck'],
            bottleneck_dim=hyperparam['bottleneck_dim']
        ),
        'hyperparam': {
            'input_dim': [1],  # ACs shape: (batch, seq_len, input_dim)
            'output_dim': [7],
            'd_model': [128, 256, 512],
            'q': [8],
            'v': [8],
            'h': [4, 8],
            'N': [2, 4],  # number of encoder layers
            'dropout': [0.3],
            'pe': [None, 'original'],
            'use_bottleneck': [True],
            'bottleneck_dim': [128, 256],
            'learning_rate': [0.01, 0.001]
        }
    }
    return(dict_models_in)

           
def plot_loss_acc(dict_training, with_test=True):

    loss_train_hist = dict_training['loss_train_hist']
    accuracy_train_hist = dict_training['accuracy_train_hist']
    if with_test:
        loss_test_hist = dict_training['loss_test_hist']
        accuracy_test_hist = dict_training['accuracy_test_hist']
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 8), dpi = 300)
    # fig, ax = plt.figure((1,2), figsize=(12, 5))
    ax[0].plot(loss_train_hist, lw=3, color = 'blue')
    ax[0].set_title('Loss', size=15)
    ax[0].set_xlabel('Epoch', size=15)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[1].plot(accuracy_train_hist, lw=3, color = 'blue')
    ax[1].set_title('Accuracy', size=15)
    ax[1].set_xlabel('Epoch', size=15)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    if with_test:
        ax[0].plot(loss_test_hist, lw=3, color = 'orange')
        ax[1].plot(accuracy_test_hist, lw=3, color = 'orange')
    plt.show()


def f_save_output_to_disc_DL(dir_out, data_in, dict_models_out):

    ## Create folder
    import datetime
    now = datetime.datetime.now()
    output_folder = now.strftime("%Y-%m-%d_%H-%M-%S") + '_' + data_in
    os.mkdir(os.path.join(dir_out, output_folder))
    
    ## Create sub-folders and save data frames
    for model in dict_models_out.keys():
        output_folder_model = os.path.join(dir_out, output_folder, model)
        os.mkdir(output_folder_model)
        pd.DataFrame(dict_models_out[model]['best_class_report']).to_csv(os.path.join(output_folder_model, 'best_class_report.csv'))
        pd.DataFrame(dict_models_out[model]['accuracy']).to_csv(os.path.join(output_folder_model, 'accuracy.csv'))
        # pd.DataFrame(dict_models_out[model]['best_accuracy_test']).to_csv(os.path.join(output_folder_model, 'best_accuracy_test.csv'))

class cl_as_tensor(object):

    def __init__(self, dtype, device):
        self.device = device
        self.dtype = dtype            

    def __call__(self, x):
        if type(x) != np.ndarray:
            x = np.asarray(x)
        t_x = torch.from_numpy(x).to(self.device, dtype = self.dtype)
        return t_x

def f_as_tensor(x, dtype, device):
    if type(x) != np.ndarray:
            x = np.asarray(x)
    t_x = torch.from_numpy(x).to(device, dtype = dtype)
    return t_x


class dataset(Dataset):
    
    def __init__(self, X, Y, transform, dtype, device):
        self.X = X
        self.y = torch.from_numpy(Y).to(torch.int64)   
        # self.y = torch.nn.functional.one_hot(Target, num_classes = 7)
        self.transform = transform
        self.device = device
        self.dtype = dtype
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        if self.transform == None:
            return self.X[index], self.y[index]
        else:
            inst_transform = self.transform(self.dtype, self.device)
            t_X = inst_transform(self.X)
            t_y = inst_transform(self.y)
            return t_X[index], t_y[index]


def f_run_DL_models(dir_out, dict_data_in, normalization, 
                    device='cuda', batch_size=128, num_epochs=2, count_time=True,
                    save_output_to_disc=True, print_conf_mat=False, return_model=True):

    warnings.filterwarnings("ignore")
    
    best_models = {}

    dict_data_in = prepare_dataset_for_DL_models(dict_data_in)

    for i, data_in in enumerate(dict_data_in):
        print('Fitting model', i+1, 'over', len(list(dict_data_in.keys())) )
        print('Input data =', data_in)

        X, Y, groups = f_data_loader(dict_data_in[data_in])
        groups = np.asarray(groups)

        # create dictonnary of models
        dict_models_in = f_create_dict_DL_models_in()

        # train + test all models on all data sets
        dict_models_out = f_fit_DL_models(dict_models_in, X, Y, groups, normalization, 
                                          num_epochs=num_epochs, count_time=True,
                                          device=device, print_conf_mat=False)

        if save_output_to_disc:
            f_save_output_to_disc_DL(dir_out, data_in, dict_models_out)

        for model in dict_models_out:
            print('Best training accuracy with model', model, 'is', 
                    np.round(np.max(dict_models_out[model]['best_accuracy']),3))
            
        best_models[data_in] = dict_models_out

        print('End of fitting for model', i+1)        
        print('.......')

    if return_model == True:
        best_global = f_return_best_model(best_models, dict_models_in)
        return(best_global)
    
def f_return_best_model(best_models, dict_models_in):
    best_global = {}
    accuracy_df = []
    hyperparam = {}
    for data_in in best_models.keys():
        for model in best_models[data_in].keys():
            num = best_models[data_in][model]['accuracy'].shape[0]
            accuracy_df.append(pd.concat([pd.Series(np.repeat(data_in, num), name = 'data_in'),
                                        pd.Series(np.repeat(model, num), name = 'model'),
                                        best_models[data_in][model]['accuracy']['accuracy']], axis = 1))
    accuracy_df = pd.concat(accuracy_df, axis = 0)
    best_idx = np.argmax(accuracy_df.accuracy)
    best_data_in = accuracy_df.iloc[best_idx, :].data_in
    best_model = accuracy_df.iloc[best_idx, :].model
    for param_key in list(dict_models_in[best_model]['hyperparam']):
        tmp = best_models[best_data_in][best_model]['accuracy']
        tmp = tmp.iloc[np.argmax(tmp.accuracy),:]
        hyperparam[param_key] = [tmp[param_key]]
    best_global['data_in'] = best_data_in
    best_global[best_model] = {
        'clf_factory': dict_models_in[best_model]['clf_factory'],
        'hyperparam': hyperparam
    }
    return(best_global)

def f_fit_DL_models(dict_models_in, X, Y, groups, normalization, 
                    num_epochs=100, count_time=True, batch_size=128, 
                    device='cuda', print_conf_mat=False):

    dict_models_out = {}
    
    for model in dict_models_in:
        model_out = {}
        model_out['class_report'] = {}
        model_out['best_class_report'] = {}
        model_out['best_accuracy'] = {}
        # model_out['class_report_test'] = {}

        # Create and save hyperparam_df
        hyperparam_df = expand_grid(dict_models_in[model]['hyperparam'])
        model_out['param'] = hyperparam_df

        ## Fit
        print('Training + testing', model)
        for i in hyperparam_df.index:
            print("param", i+1, "over", len(hyperparam_df.index))
            
            ## Define model
            hyperparam_row = hyperparam_df.iloc[i, :]
            NN_clf = dict_models_in[model]['clf_factory'](hyperparam_row)
            NN_clf.to(device)
            
            ## Fit and record predictions + score
            gkf = GroupKFold(n_splits = 5, shuffle = True, random_state=42)
            for j, (train, test) in enumerate(gkf.split(X, Y, groups=groups)):
                print("CV fold", j+1, "over", gkf.n_splits)
                X_train = X[train,:]
                Y_train = Y[train]
                X_test = X[test,:]
                Y_test = Y[test]

                # print("CV fold", j+1, "over", gkf.n_splits)
                if normalization == 'minmax':
                    X_train, X_test = apply_min_max_normalization_TR_TE(X_train, X_test)
                elif normalization == 'standard':
                    X_train, X_test = apply_standard_scaler_normalization_TR_TE(X_train, X_test)
                elif normalization == 'None':
                    X_train = X[train,:]
                    X_test = X[test,:]

                data_train = dataset(X_train, Y_train, transform=cl_as_tensor, device=device, dtype=torch.float32)
                data_test = dataset(X_test, Y_test, transform=cl_as_tensor, device=device, dtype=torch.float32)
                train_dataloader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
                test_dataloader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)

                dict_training_return = training_testing_LK(NN_clf, train_dataloader, test_dataloader, 
                                                           num_epochs=num_epochs, lr=hyperparam_row['learning_rate'],
                                                           count_time=count_time, 
                                                           device=device, print_conf_mat=print_conf_mat)
                # class_report_train = classification_report(y_true=dict_training_return['y_train_true'], y_pred=dict_training_return['y_train_pred'], target_names=classes, digits=4, output_dict=True)
                df_classes = pd.read_csv(os.path.join(dir_data, 'df_classes.csv'), index_col='Unnamed: 0')
                classes = df_classes.sort_values('Target_cat').Target.tolist()
                class_report = classification_report(y_true=dict_training_return['y_test_true'], y_pred=dict_training_return['y_test_pred'], target_names=classes, digits=4, output_dict=True)
                model_out['class_report'][i] = class_report
            
        ## Mean accuracy for each hyperparameter_row
        accuracy = []
        for key in list(model_out['class_report'].keys()):
            accuracy.append(model_out['class_report'][key]['accuracy'])
        model_out['accuracy'] = pd.concat([hyperparam_df, pd.Series(accuracy, index = np.arange(len(accuracy)), name = 'accuracy')], axis=1)
        
        ## Best model
        best_mod = np.argmax(accuracy)
        
        ## Save classification report for best model and CV_accuracy
        model_out['best_class_report'] = model_out['class_report'][best_mod]
        model_out['best_accuracy'] = model_out['accuracy'].loc[best_mod,:]['accuracy']

        dict_models_out[model] = model_out

    return(dict_models_out)

def print_conf_mat_DL(y_true, y_pred, title='Confusion Matrix'):
    from sklearn.metrics import ConfusionMatrixDisplay
    df_classes = pd.read_csv(os.path.join(dir_data, 'df_classes.csv'), index_col='Unnamed: 0')
    classes = df_classes.sort_values('Target_cat').Target.tolist()
    labels = sorted(list(set(y_true)))
    classes_final = [classes[idx] for idx in labels]
    ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, 
                                                    labels = labels,
                                                    display_labels=classes_final,
                                                    include_values=True,
                                                    xticks_rotation=45)
    plt.grid(False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def prepare_dataset_for_DL_models(dict_data):
    
    dict_data_return = {}
    for key in dict_data.keys():
        df = dict_data[key]

        if key == "df_AC":
            dict_data_return[key] = {
                'groups': df.Exp_ID_Channel,
                'X': np.array(df.filter(regex = r'\d+\.?\d*')),
                'Y': np.array(df.Target_cat)
                # 'X_test': np.array(df.filter(regex = r'\d+\.?\d*')),
                # 'Y_test': np.array(df.Target)
            }

    return dict_data_return

def training_testing_LK(model, train_dataloader, test_dataloader, 
                    device, num_epochs=2, lr=0.001,
                    count_time=True, print_conf_mat=False, print_loss_acc=False):

    loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = num_epochs

    loss_train_hist = [0] * num_epochs
    loss_test_hist = [0] * num_epochs
    accuracy_train_hist = [0] * num_epochs
    accuracy_test_hist = [0] * num_epochs

    print('Training on device:')
    print(device)

    start_time_tot = time.time()

    for epoch in range(num_epochs):

        start_time_epoch = time.time()

        print('**********************************************')
        print('epoch', epoch)
        print('**********************************************')
        
        ## TRAINING
        model.train()
        y_train_pred = []
        y_train_true = []
        for i, (x_train_batch, y_train_batch) in enumerate(tqdm.tqdm(train_dataloader)):
            if x_train_batch.shape[0]>1:
                x_train_batch = x_train_batch.to(device)
                y_train = y_train_batch.type(torch.LongTensor).to(device)
                
                y_train_hat = model(x_train_batch)
                if isinstance(y_train_hat, tuple):
                    y_train_hat = y_train_hat[0]  # Use the first element as predictions
                y_train_pred.append(torch.argmax(y_train_hat, dim=1))
                y_train_true.append(y_train)
                loss_train = loss_fn(y_train_hat, y_train)
                loss_train.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_train_hist[epoch] += loss_train.item()
                is_correct = (torch.argmax(y_train_hat, dim=1) == y_train).float()
                accuracy_train_hist[epoch] += is_correct.sum().item()
                
        ## TESTING
        model.eval()
        y_test_pred = []
        y_test_true = []
        with torch.no_grad():  # No gradient calculation for testing
            for j, (x_test_batch, y_test_batch) in enumerate(tqdm.tqdm(test_dataloader)):
                if x_test_batch.shape[0] > 1:
                    x_test_batch = x_test_batch.to(device)
                    y_test = y_test_batch.type(torch.LongTensor).to(device)
                    y_test_hat = model(x_test_batch)
                    if isinstance(y_test_hat, tuple):
                        y_test_hat = y_test_hat[0]
                    y_test_pred.append(torch.argmax(y_test_hat, dim=1))
                    y_test_true.append(y_test)
                    loss_test = loss_fn(y_test_hat, y_test)
                    loss_test_hist[epoch] += loss_test.item()
                    is_correct = (torch.argmax(y_test_hat, dim=1) == y_test).float()
                    accuracy_test_hist[epoch] += is_correct.sum().item()
        
        loss_train_hist[epoch] = np.divide(loss_train_hist[epoch],i)
        loss_test_hist[epoch] = np.divide(loss_train_hist[epoch],j)
        accuracy_train_hist[epoch] = np.divide(accuracy_train_hist[epoch], len(train_dataloader.dataset))
        accuracy_test_hist[epoch] = np.divide(accuracy_test_hist[epoch], len(test_dataloader.dataset))
        
        print('loss_train_hist')
        print(loss_train_hist[epoch])
        print('loss_test_hist')
        print(loss_test_hist[epoch])
        print('accuracy_train_hist')
        print(accuracy_train_hist[epoch])   
        print('accuracy_test_hist') 
        print(accuracy_test_hist[epoch])

        end_time_epoch = time.time()
        duration_epoch = end_time_epoch - start_time_epoch
        if count_time:
            print('Duration of training for epoch', epoch, '=', round(duration_epoch), 'sec.')

        # Conf mat
        # train
        y_train_true = flatten_list_of_tensors(y_train_true)
        y_train_pred = flatten_list_of_tensors(y_train_pred)
        cm_train = confusion_matrix(y_train_true, y_train_pred)
        # test
        y_test_true = flatten_list_of_tensors(y_test_true)
        y_test_pred = flatten_list_of_tensors(y_test_pred)
        cm_test = confusion_matrix(y_test_true, y_test_pred)
        if print_conf_mat:
            print_conf_mat_DL(y_train_true, y_train_pred, title='Confusion matrix for training dataset, epoch=' + str(epoch+1))
            print_conf_mat_DL(y_test_true, y_test_pred, title='Confusion matrix for testing dataset, epoch=' + str(epoch+1))

        # Create return dictionnary
        dict_training_return = {}
        dict_training_return['model_trained'] = model
        dict_training_return['loss_train_hist'] = loss_train_hist
        dict_training_return['loss_test_hist'] = loss_test_hist
        dict_training_return['accuracy_train_hist'] = accuracy_train_hist
        dict_training_return['accuracy_test_hist'] = accuracy_test_hist
        dict_training_return['y_train_true'] = y_train_true
        dict_training_return['y_train_pred'] = y_train_pred
        dict_training_return['y_test_true'] = y_test_true
        dict_training_return['y_test_pred'] = y_test_pred
        dict_training_return['cm_train'] = cm_train
        dict_training_return['cm_test'] = cm_test

        if print_loss_acc:
            plot_loss_acc(dict_training_return)#, epoch)

    end_time_tot = time.time()
    duration_tot = end_time_tot - start_time_tot
    if count_time:
        print('Total duration of training =', round(duration_tot/60), 'min.')

    return dict_training_return



### load datasets
##########################################################################

if __name__ == "__main__":

    df_dPCR_GB = pd.read_csv(os.path.join(dir_data, "df_dPCR_GB_2025.csv"))
    df_AC = df_dPCR_GB

    # Check for metadata columns
    numeric_cols = [col for col in df_dPCR_GB.columns if col.replace('.', '').isdigit()]
    metadata_cols = [col for col in df_dPCR_GB.columns if col not in numeric_cols]

    # df_AC_minmax = apply_min_max_normalization_TR_TE(df_AC)
    # df_AC_stdscaler = apply_standard_scaler_normalization_TR_TE(df_AC)

    # df_4param = pd.read_csv(os.path.join(dir_data, "param_df_4_dPCR_GB.csv"), index_col='Unnamed: 0')
    # df_4param[metadata_cols] = df_AC[metadata_cols]
    # df_5param = pd.read_csv(os.path.join(dir_data, "param_df_5_dPCR_GB.csv"), index_col='Unnamed: 0')
    # df_5param[metadata_cols] = df_AC[metadata_cols]
    # df_6param = pd.read_csv(os.path.join(dir_data, "param_df_6_dPCR_GB.csv"), index_col='Unnamed: 0')
    # df_6param[metadata_cols] = df_AC[metadata_cols]

    dict_data_in = {
        'df_AC': df_dPCR_GB,
        # 'df_AC_minmax': df_AC_minmax,
        # 'df_AC_stdscaler': df_AC_stdscaler,
        # 'df_4param': df_4param,
        # 'df_5param': df_5param,
        # 'df_6param': df_6param
    }

    best_global = f_run_DL_models(dir_out, 
                                   dict_data_in, 
                                   normalization='None',
                                   save_output_to_disc=True, 
                                   print_conf_mat=False, 
                                   return_model=True)

    best_global = f_run_DL_models(dir_out, 
                                   dict_data_in, 
                                   normalization='minmax',
                                   save_output_to_disc=True, 
                                   print_conf_mat=False, 
                                   return_model=True)
    
    best_global = f_run_DL_models(dir_out, 
                                   dict_data_in, 
                                   normalization='standard',
                                   save_output_to_disc=True, 
                                   print_conf_mat=False, 
                                   return_model=True)