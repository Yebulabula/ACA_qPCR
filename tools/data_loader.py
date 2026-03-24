import torch 
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader

def f_as_tensor(x, dtype, device):
    if type(x) != np.ndarray:
            x = np.asarray(x)
    t_x = torch.from_numpy(x).to(device, dtype = dtype)
    return t_x

class cl_as_tensor(object):

    def __init__(self, dtype, device):
        self.device = device
        self.dtype = dtype            

    def __call__(self, x):
        if type(x) != np.ndarray:
            x = np.asarray(x)
        
        t_x = torch.from_numpy(x).to(self.device, dtype = self.dtype)
        return t_x

def sigmoid5_spline_lin(x, Fm, Fb, Sc, Cs, As, a, b):
    """5-parameter sigmoid model for PCR amplification curves with linear spline"""
    # Use numpy.maximum to handle element-wise comparison correctly
    sigmoid_part = Fm / (1. + np.exp(-(x-Cs)*Sc))**As + Fb
    linear_part = np.maximum(0, a*x + b)
    return sigmoid_part + linear_part
 
 
def augment_fct(params_df, 
                idx,
                sigmoid_fn=sigmoid5_spline_lin, 
                plot = True):
    param_cols = ['Fm', 'Fb', 'Sc', 'Cs', 'As', 'a', 'b']
    params = params_df.loc[idx, param_cols]
    
    param_to_change = str('Cs')
    params_aug = params.copy()
    
    mean = 24.835403682791412
    sigma = 7.098645591811588
    dist = np.random.normal(loc=mean, scale=sigma, size=len(params_df))

    lower_bound = np.percentile(dist, 0)
    upper_bound = np.percentile(dist, 100)
    values_to_sample = params_df[param_to_change][(params_df[param_to_change] > lower_bound) & (params_df[param_to_change] < upper_bound)]
    params_aug[param_to_change] = np.random.choice(values_to_sample)
    x = np.arange(1, 50 + 1)
    y_aug = sigmoid_fn(x, *params_aug)

    return y_aug

class dataset(Dataset):
    
    def __init__(self, X, Y, transform, dtype, device):
        self.X = X.to_numpy()
        
        self.y = torch.from_numpy(Y.to_numpy())
        self.transform = transform
        self.device = device
        self.dtype = dtype
        self.X = torch.from_numpy(self.X).to(dtype = self.dtype)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    
# class aug_dataset(Dataset):
#     def __init__(self, X, Y, params_df, transform, dtype, device):
#         self.X = X.to_numpy()
#         self.params_df = pd.read_csv(params_df, index_col=0).dropna()
#         self.y = torch.from_numpy(Y.to_numpy())
#         self.transform = transform
#         self.device = device
#         self.dtype = dtype
#         self.X = torch.from_numpy(self.X).to(dtype = self.dtype)
    
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, index):
#         # plot the data
        
#         c1 = augment_fct(self.params_df, 
#             index,
#             sigmoid_fn=sigmoid5_un, 
#             plot = False)
        
#         c1 = torch.from_numpy(c1).to(dtype = self.dtype)
#         return c1, self.y[index]


def generate_data_loader(dir_data, 
                         train_bs, train_bt, test_b, 
                         source_data_name, target_data_name, test_data_name, 
                         device,
                         use_normalize = 'mix_max' #'mix_max', 
                         ):
    """
        This function is used to create dataloader with the given batch size for source and target data.

        INPUT:
        config: framework configuration.
        source_data: synthetic DNA dataframes.
        target_data: clinical isolates dataframes.
        device: check if using cpu/cuda.

        RETURN:
        dset_loader: A dictionary which stores source domain, target domain and testing dataloaders.
    """

    # dir_data = 'C:\\Users\91001010\\Doc_LK\\Imperial_College\\PhD_PROJECTS\DEEP_ACA_7plex\\data\\new_data'
    # device = 'cpu'

    # empty dataframe
    source_X_df = pd.DataFrame()
    source_Y_df = pd.DataFrame()
    source_df = pd.DataFrame()
    for f in source_data_name:
        src_df = pd.read_csv(os.path.join(dir_data, f)) #,index_col = 'Unnamed: 0').dropna()
        print(src_df.head)
        
        src_X_df = src_df.filter(regex = r'\d+\.?\d*')
        src_Y_df = src_df.Target_cat
        source_X_df = pd.concat([source_X_df, src_X_df], axis = 0)
        source_Y_df = pd.concat([source_Y_df, src_Y_df], axis = 0)
        source_df = pd.concat([source_df, src_df], axis = 0)
        
    source_X_df = source_X_df.dropna(axis=1) 
    source_df = pd.concat([pd.read_csv(os.path.join(dir_data, f)) for f in source_data_name], axis=0)
    target_df = pd.read_csv(os.path.join(dir_data, target_data_name)).dropna() #, index_col='Unnamed: 0')
    test_df = pd.read_csv(os.path.join(dir_data, test_data_name)).dropna()
    
    source_X_df = source_df.filter(regex = r'\d+\.?\d*')
    source_Y_df = source_df.Target_cat.astype(int)
    target_X_df = target_df.filter(regex = r'\d+\.?\d*')
    target_Y_df = target_df.Target_cat.astype(int)
    test_X_df = test_df.filter(regex = r'\d+\.?\d*')
    test_Y_df = test_df.Target_cat.astype(int)
    import matplotlib.pyplot as plt
    if use_normalize == 'min_max':
        scaler = MinMaxScaler()
        source_X_df = pd.DataFrame(scaler.fit_transform(source_X_df), columns=source_X_df.columns)
        target_X_df = pd.DataFrame(scaler.transform(target_X_df), columns=target_X_df.columns)
        test_X_df = pd.DataFrame(scaler.transform(test_X_df), columns=test_X_df.columns)

    elif use_normalize == 'standard':
        scaler = StandardScaler()
        source_X_df = pd.DataFrame(scaler.fit_transform(source_X_df), columns=source_X_df.columns)
        target_X_df = pd.DataFrame(scaler.transform(target_X_df), columns=target_X_df.columns)
    
        test_X_df = pd.DataFrame(scaler.transform(test_X_df), columns=test_X_df.columns)

    elif use_normalize == 'None':
        # scaler = StandardScaler()
        source_X_df = source_X_df
        target_X_df = target_X_df
        test_X_df = test_X_df

    else:
        raise ValueError("Unsupported normalization method. Use 'min_max' or 'standard'.")

    source_data = dataset(source_X_df, source_Y_df, transform=cl_as_tensor, device=device, dtype=torch.float32)
    target_data = dataset(target_X_df, target_Y_df, transform=cl_as_tensor, device=device, dtype=torch.float32)
    test_data = dataset(test_X_df, test_Y_df, transform=cl_as_tensor, device=device, dtype=torch.float32)

    data_loaders = {
        "source": DataLoader(source_data, batch_size=train_bs, shuffle=True, drop_last=True),
        "target": DataLoader(target_data, batch_size=train_bt, shuffle=True, drop_last=True),
        "test": DataLoader(test_data, batch_size=test_b, shuffle=True)
    }

    # Print dataset lengths for verification
    print("Lengths - Source:", len(source_data), "Target:", len(target_data), "Test:", len(test_data))

    return data_loaders









