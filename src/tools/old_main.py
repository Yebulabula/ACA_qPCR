
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import torch
from dataset import PCRDataset
from torch.utils.data import random_split,DataLoader

if __name__ == '__main__':
    df  = pd.read_csv('df_dPCR_SP_bs_man_norm.csv', index_col="Unnamed: 0")
    df = df.filter(regex = r'\d+\.?\d*')
    n_pad_left=20
    n_pad_right=20
    
    curves = np.array([df.iloc[i,:].to_numpy() for i in range(df.shape[0])])
    
    dataset = PCRDataset(curves)
    
    
    # Splitting dataset into training and testing
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Creating the DataLoader for testing
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    epoch = 10
    
    for i in range(epoch):
        for i, (x, y) in enumerate(train_loader):
            print(x.shape)
            print(y.shape)
            breakpoint()
        