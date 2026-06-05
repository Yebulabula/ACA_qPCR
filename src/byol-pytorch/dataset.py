from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def augment_curve(curve, n_pad_before, n_pad_after):
    curve_padded_before = np.repeat(curve[0], n_pad_before)
    curve_padded_after = np.repeat(curve[-1], n_pad_after)
    curve_padded = np.concatenate((curve_padded_before, curve, curve_padded_after))
    
    rand_shift = random.randint(10, 30)
    col_to_keep = np.arange(rand_shift, rand_shift+60, 1)
    curve_shifted = curve_padded[col_to_keep]
    return curve_shifted

def standardize_curve(curve):
    return (curve - np.min(curve)) / (np.max(curve) - np.min(curve))

class PCRDataset(Dataset):

    def __init__(self, raw_curves, n_pad_before, n_pad_after, normalize = 'min_max'):
        self.raw_curves = raw_curves
        self.n_pad_before=n_pad_before
        self.n_pad_after=n_pad_after

        self.augmented_curves = np.empty((self.raw_curves.shape[0], 60))
        for i in range(self.raw_curves.shape[0]):
            self.augmented_curves[i] = augment_curve(self.raw_curves[i], self.n_pad_before, self.n_pad_after)

        if normalize == 'min_max':
            scaler = MinMaxScaler()
            self.scaled_raw_curves = scaler.fit_transform(self.raw_curves)
            self.scaled_augmented_curves = scaler.transform(self.augmented_curves)

        elif normalize == 'std_scaler':
            scaler = StandardScaler()
            self.scaled_raw_curves = scaler.fit_transform(self.raw_curves)
            self.scaled_augmented_curves = scaler.transform(self.augmented_curves)

        elif normalize == 'None':
            self.scaled_raw_curves = self.raw_curves
            self.scaled_augmented_curves = self.augmented_curves

        # for i in range(20):
        #     figure = plt.figure()
        #     plt.plot(self.scaled_raw_curves[i], linewidth=2, color='blue')
        #     plt.plot(self.scaled_augmented_curves[i], linewidth=2, color='red')
        #     plt.grid(False)
        #     plt.show()

    def __len__(self):
        return len(self.raw_curves)

    def __getitem__(self, idx):
        return self.raw_curves[idx], self.augmented_curves[idx]
    
class class_dataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        # plt.plot(self.X[0])
        # plt.savefig('test.png')
        # breakpoint()
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.tensor(self.y[index]).to(torch.long)