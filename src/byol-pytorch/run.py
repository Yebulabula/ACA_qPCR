
import torch
from byol_pytorch import BYOL
from torchvision import models
from basenetwork import Transformer_for_byol,Transformer1D
import numpy as np
import pandas as pd
import torch
from dataset import PCRDataset
from torch.utils.data import random_split,DataLoader
import matplotlib.pyplot as plt
import tqdm
from sklearn.preprocessing import StandardScaler
import os

dir_data = '7_plex'

if __name__ == '__main__':

    df  = pd.read_csv(os.path.join(dir_data, 'df_dPCR_SP_2025.csv'))
    df = df.filter(regex = r'\d+\.?\d*')
    n_pad_before=20
    n_pad_after=20
    curves = np.array([df.iloc[i,:].to_numpy() for i in range(df.shape[0])])
    
    # scaler = StandardScaler()
    # scaler.fit(curves)
    # curves = scaler.fit_transform(curves)
    
    dataset = PCRDataset(curves, n_pad_before, n_pad_after, normalize = 'None')
    
    # Splitting dataset into training and testing
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_sizeTransformer1D
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # resnet = models.resnet50(pretrained=True)
    
    F_config = {
        'd_input': 1,
        'd_model': 512, # Lattent dim
        'q': 8, # Query size
        'v': 8,  # Value size
        'h': 4, # Number of self-attention heads
        'N': 4, # Nuember of encoder to stack
        'attention_size': None,  # Attention wsindow size
        'dropout': 0.3, # drop out rate
        'chunk_mode': None,
        'pe': "regular",  # Positional encoding metric
        # 'batch_first': True,
        'pe_period': 20
    }

    transformer_model = Transformer_for_byol(**F_config)
    # print(transformer_model)
    # transformer_model = Transformer1D(input_dim=1, d_model=16, nhead=4, num_layers=2)
    transformer_model = transformer_model.to('cuda')

    # x = next(iter(train_loader))[0]
    # print(x.shape)
    # embedding, output = transformer_model(x.to('cuda')) 
    # print(embedding.shape, output.shape)

    learner = BYOL(
        transformer_model,
        image_size = 60,
        hidden_layer = 'avgpool'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-5, weight_decay=0.0001)

    log = {'train_loss': []}
    
    best_loss = np.inf
    
    # use tqdm
    for epoch in range(50):
        avg_loss = 0
        for i, (x, y) in enumerate(tqdm.tqdm(train_loader)):
            x, y = x.to('cuda'), y.to('cuda')
            loss = learner(x, y)
            log['train_loss'].append(loss.item())
            # print(f'Iter {i} Loss: {loss.item()}')
            avg_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
        
        avg_loss /= i
        print('Avg Loss:', avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(learner.online_encoder.net.state_dict(), 'best_model_encode_LK.pth')
            print(f'Epoch {epoch}: Model saved!')
    
    # save the learning curve.
    plt.plot(log['train_loss'])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
