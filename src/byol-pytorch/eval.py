import torch
import pandas as pd 
import torch.nn as nn
from dataset import class_dataset
from basenetwork import Transformer1D
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = 'cuda'
torch.manual_seed(42)
len_seq = 60    
d_model = 512
class_num = 7

dir_data = '7_plex'
# class classifier_model(nn.Module):
    
#     def __init__(self, model):
#         super(classifier_model, self).__init__()
#         self.net = model
#         self.last_layer = nn.Linear(len_seq * d_model, 7)
    
#     def forward(self, x):
#         x, _ = self.net(x)  
#         x = x.view(x.size(0), -1)
#         x = self.last_layer(x)  
#         return x

class classifier_model(nn.Module):
    
    def __init__(self, model, len_seq, d_model, class_num, dropout):
        super(classifier_model, self).__init__()
        self.net = model
        self.len_seq = len_seq
        self._d_model = d_model
        self.class_num = class_num
        self.dropout = dropout
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features = self._d_model * self.len_seq, out_features = 128),
                nn.Dropout(),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(in_features = 128, out_features = 128)
            )
        self.fc = nn.Linear(128, class_num)

    def forward(self, x):
        x, _ = self.net(x)  
        x = self.classifier(x)  
        x = self.fc(x)
        return x

df = pd.read_csv(os.path.join(dir_data, 'df_dPCR_GB_2025.csv'))

# split train and test data by sampling data from df
# df_train = df.sample(frac=0.8, random_state=42)
# df_test = df.drop(df_train.index)
# print(df_train.shape, df_test.shape)

df_train = df[df.train_index == 'train']
df_test = df[df.train_index == 'test']

X_train = df_train.filter(regex = r'\d+\.?\d*').to_numpy()
y_train = np.asarray(df_train.Target_cat)

X_test = df_test.filter(regex = r'\d+\.?\d*').to_numpy()
y_test = np.asarray(df_test.Target_cat)
print("number of classes in train:", len(np.unique(y_train))) 
print("number of classes in test:", len(np.unique(y_test)))
# Initialize the StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Scale data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

data_train = class_dataset(X_train, y_train)
data_test = class_dataset(X_test, y_test)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=128, shuffle=False)

loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device)

num_epochs = 50
loss_train_hist = [0] * num_epochs
loss_test_hist = [0] * num_epochs
accuracy_train_hist = [0] * num_epochs
accuracy_test_hist = [0] * num_epochs

# model = Transformer1D(input_dim=1, d_model=512, nhead=4, num_layers=4, dropout=0.5, batch_first=True)

F_config = {
    'd_input': 1,
    'd_model': 512, # Lattent dim
    'q': 8, # Query size
    'v': 8,  # Value size
    'h': 4, # Number of self-attention heads
    'N': 4, # Nuember of encoder to stack
    'attention_size': None,  # Attention wsindow size
    'dropout': 0.5, # drop out rate
    'chunk_mode': None,
    'pe': "regular",  # Positional encoding metric
    # 'batch_first': True,
    'pe_period': 20
}

transformer_model = Transformer_for_byol(**F_config)

## PART 1: NAIVE TRAINING
# final_model = classifier_model(transformer_model, len_seq, d_model, class_num, dropout=0.5)

## PART 1: WITH CL-PRETRAINING
# freeze the model except the last laeyer and attention layers
transformer_model.load_state_dict(torch.load('best_model_encode_LK.pth'), strict=False)
final_model = classifier_model(transformer_model, len_seq, d_model, class_num, dropout=0.5)

for param in final_model.net.parameters():
    param.requires_grad = False
    
for name, param in final_model.named_parameters():
    # print(name)
    if 'linear' in name:
        param.requires_grad = True
        print(name)
    
final_model = final_model.to(device)
final_model.train()


lr = 3e-3

optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

best_test_accuracy = 0
for epoch in range(num_epochs):
    
    ## TRAINING
    y_train_pred = []
    y_train_true = []
    train_acc = 0
    for i, (x_train_batch, y_train_batch) in enumerate(train_dataloader):
        
        x_train_batch = x_train_batch.to(device)
        y_train = y_train_batch.to(device)
        if x_train_batch.shape[0]>1:
            y_train_hat = final_model(x_train_batch)
            y_train_pred.append(torch.argmax(y_train_hat, dim=1))
            y_train_true.append(y_train)
            loss_train = loss_fn(y_train_hat, y_train)
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_train_hist[epoch] += loss_train.item()
            is_correct = (torch.argmax(y_train_hat, dim=1) == y_train).float()
            accuracy_train_hist[epoch] += is_correct.sum().item()
            
            train_acc += is_correct.sum().item()
            
            # if i % 10 == 0:
            #     print('Training')
            #     print('batch', i)
            #     print('Loss train =', loss_train)
            #     print('Accuracy train =', is_correct.mean())
    scheduler.step()
    # print('Epoch:', epoch, 'Train Accuracy:', train_acc/len(data_train))
    ## TESTING
    
    final_model.eval()
    y_test_pred = []
    y_test_true = []
    
    test_accuracy = 0
    test_loss = 0
    final_model.eval()
    with torch.no_grad():
        for j, (x_test_batch, y_test_batch) in enumerate(test_dataloader):
            if x_test_batch.shape[0]>1:
                y_test_hat = final_model(x_test_batch.to(device)).to(device)
                y_test = y_test_batch.type(torch.LongTensor).to(device)
                y_test_pred.append(torch.argmax(y_test_hat, dim=1))
                y_test_true.append(y_test)
                loss_test = loss_fn(y_test_hat, y_test)
                loss_test_hist[epoch] += loss_test.item()
                is_correct = (torch.argmax(y_test_hat, dim=1) == y_test).float()
                accuracy_test_hist[epoch] += is_correct.sum().item()

                test_accuracy += is_correct.sum().item()
                test_loss += loss_test.item()
    
    if best_test_accuracy < test_accuracy/len(data_test):
        best_test_accuracy = test_accuracy/len(data_test)
        torch.save(final_model.state_dict(), 'best_classifier1.pth')
        
    print('Epoch:', epoch, 'Test Accuracy:', test_accuracy/len(data_test), 'Test Loss:', test_loss/len(data_test))
    
    # loss_train_hist[epoch] /= len(train_dataloader.dataset)
    # loss_test_hist[epoch] /= len(test_dataloader.dataset)
    # accuracy_train_hist[epoch] /= len(train_dataloader.dataset)
    # accuracy_test_hist[epoch] /= len(test_dataloader.dataset)
