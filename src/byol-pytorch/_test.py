import torch
import pandas as pd 
import torch.nn as nn
from dataset import class_dataset
from basenetwork import Transformer1D
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = 'cuda'
torch.manual_seed(42)

class classifier_model(nn.Module):
    def __init__(self, model, d_model):
        super(classifier_model, self).__init__()
        self.net = model
        self.last_layer = nn.Linear(60*d_model,7)
    
    def forward(self, x):
        x, _ = self.net(x)  
        x = x.view(x.size(0), -1)
        x = self.last_layer(x)  
        return x

config = {
        # "method": 'CDAN+E',
        "num_epochs": 4000,
        # "test_interval": 20,
        "output_path": "CDAN_ACA/" + 'log',
        # "loss": {"trade_off": 1.0, "random": False, "random_dim": 512},
        "optimizer": {"type":optim.Adam, "optim_params":{'lr':1e-3, \
                            "weight_decay":0.0001}, "lr_type":"inv", \
                            "lr_param":{"lr":1e-3, "gamma":0.001, "power":0.75}},
        # "data": {"dir_data": dir_data, 
        #          "source":{"name": "df_dPCR_GB_bs_man_norm.csv", "batch_size":128}, 
        #          "target":{"name": "df_dPCR_SP_bs_man_norm.csv", "batch_size":128},
        #          "test":{"name": "df_qPCR_SP_bs_man_norm.csv", "batch_size":32}},
        "class_num": 7
    }

df = pd.read_csv(os.path.join(dir_data, 'df_dPCR_GB_2025.csv'))

df_train = df[df.train_index == 'train']
df_test = df[df.train_index == 'test']

X_train = df_train.filter(regex = r'\d+\.?\d*').to_numpy()
y_train = np.asarray(df_train.Target_cat)

X_test = df_test.filter(regex = r'\d+\.?\d*').to_numpy()
y_test = np.asarray(df_test.Target_cat)
print("number of classes in test:", len(np.unique(y_test))) 

# Initialize the StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Scale data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
data_train = class_dataset(X_train, y_train)
data_test = class_dataset(X_test, y_test)
print('training data')
print(X_train.shape, y_train.shape)
print('testing data')
print(X_test.shape, y_test.shape)

# Create dataloaders
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=128, shuffle=False)

# Initialize the model
model = Transformer1D(input_dim=1, d_model=512, nhead=4, num_layers=4, dropout=0.5, batch_first=True)
class_model = classifier_model(model, d_model=512)

# Train the model
class_model = class_model.to(device)
class_model.train()
optimizer = optim.Adam(class_model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
best_acc = 0.0
best_model = None

for i in range(config["num_epochs"]):
    optimizer.zero_grad()
    
    # Get a batch from the dataloader
    x_train_batch, y_train_batch = next(iter(train_dataloader))
    x_train_batch = x_train_batch.to(device)
    y_train_batch = y_train_batch.to(device)

    # Forward pass
    outputs = class_model(x_train_batch)  # Model predictions (logits)
    labels = y_train_batch

    # Compute loss
    classifier_loss = nn.CrossEntropyLoss()(outputs, labels.long())

    # Backward pass
    classifier_loss.backward()
    optimizer.step()

    # Compute training accuracy
    with torch.no_grad():  # No need to track gradients for accuracy calculation
        preds = torch.argmax(outputs, dim=1)  # Get class predictions
        correct = (preds == labels).sum().item()  # Count correct predictions
        total = labels.size(0)  # Number of samples in the batch
        train_acc = correct / total  # Compute accuracy

    # Print progress every 10 iterations
    if i % 10 == 0:
        print(f"Iteration: {i}, Loss: {classifier_loss.item():.4f}, Training Accuracy: {train_acc * 100:.2f}%")

    # Save the best model based on training accuracy
    if train_acc > best_acc:
        best_acc = train_acc
        best_model = class_model.state_dict()  # Save model parameters

print("Best Training Accuracy: {:.2f}%".format(best_acc * 100))

## TEST MODEL

final_model = final_model.to(device)
final_model.eval()
y_test_pred = []
y_test_true = []

test_accuracy = 0
test_loss = 0
final_model.eval()

# plot confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_test_pred = []
y_test_true = []

def plot_confusion_matrix(y_true, y_pred, class_names, fontsize=15):
    """
    Plots the confusion matrix.
    
    Parameters:
    - y_true: List or array of true labels
    - y_pred: List or array of predicted labels
    - class_names: List of class names
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": fontsize})
    
    plt.xlabel('Predicted', fontsize=fontsize)
    plt.ylabel('Actual', fontsize=fontsize)
    plt.title('without Contrastive Learning', fontsize=fontsize)
    
    # Set tick labels font size
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Save the figure
    plt.savefig('without Contrastive Learning.png')
    
with torch.no_grad():
    for j, (x_test_batch, y_test_batch) in enumerate(test_dataloader):
        if x_test_batch.shape[0]>1:
            y_test_hat = final_model(x_test_batch.to(device)).to(device)
            y_test = y_test_batch.type(torch.LongTensor).to(device)
            
            # add to confusion matrix
            
            y_test_pred.append(torch.argmax(y_test_hat, dim=1))
            y_test_true.append(y_test)
            is_correct = (torch.argmax(y_test_hat, dim=1) == y_test).float()
            test_accuracy += is_correct.sum().item()
    
    y_test_pred = torch.cat(y_test_pred)
    y_test_true = torch.cat(y_test_true)
    
    plot_confusion_matrix(y_test_true.cpu().numpy(), y_test_pred.cpu().numpy(), ['Adeno', 'Cov_229E', 'Cov_HKU1', 'Cov_NL63', 'Cov_OC43', 'MERS',
       'COVID'])

print("Test Accuracy: ", test_accuracy/len(data_test))