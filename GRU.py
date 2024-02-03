import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using {device} device")

df = pd.read_csv('./data/gru_input.csv')

# 创建数据序列
def create_sequences(df, n_steps):
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df.iloc[i:(i + n_steps), :-1].values)
        y.append(df.iloc[i + n_steps, -1])
    return np.array(X), np.array(y)

n_steps = 1

features = df.drop(['game_victor'], axis=1)
labels = df['game_victor']

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 将DataFrame转换回来
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled['game_victor'] = labels.reset_index(drop=True)

X, y = create_sequences(df_scaled, n_steps)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

X_train = X_tensor[:int(0.7*len(X_tensor))]
y_train = y_tensor[:int(0.7*len(y_tensor))]
X_test = X_tensor[int(0.7*len(X_tensor)):]
y_test = y_tensor[int(0.7*len(y_tensor)):]

# 创建数据集和数据加载器
class TennisDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = sequences
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = TennisDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
""" dataset = TennisDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataset = TennisDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True) """

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_dim = X.shape[2] 
hidden_dim = 128
output_dim = 2
n_layers = 5

model = GRUModel(input_dim, hidden_dim, output_dim, n_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

losses = []

def train_model(model, dataloader, criterion, optimizer, epochs):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    for epoch in range(epochs):
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels.long())
            
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig('./image/gru_loss.png')

def test_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            print(outputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {correct / total}')

train_model(model, dataloader, criterion, optimizer, epochs=20)
# test_model(model, test_dataloader)
test_model(model, dataloader)

# 保存模型
torch.save(model.state_dict(), './models/gru_model.pth')

""" # 画出准确率随着n_steps的变化
accuracies = []
for n_steps in range(1, 11):
    X, y = create_sequences(df_scaled, n_steps)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    X_train = X_tensor[:int(0.7*len(X_tensor))]
    y_train = y_tensor[:int(0.7*len(y_tensor))]
    X_test = X_tensor[int(0.7*len(X_tensor)):]
    y_test = y_tensor[int(0.7*len(y_tensor)):]
    dataset = TennisDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_dataset = TennisDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = GRUModel(X.shape[2], hidden_dim, output_dim, n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_model(model, dataloader, criterion, optimizer, epochs=20)
    accuracies.append(test_model(model, test_dataloader))

plt.plot(range(1, 11), accuracies)
plt.xlabel('n_steps')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('./image/gru_accuracy_nsteps.png')
plt.show() """