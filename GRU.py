import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

df = pd.read_csv('./data/Wimbledon_washed_features.csv')

# 创建数据序列
def create_sequences(df, n_steps):
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df.iloc[i:(i + n_steps), :-1].values)
        y.append(df.iloc[i + n_steps, -1])
    return np.array(X), np.array(y)

n_steps = 5

features = df.drop(['nextpoint_victor'], axis=1)
labels = df['nextpoint_victor']

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 将DataFrame转换回来
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled['nextpoint_victor'] = labels.reset_index(drop=True)

X, y = create_sequences(df_scaled, n_steps)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

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
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_dim = X.shape[2]  # 特征数量
hidden_dim = 64  # 隐藏层维度
output_dim = 2  # 输出维度，假设是2分类问题
n_layers = 2  # GRU层的数量

model = GRUModel(input_dim, hidden_dim, output_dim, n_layers)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for sequences, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

train_model(model, dataloader, criterion, optimizer, epochs=10)
