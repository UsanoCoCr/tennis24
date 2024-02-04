import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using {device} device")

df = pd.read_csv('./data/gru_input.csv', error_bad_lines=False)

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

# 创建数据集和数据加载器
class TennisDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = sequences
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

player1_performance = []
player2_performance = []   
def test_model(model, dataloader):
    with torch.no_grad():
        correct = 0
        total = 0
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            player1_performance.append(softmax_outputs[0][0].item())
            player2_performance.append(softmax_outputs[0][1].item())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {correct / total}')

test_dataset = TennisDataset(X_tensor, y_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
state_dict = torch.load('./models/gru_model.pth')
model.load_state_dict(state_dict)

test_model(model, test_dataloader)

# 保存player1和player2的表现到csv
player1_performance_df = pd.DataFrame(player1_performance, columns=['player1_performance'])
player2_performance_df = pd.DataFrame(player2_performance, columns=['player2_performance'])
player_performance_df = pd.concat([player1_performance_df, player2_performance_df], axis=1)
player_performance_df.to_csv('./data/player_performance.csv', index=False)

""" weights = [0.1, 0.15, 0.2, 0.25, 0.3]
player1_weighted = []
player2_weighted = []
def calculate_weighted_performance(performance_list, weights):
    weighted_performance = []
    for i in range(len(performance_list)):
        if i < len(weights):
            weighted_performance.append(performance_list[i])
        else:
            weighted_performance.append(np.dot(performance_list[i-len(weights):i], weights))
    return weighted_performance

player1_weighted = calculate_weighted_performance(player1_performance, weights)
player2_weighted = calculate_weighted_performance(player2_performance, weights) """



""" x_new = np.linspace(1, len(player1_weighted), 1000)
spl1 = make_interp_spline(range(1, len(player1_weighted) + 1), player1_weighted, k=3) 
spl2 = make_interp_spline(range(1, len(player2_weighted) + 1), player2_weighted, k=3)
player1_smooth = spl1(x_new)
player2_smooth = spl2(x_new)
player1_smooth_clipped = np.clip(player1_smooth, 0, 1)
player2_smooth_clipped = np.clip(player2_smooth, 0, 1) """

""" plt.figure(figsize=(12, 4))
plt.plot(x_new, player1_smooth_clipped, label='Fan Zhendong')
plt.plot(x_new, player2_smooth_clipped, label='Ma Long')
plt.xlabel('Checkpoint')
plt.ylabel('Performance')
plt.legend()
plt.tight_layout()
plt.savefig('./image/pingpong_reset.pth.png')
plt.show() """