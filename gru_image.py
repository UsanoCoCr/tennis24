""" import matplotlib.pyplot as plt
accuracies = [0.7657, 0.7497, 0.7616, 0.7340, 0.7431, 0.7363, 0.7276, 0.7554, 0.7389, 0.7444]
plt.plot(range(1, 11), accuracies)
plt.xlabel('n_steps')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('./image/gru_accuracy_nsteps.png')
plt.show()  """

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

df = pd.read_csv('./data/one_match.csv', error_bad_lines=False)

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

def calculate_discounted_performance(performance_list, decay_rate):
    discounted_performance_vector = []
    n = len(performance_list)
    for i, performance in enumerate(performance_list):
        discount_factor = decay_rate * discounted_performance_vector[-1] if i > 0 else 1
        discounted_performance = discount_factor + performance
        discounted_performance_vector.append(discounted_performance)
    return discounted_performance_vector

def normalize_vectors(vector1, vector2):
    assert len(vector1) == len(vector2), "Vectors must have the same length"
    
    normalized_vector1 = []
    normalized_vector2 = []
    
    for v1, v2 in zip(vector1, vector2):
        total = v1 + v2
        if total > 0:
            normalized_vector1.append(v1 / total)
            normalized_vector2.append(v2 / total)
        else:
            normalized_vector1.append(0.5)
            normalized_vector2.append(0.5)
    
    return normalized_vector1, normalized_vector2

decay_rate = 0.9

player1_discounted_vector = calculate_discounted_performance(player1_performance, decay_rate)
player2_discounted_vector = calculate_discounted_performance(player2_performance, decay_rate)
player1_weighted, player2_weighted = normalize_vectors(player1_discounted_vector, player2_discounted_vector)

x_new = np.linspace(1, len(player1_weighted), 2000)
spl1 = make_interp_spline(range(1, len(player1_weighted) + 1), player1_weighted, k=3) 
spl2 = make_interp_spline(range(1, len(player2_weighted) + 1), player2_weighted, k=3)
player1_smooth = spl1(x_new)
player2_smooth = spl2(x_new)
player1_smooth_clipped = np.clip(player1_smooth, 0, 1)
player2_smooth_clipped = np.clip(player2_smooth, 0, 1)

""" player1_smooth_clipped = player1_weighted
player2_smooth_clipped = player2_weighted """

# save the performance vector in csv file
""" df = pd.DataFrame({'player1': player1_smooth_clipped, 'player2': player2_smooth_clipped})
df.to_csv('./Alcalaz/performance.csv', index=False) """


plt.figure(figsize=(12, 4))
plt.plot(x_new, player1_smooth_clipped, label='Daniil Medvedev')
plt.plot(x_new, player2_smooth_clipped, label='Carlos Alcaraz')
""" plt.plot(range(1, len(player1_smooth_clipped) + 1), player1_smooth_clipped, label='Matteo Berrettini')
plt.plot(range(1, len(player2_smooth_clipped) + 1), player2_smooth_clipped, label='Carlos Alcaraz') """
plt.xlabel('Points Played')
plt.ylabel('Performance')
plt.legend()
plt.tight_layout()
plt.savefig('./Alcalaz/vsDaniilMedvedev.png')
plt.show()