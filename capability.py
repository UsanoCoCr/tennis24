import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# load matches 2015-2023
for i in range(2015, 2024):
    df = pd.read_csv(f'./data/player_history/atp_matches_{i}.csv')
    if i == 2015:
        matches = df
    else:
        matches = pd.concat([matches, df])

X

class LinearModel(nn.modules):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(LinearModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
    

