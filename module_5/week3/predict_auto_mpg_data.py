import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run():
    random_state = 59
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)

    _ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read data
    dataset_path = '../data/titanic_modified_dataset.csv'
    dataset = pd.read_csv(dataset_path, index_col='PassengerId')

    # Preprocessing
    _ = dataset.drop(columns='MPG').to_numpy()
    _ = dataset['MPG'].values


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
