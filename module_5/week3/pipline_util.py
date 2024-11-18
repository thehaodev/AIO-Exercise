import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from module_5.util import data_processing_util


def set_random_value():
    random_state = 59
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return random_state, device


def split_data(random_state, x, y, dtype_x, dtype_y):
    val_size = 0.2
    test_size = 0.125
    x_train, y_train, x_test, y_test, x_val, y_val = data_processing_util.sklearn_split_data(val_size=val_size,
                                                                                             test_size=test_size,
                                                                                             random_state=random_state,
                                                                                             x_features=x, y_label=y)
    x_train, x_val, x_test = data_processing_util.sklearn_normalizer(x_train, x_val, x_test)

    X_TRAIN = torch.tensor(x_train, dtype=dtype_x)
    X_VAL = torch.tensor(x_val, dtype=dtype_x)
    X_TEST = torch.tensor(x_test, dtype=dtype_x)

    y_train = torch.tensor(y_train, dtype=dtype_y)
    y_val = torch.tensor(y_val, dtype=dtype_y)
    y_test = torch.tensor(y_test, dtype=dtype_y)

    return X_TRAIN, X_VAL, X_TEST, y_train, y_val, y_test


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
