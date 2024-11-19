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


def training(epochs, model, device, optimizer, criterion, train_loader, val_loader):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        train_loss = 0.0
        train_target = []
        train_predict = []
        model.train()
        for x_samples, y_samples in train_loader:
            x_samples = x_samples.to(device)
            y_samples = y_samples.to(device)
            optimizer.zero_grad()
            outputs = model(x_samples)
            loss = criterion(outputs, y_samples)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_predict.append(outputs.detach().cpu())
            train_target.append(y_samples.cpu())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        train_predict = torch.cat(train_predict)
        train_target = torch.cat(train_target)
        train_acc = compute_accuracy(train_predict, train_target)
        train_accs.append(train_acc)

        val_loss = 0.0
        val_target = []
        val_predict = []
        model.eval()
        with torch.no_grad():
            for x_samples, y_samples in val_loader:
                x_samples = x_samples.to(device)
                y_samples = y_samples.to(device)
                outputs = model(x_samples)
                val_loss += criterion(outputs, y_samples).item()

                val_predict.append(outputs.cpu())
                val_target.append(y_samples.cpu())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        val_predict = torch.cat(val_predict)
        val_target = torch.cat(val_target)
        val_acc = compute_accuracy(val_predict, val_target)
        val_accs.append(val_acc)

        print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}')

    return train_losses, val_losses, train_accs, val_accs


def compute_accuracy(y_hat, y_true):
    _, y_hat = torch.max(y_hat, dim=1)
    correct = (y_hat == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy


def compute_evaluate(model, loader, device):
    # Evaluation on the test set
    target = []
    predict = []
    model.eval()

    with torch.no_grad():
        for x_samples, y_samples in loader:
            x_samples = x_samples.to(device)
            y_samples = y_samples.to(device)
            outputs = model(x_samples)

            predict.append(outputs.cpu())
            target.append(y_samples.cpu())

            # Concatenating the test predictions and targets
    predict = torch.cat(predict)
    target = torch.cat(target)

    return predict, target



class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
