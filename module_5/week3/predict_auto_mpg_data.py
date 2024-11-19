import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pipline_util


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        out = self.output(x)
        return out.squeeze(1)


def r_squared(y_true, y_pred, device):
    # Convert inputs to tensors and move to the appropriate device
    y_true = torch.Tensor(y_true).to(device)
    y_pred = torch.Tensor(y_pred).to(device)

    # Calculate mean of true values
    mean_true = torch.mean(y_true)

    # Calculate total sum of squares
    ss_tot = torch.sum((y_true - mean_true) ** 2)

    # Calculate residual sum of squares
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Calculate R²
    r2 = 1 - (ss_res / ss_tot)
    return r2


def run():
    random_state, device = pipline_util.set_random_value()

    # Read data
    dataset_path = '../data/Auto_MPG_data.csv'
    dataset = pd.read_csv(dataset_path)

    # Preprocessing
    X = dataset.drop(columns='MPG').to_numpy()
    y = dataset['MPG'].values

    X_TRAIN, X_VAL, X_TEST, y_train, y_val, y_test = pipline_util.split_data(random_state, X, y,
                                                                             dtype_x=torch.float32,
                                                                             dtype_y=torch.float32)

    batch_size = 32
    train_dataset = pipline_util.CustomDataset(X_TRAIN, y_train)
    val_dataset = pipline_util.CustomDataset(X_VAL, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dims = X_TRAIN.shape[1]
    output_dims = 1
    hidden_dims = 64
    model = MLP(input_dims=input_dims, hidden_dims=hidden_dims, output_dims=output_dims).to(device)

    lr = 1e-2
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0, lr=lr, momentum=0)

    epochs = 100
    train_losses = []
    val_losses = []
    train_r2 = []
    val_r2 = []

    for epoch in range(epochs):
        train_loss = 0.0
        train_target = []
        val_target = []
        train_predict = []
        val_predict = []

        model.train()
        for x_samples, y_samples in train_loader:
            x_samples = x_samples.to(device)
            y_samples = y_samples.to(device)

            optimizer.zero_grad()
            outputs = model(x_samples)
            train_predict += outputs.tolist()
            train_target += y_samples.tolist()

            loss = criterion(outputs, y_samples)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_r2.append(r_squared(train_target, train_predict, device))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_samples, y_samples in val_loader:
                x_samples = x_samples.to(device)
                y_samples = y_samples.to(device)

                outputs = model(x_samples)
                val_predict += outputs.tolist()
                val_target += y_samples.tolist()

                val_loss += criterion(outputs, y_samples).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_r2.append(r_squared(val_target, val_predict, device))

        print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}')

    model.eval()
    with torch.no_grad():
        y_hat = model(X_TEST)
        test_set_r2 = r_squared(y_hat, y_test, device)
        print('Evaluation on test set:')
        print(f'R2: {test_set_r2}')


run()
