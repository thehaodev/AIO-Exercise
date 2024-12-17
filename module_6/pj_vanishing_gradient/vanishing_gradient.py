import torch
import random
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from module_6 import util


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, hidden_dims)
        self.layer4 = nn.Linear(hidden_dims, hidden_dims)
        self.layer5 = nn.Linear(hidden_dims, hidden_dims)
        self.layer6 = nn.Linear(hidden_dims, hidden_dims)
        self.layer7 = nn.Linear(hidden_dims, output_dims)
        self.output = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Sigmoid()(x)
        x = self.layer3(x)
        x = nn.Sigmoid()(x)
        x = self.layer4(x)
        x = nn.Sigmoid()(x)
        x = self.layer5(x)
        x = nn.Sigmoid()(x)
        x = self.layer6(x)
        x = nn.Sigmoid()(x)
        x = self.layer7(x)
        x = nn.Sigmoid()(x)
        out = self.output(x)
        return out


def run():
    SEED = 42
    util.random_seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    batch_size = 512
    train_ratio = 0.9
    train_size = int(len(train_dataset) * train_ratio)
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_subset)}")
    print(f"Validation size: {len(val_subset)}")
    print(f"Test size: {len(test_dataset)}")

    input_dims = 784
    hidden_dims = 128
    output_dims = 10
    lr = 1e-2

    model = MLP(input_dims=input_dims,
                hidden_dims=hidden_dims,
                output_dims=output_dims).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    epochs = 100
    train_loss_lst = []
    train_acc_lst = []
    val_loss_lst = []
    val_acc_lst = []

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        count = 0

        model.train()
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (torch.argmax(outputs, 1) == y_train).sum().item()
            count += len(y_train)

        train_loss /= len(train_loader)
        train_loss_lst.append(train_loss)
        train_acc /= count
        train_acc_lst.append(train_acc)

        val_loss = 0.0
        val_acc = 0.0
        count = 0
        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                val_acc += (torch.argmax(outputs, 1) == y_val).sum().item()
                count += len(y_val)

        val_loss /= len(test_loader)
        val_loss_lst.append(val_loss)
        val_acc /= count
        val_acc_lst.append(val_acc)

        print(
            f"EPOCH {epoch + 1}/{epochs}, Train_Loss: {train_loss:.4f}, "
            f"Train_Acc: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Val_Acc: {val_acc:.4f}")

        val_target = []
        val_predict = []

        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                outputs = model(X_val)

                val_predict.append(outputs.cpu())
                val_target.append(y_val.cpu())

            val_predict = torch.cat(val_predict)
            val_target = torch.cat(val_target)
            val_acc = (torch.argmax(val_predict, 1) == val_target).sum().item() / len(val_target)

            print('Evaluation on val set:')
            print(f'Accuracy: {val_acc}')
