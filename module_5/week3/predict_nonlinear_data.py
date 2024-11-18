import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pipline_util
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.output(x)
        return out.squeeze(1)


def compute_accuracy(y_hat, y_true):
    _, y_hat = torch.max(y_hat, dim=1)
    correct = (y_hat == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy


def run():
    random_state, device = pipline_util.set_random_value()

    # Read data
    dataset_path = '../data/NonLinear_data.npy'
    data = np.load(dataset_path, allow_pickle=True).item()

    # Preprocessing
    X = data['X']
    y = data['labels']

    X_TRAIN, X_VAL, X_TEST, y_train, y_val, y_test = pipline_util.split_data(random_state, X, y,
                                                                             dtype_x=torch.float32,
                                                                             dtype_y=torch.long)

    batch_size = 32
    # Dataset
    train_dataset = pipline_util.CustomDataset(X_TRAIN, y_train)
    val_dataset = pipline_util.CustomDataset(X_VAL, y_val)
    test_dataset = pipline_util.CustomDataset(X_TEST, y_test)

    # Loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dims = X_TRAIN.shape[1]
    output_dims = torch.unique(y_train, dim=0).shape[0]
    hidden_dims = 128
    model = MLP(input_dims=input_dims,
                hidden_dims=hidden_dims,
                output_dims=output_dims).to(device)

    lr = 1e-1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0, lr=lr, momentum=0)

    epochs = 100
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
    _, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Plotting training and validation loss
    ax[0, 0].plot(train_losses, color='green')
    ax[0, 0].set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
    ax[0, 1].plot(val_losses, color='orange')
    ax[0, 1].set(xlabel='Epoch', ylabel='Loss', title='Validation Loss')

    # Plotting training and validation accuracy
    ax[1, 0].plot(train_accs, color='green')
    ax[1, 0].set(xlabel='Epoch', ylabel='Accuracy', title='Training Accuracy')
    ax[1, 1].plot(val_accs, color='orange')
    ax[1, 1].set(xlabel='Epoch', ylabel='Accuracy', title='Validation Accuracy')

    plt.tight_layout()
    plt.show()

    # Evaluation on the test set
    test_target = []
    test_predict = []
    model.eval()

    with torch.no_grad():
        for x_samples, y_samples in test_loader:
            x_samples = x_samples.to(device)
            y_samples = y_samples.to(device)
            outputs = model(x_samples)

            test_predict.append(outputs.cpu())
            test_target.append(y_samples.cpu())

            # Concatenating the test predictions and targets
    test_predict = torch.cat(test_predict)
    test_target = torch.cat(test_target)

    # Compute test accuracy
    test_acc = compute_accuracy(test_predict, test_target)
    print('Evaluation on test set:')
    print(f'Accuracy: {test_acc}')


run()
