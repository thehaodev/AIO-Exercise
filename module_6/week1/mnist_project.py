import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

import util


class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding='same')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = F.relu(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = F.relu(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)
        return outputs


def plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].plot(epochs, train_accs, label="Training")
    axs[0].plot(epochs, eval_accs, label="Evaluation")
    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Loss")
    plt.legend()


def run():
    util.random_seed(42)

    ROOT = './data'
    train_data = datasets.MNIST(
        root=ROOT,
        train=True,
        download=True
    )
    test_data = datasets.MNIST(
        root=ROOT,
        train=False,
        download=True
    )

    # Split training: validation = 0.9 : 0.1
    VALID_RATIO = 0.9
    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(
        train_data,
        [n_train_examples, n_valid_examples]
    )

    # Compute mean and std for normalization
    mean = train_data.dataset.data.float().mean() / 255
    std = train_data.dataset.data.float().std() / 255

    # Define transformations
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    # Apply transformations to datasets
    train_data.dataset.transform = train_transforms
    valid_data.dataset.transform = test_transforms

    # Create data loaders
    BATCH_SIZE = 256

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0

    )

    valid_dataloader = DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    num_classes = len(train_data.dataset.classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lenet_model = LeNetClassifier(num_classes)
    lenet_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet_model.parameters(), weight_decay=0, lr=0.001)

    util.training(lenet_model, optimizer, criterion, train_dataloader, device, valid_dataloader)

    test_data.transform = test_transforms
    test_dataloader = data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    test_acc, test_loss = util.evaluate(lenet_model, criterion, test_dataloader, device)
    print(test_acc, test_loss)


run()
