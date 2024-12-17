import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from module_6 import util
import file_util


class WeatherDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.transform = transform
        self.img_paths = X
        self.labels = y

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Sequential()

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x += self.downsample(shortcut)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, residual_block, n_blocks_lst, n_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self.create_layer(residual_block, 64, 64, n_blocks_lst[0], stride=1)
        self.conv3 = self.create_layer(residual_block, 64, 128, n_blocks_lst[1], stride=2)
        self.conv4 = self.create_layer(residual_block, 128, 256, n_blocks_lst[2], stride=2)
        self.conv5 = self.create_layer(residual_block, 256, 512, n_blocks_lst[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, n_classes)

    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        blocks = []
        first_block = residual_block(in_channels, out_channels, stride)
        blocks.append(first_block)

        for _ in range(1, n_blocks):
            block = residual_block(out_channels, out_channels)
            blocks.append(block)

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


def run():
    seed = 59
    util.random_seed(seed)
    img_paths, labels, classes = file_util.read_file("../week2/img_cls_weather_dataset/weather-dataset/dataset")

    val_size = 0.2
    test_size = 0.125
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(
        img_paths, labels,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    train_dataset = WeatherDataset(
        X_train, y_train,
        transform=file_util.transform
    )
    val_dataset = WeatherDataset(
        X_val, y_val,
        transform=file_util.transform
    )
    test_dataset = WeatherDataset(
        X_test, y_test,
        transform=file_util.transform
    )

    train_batch_size = 512
    test_batch_size = 8
    train_loader, val_loader, test_loader = file_util.data_loader(train_dataset, val_dataset, test_dataset,
                                                                  train_batch_size, test_batch_size)

    n_classes = len(list(classes.keys()))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet(
        residual_block=ResidualBlock,
        n_blocks_lst=[2, 2, 2, 2],
        n_classes=n_classes
    ).to(device)

    lr = 1e-2
    epochs = 25

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    train_losses, val_losses = util.fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(train_losses)
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].plot(val_losses, color='orange')
    ax[1].set_title('Val Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    plt.show()

    val_loss, val_acc = util.evaluate_cnn(
        model,
        val_loader,
        criterion,
        device
    )
    test_loss, test_acc = util.evaluate_cnn(
        model,
        test_loader,
        criterion,
        device
    )

    print('Evaluation on val/test dataset')
    print('Val accuracy: ', val_acc)
    print('Test accuracy: ', test_acc)


run()
