import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from torchvision.io import read_image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pipline_util


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims * 4)
        self.linear2 = nn.Linear(hidden_dims * 4, hidden_dims * 2)
        self.linear3 = nn.Linear(hidden_dims * 2, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        out = self.output(x)
        return out.squeeze(1)


def visualize_image(image_batch, label_batch, idx2label):
    # Create a figure for plotting
    plt.figure(figsize=(10, 10))

    # Plotting the images
    for i in range(9):
        _ = plt.subplot(3, 3, i + 1)

        # Get the minimum and maximum values for normalization
        minv = image_batch[i].numpy().min()
        maxv = image_batch[i].numpy().max()

        # Display the image with proper normalization
        plt.imshow(np.squeeze(image_batch[i].numpy()), vmin=minv, vmax=maxv, cmap="gray")

        # Display the corresponding label
        label = label_batch[i]
        plt.title(idx2label[label.item()])
        plt.axis("off")  # Turn off axis

    plt.show()  # Show the figure


class ImageDataset(Dataset):
    def __init__(self, random_state, img_dir, norm, img_height, img_width, label2idx, split='train', train_ratio=0.8):
        self.resize = Resize((img_height, img_width))
        self.norm = norm
        self.split = split
        self.train_ratio = train_ratio
        self.img_dir = img_dir
        self.label2idx = label2idx
        self.img_paths, self.img_labels = self.read_img_files()

        if split in ['train', 'val'] and 'train' in img_dir.lower():
            train_data, val_data = train_test_split(
                list(zip(self.img_paths, self.img_labels)),
                train_size=train_ratio,
                random_state=random_state,
                stratify=self.img_labels
            )
            if split == 'train':
                self.img_paths, self.img_labels = zip(*train_data)
            elif split == 'val':
                self.img_paths, self.img_labels = zip(*val_data)

    def read_img_files(self):
        img_paths = []
        img_labels = []
        for cls in self.label2idx.keys():
            for img in os.listdir(os.path.join(self.img_dir, cls)):
                img_paths.append(os.path.join(self.img_dir, cls, img))
                img_labels.append(cls)
        return img_paths, img_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        cls = self.img_labels[idx]
        img = self.resize(read_image(img_path))
        img = img.type(torch.float32)
        label = self.label2idx[cls]
        if self.norm:
            img = (img / 127.5) - 1
        return img, label


def run():
    random_state, device = pipline_util.set_random_value()

    train_dir = r'D:\AI_VIETNAM\CODE_EXERCISE\AIO-Exercise\module_5\week3\train'
    test_dir = r'D:\AI_VIETNAM\CODE_EXERCISE\AIO-Exercise\module_5\week3\test'

    classes = os.listdir(train_dir)

    label2idx = {cls: idx for idx, cls in enumerate(classes)}
    idx2label = {idx: cls for cls, idx in label2idx.items()}

    test_img_path = r"D:\AI_VIETNAM\CODE_EXERCISE\AIO-Exercise\module_5\week3\test\angry\PrivateTest_88305.jpg"
    img = plt.imread(test_img_path)
    img_height, img_width = img.shape
    print(f'Image height: {img_height}')
    print(f'Image width: {img_width}')

    batch_size = 256

    # Create training dataset and loader
    train_dataset = ImageDataset(random_state, train_dir, True, img_height, img_width, label2idx, split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create validation dataset and loader
    val_dataset = ImageDataset(random_state, train_dir, True, img_height, img_width, label2idx, split='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create test dataset and loader
    test_dataset = ImageDataset(random_state, test_dir, True, img_height, img_width, label2idx, split='test')
    _ = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Set input dimensions, output dimensions, and hidden dimensions
    input_dims = img_height * img_width
    output_dims = len(classes)
    hidden_dims = 64
    lr = 1e-2

    image_batch, label_batch = next(iter(train_loader))
    visualize_image(image_batch, label_batch, idx2label)
    # Create the model
    model = MLP(input_dims=input_dims,
                hidden_dims=hidden_dims,
                output_dims=output_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0, lr=lr, momentum=0)

    epochs = 100
    train_losses, val_losses, train_accs, val_accs = pipline_util.training(epochs, model, device, optimizer,

                                                                           criterion, train_loader, val_loader)
    _, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].plot(train_losses, color='green')
    ax[0, 0].set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
    ax[0, 1].plot(val_losses, color='orange')
    ax[0, 1].set(xlabel='Epoch', ylabel='Loss', title='Validation Loss')

    # Plotting training and validation accuracy
    ax[1, 0].plot(train_accs, color='green')
    ax[1, 0].set(xlabel='Epoch', ylabel='Accuracy', title='Training Accuracy')
    ax[1, 1].plot(val_accs, color='orange')
    ax[1, 1].set(xlabel='Epoch', ylabel='Accuracy', title='Validation Accuracy')

    # Display the plots
    plt.tight_layout()  # Adjusts subplot params for better layout
    plt.show()

    val_predict, val_target = pipline_util.compute_evaluate(model, val_loader, device)

    # Compute test accuracy
    val_acc = pipline_util.compute_accuracy(val_predict, val_target)
    print('Evaluation on val set:')
    print(f'Accuracy: {val_acc}')


run()
