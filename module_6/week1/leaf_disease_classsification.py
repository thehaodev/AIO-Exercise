import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from module_6 import util


def loader(path):
    return Image.open(path)


class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding='same')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjusted dimensions after conv and pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = F.relu(outputs)
        outputs = self.avgpool1(outputs)
        outputs = self.conv2(outputs)
        outputs = F.relu(outputs)
        outputs = self.avgpool2(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc1(outputs)
        outputs = F.relu(outputs)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)
        return outputs


def run():
    util.random_seed(42)

    data_paths = {
        'train': './train',
        'valid': './validation',
        'test': './test'
    }

    img_size = 150
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(
        root=data_paths['train'],
        loader=loader,
        transform=train_transforms
    )
    valid_data = datasets.ImageFolder(
        root=data_paths['valid'],
        transform=train_transforms
    )
    test_data = datasets.ImageFolder(
        root=data_paths['test'],
        transform=train_transforms
    )

    BATCH_SIZE = 512

    train_dataloader = data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    valid_dataloader = data.DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    test_dataloader = data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    num_classes = len(train_data.classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lenet_model = LeNetClassifier(num_classes)
    lenet_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 2e-4
    optimizer = optim.Adam(lenet_model.parameters(), learning_rate, weight_decay=0)

    util.training(lenet_model, optimizer, criterion, train_dataloader, device, valid_dataloader)
    test_acc, test_loss = util.evaluate(lenet_model, criterion, test_dataloader, device)
    print(test_acc, test_loss)
