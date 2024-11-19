import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def plot_mnist_images():
    # Load the MNIST dataset - Transform PIL image to tensor format [0, 255] to [0,1]
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    mnist_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)

    # Create a data loader to iterate through the dataset
    data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=10, shuffle=True, num_workers=0)

    # Plot the first 10 images from the dataset
    for images, labels in data_loader:
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].squeeze().numpy(), cmap='gray')
            plt.title(f'Label: {labels[i]}')
            plt.axis('off')
        plt.show()
        break  # Only plot the first batch


def run():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True, num_workers=0)

    test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    # Input and output dimensions
    input_dim = 28 * 28  # MNIST images are 28x28
    output_dim = 10  # 10 classes for MNIST

    # Create a Sequential model
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(input_dim, output_dim))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), weight_decay=0, lr=0.1, momentum=0)

    # Training loop
    num_epochs = 5
    losses = []  # to store the average loss for each epoch

    for epoch in range(num_epochs):
        loss = 0
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the loss every epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Testing the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy on 10,000 test images: {100 * correct / total:.2f}%')

    # Plotting the loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()


run()
