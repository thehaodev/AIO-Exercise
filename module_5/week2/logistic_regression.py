import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def run():
    # Data preparation
    data = np.genfromtxt('../data/Iris.csv', delimiter=',', skip_header=1, dtype=str)

    # Separating features (X) and target (y)
    X = data[:, 1:-1].astype(float)   # All columns except the last
    y = data[:, -1]  # Last column

    # Encoding the target labels manually
    unique_classes = np.unique(y)
    label_encoding = {label: index for index, label in enumerate(unique_classes)}
    y_encoded = np.array([label_encoding[label] for label in y])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_encoded, dtype=torch.int64)

    # Create a linear layer
    input_dim = X.size(dim=1)
    output_dim = len(torch.unique(y, dim=0))
    linear = nn.Linear(input_dim, output_dim)

    # some params
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(linear.parameters(), weight_decay=0, lr=0.1, momentum=0)

    epochs = 2000

    # training
    losses = []
    for _ in range(epochs):
        outputs = linear(X)

        # Compute loss
        loss = criterion(outputs, y)
        losses.append(loss.item())
        
        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        # update
        optimizer.step()

    plt.plot(losses)
    plt.show()

    # Compute accuracy for data X
    with torch.no_grad():
        outputs = linear(X)
        predicted = torch.argmax(outputs, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)
        print(f'Accuracy: {accuracy * 100:.2f}%')


run()
