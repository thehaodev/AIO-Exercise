import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary


def run():
    # Data preparation
    data = np.genfromtxt('../data/data.csv', delimiter=',')
    x_data = torch.from_numpy(data[:, 0:1]).float()
    y_data = torch.from_numpy(data[:, 1:]).float()

    print(y_data)

    # Create a linear layer
    linear = nn.Linear(1, 1)
    summary(model=linear, input_data=(1,))

    # set value (for illustration)
    linear.weight.data = torch.Tensor([[-0.34]])
    linear.bias.data = torch.Tensor([0.04])

    # some params
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(linear.parameters(), weight_decay=0, lr=0.01, momentum=0)
    epochs = 1000

    # training
    losses = []
    for _ in range(epochs):
        # y_hat
        y_hat = linear(x_data)

        # loss
        loss = loss_fn(y_hat, y_data)
        losses.append(loss.item())

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        # update
        optimizer.step()

    plt.plot(losses)
    plt.show()


run()
