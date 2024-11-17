import torch
import torch.nn as nn


class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        # set value (for illustration)
        self.linear.weight.data = torch.Tensor([[0.2], [-0.1]])
        self.linear.bias.data = torch.Tensor([0.1, 0.05])

    def forward(self, x):
        return self.linear(x)
