import torch
import torch.nn as nn
import math


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        try:
            return math.exp(x) / self.__sum_exp
        except OverflowError:
            return float("nan")

    def __call__(self, tensor: torch.Tensor):
        self.__tensor = tensor
        self.__sum_exp = 0
        for data in self.__tensor.data:
            try:
                self.__sum_exp += math.exp(data)
            except OverflowError:
                self.__sum_exp += math.inf

        for i in range(torch.numel(self.__tensor)):
            self.__tensor.data[i] = self.forward(tensor.data[i])

        return print(self.__tensor)


class SoftmaxStable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, max_c):
        return math.exp(x - max_c) / self.__sum_exp

    def __call__(self, tensor: torch.Tensor):
        self.__tensor = tensor
        self.__sum_exp = 0
        max_c = max(self.__tensor.data)
        for data in self.__tensor.data:
            try:
                self.__sum_exp += math.exp(data - max_c)
            except OverflowError:
                self.__sum_exp += math.inf

        for i in range(torch.numel(self.__tensor)):
            self.__tensor.data[i] = self.forward(tensor.data[i], max_c)

        return print(self.__tensor)
