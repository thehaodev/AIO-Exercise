import torch
import torch.nn as nn

torch.manual_seed(42)

# Simulate one image with size 6 x 6 pixel
input_image = torch.randint(5, (1, 6, 6), dtype=torch.float32)
print(input_image)

# Define convolutional 2D with kernel size = 3x3
conv_layer = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,  # create a kernel: 3 x 3
    bias=False
)

# Define custom weight
init_kernel_weight = torch.randint(
    high=2,
    size=conv_layer.weight.data.shape,
    dtype=torch.float32
)
# Custom kernel
conv_layer.weight.data = init_kernel_weight
print(conv_layer.weight)

# Simple output
output = conv_layer(input_image)
print(output)

# Bias
print(conv_layer.bias)
conv_layer.bias = nn.Parameter(torch.tensor([1],dtype=torch.float32))
print(conv_layer.bias)

# custom padding=(2,1)
conv_layer = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    padding='same'
)

# Stride custom stride=(2,1)
conv_layer_stride = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=2
)

# Max pooling
max_pool_layer = nn.MaxPool2d(
    kernel_size=2,
    stride=(1,2)
)

# Flatten_layer
flatten_layer = nn.Flatten

