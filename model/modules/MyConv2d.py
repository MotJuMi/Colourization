import torch.nn as nn

class MyConv2d(nn.Module):
    """
    My simplified implemented of nn.Conv2d module for 2D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding
        self.weight = nn.parameter.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.parameter.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding)

class MyDilatedConv2d(MyConv2d):
    """
    Dilated Convolution 2D
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(MyDilatedConv2d, self).__init__(in_channels,
                                              out_channels,
                                              kernel_size)
        self.dilation = dilation

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding, dilation=self.dilation)