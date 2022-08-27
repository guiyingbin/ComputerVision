import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self, input_channel):
        super(SPP, self).__init__()
        self.max_pool1 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=9, padding=4, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=13, padding=6, stride=1)
        self.conv1 = nn.Conv2d(in_channels=input_channel*4, out_channels=input_channel,
                               kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x1 = self.max_pool1(x)
        x2 = self.max_pool2(x)
        x3 = self.max_pool3(x)
        x = torch.cat([x, x1, x2, x3], dim=1)
        output = self.conv1(x)
        return output