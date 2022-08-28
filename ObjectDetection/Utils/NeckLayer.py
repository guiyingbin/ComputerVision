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

# class FPN(nn.Module):
#     def __init__(self):
#
#     def forward(self):

class Modified_PAN(nn.Module):
    def __init__(self, channels=[128, 256, 1024]):
        super(Modified_PAN, self).__init__()
        self.down_sample = []
        self.conv_set = []
        for i in range(len(channels)-1):
            self.down_sample.append(nn.Conv2d(in_channels=channels[i],
                                              out_channels=channels[i+1],
                                              kernel_size=3,
                                              padding=1,
                                              stride=2))
            self.conv_set.append(nn.Conv2d(in_channels=channels[i+1]*2,
                                           out_channels=channels[i+1],
                                           kernel_size=1,
                                           padding=0,
                                           stride=1))
    def forward(self, x:list):
        """
        :param x: [C4, C5, C6 ..]
        :return: [C4, C5, C6 ..]
        """
        n = len(x)
        assert n == len(self.down_sample)+1
        output = [x[0]]
        for i in range(n-1):
            C_i = x[i]
            C_i_downsample = self.down_sample[i](C_i)
            temp = torch.cat([x[i+1], C_i_downsample], dim=1)
            output.append(self.conv_set[i](temp))
        return output

if __name__ == "__main__":
    x = [torch.rand((1, 128, 52, 52)),
         torch.rand((1, 256, 26, 26)),
         torch.rand((1, 1024, 13, 13))]
    mpan = Modified_PAN()
    [c4,c5,c6] = mpan(x)
    print(c4.shape)
    print(c5.shape)
    print(c6.shape)