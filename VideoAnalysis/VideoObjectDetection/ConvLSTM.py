import torch.nn as nn
import torch
from torch.autograd import Variable

class BottleneckLSTM(nn.Module):
    """
    The Bottleneck LSTM module in paper:https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Mobile_Video_Object_CVPR_2018_paper.pdf
    """
    def __init__(self, input_channel, output_channel, input_shape, device):
        super(BottleneckLSTM, self).__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        self.conv_x = nn.Conv2d(in_channels=self.in_channel,
                                out_channels=self.in_channel,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=self.in_channel)
        self.conv_bottleneck = nn.Conv2d(in_channels=(self.in_channel+self.out_channel),
                                         out_channels=self.out_channel,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.conv_forgetgate = nn.Conv2d(in_channels=self.out_channel,
                                         out_channels=self.out_channel,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.conv_inputgate = nn.Conv2d(in_channels=self.out_channel,
                                        out_channels=self.out_channel,
                                        kernel_size=3,
                                        stride=1,
                                        padding=0,
                                        groups=self.out_channel)
        self.conv = nn.Conv2d(in_channels=self.out_channel,
                              out_channels=self.out_channel,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.conv_outputgate = nn.Conv2d(in_channels=self.out_channel,
                                         out_channels=self.out_channel,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.c, self.h = self.init_state(input_shape, device)
        self.relu = nn.ReLU()
    def init_state(self, shape, device):
        """

        :param shape: (B, C, H, W), for example (1, 3, 224, 224)
        :param device: cpu or cuda
        :return: c, h
        """
        return Variable(torch.zeros(size=shape)).to(device), Variable(torch.zeros(size=shape)).to(device)

    def cell_forward(self, x, c, h):
        h_x = self.conv_x(x)
        h_b = self.conv_bottleneck(torch.cat([h_x, h], dim=1))
        c_1 = torch.sigmoid(self.conv_forgetgate(h_b))
        h_i = torch.sigmoid(self.conv_inputgate(h_b))
        h_ = self.conv(h_b)
        h_o = self.conv_outputgate(h_b)
        c_2 = h_i * h_
        c_t = c * c_1 + c_2
        h_t = self.relu(h_o)*c_t
        return c_t, h_t

    def forward(self, x):
        c_t, h_t = self.cell_forward(x, self.c, self.h)
        self.c = c_t
        self.h = h_t
        return h_t

if __name__ == "__main__":
    a = nn.Conv2d(in_channels=3, out_channels=12, groups=3, kernel_size=3, padding=1)
    inp = torch.randn(size=(1, 3, 224, 224))
    print(a(inp).shape)