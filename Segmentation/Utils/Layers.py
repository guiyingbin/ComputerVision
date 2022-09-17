from collections import OrderedDict
import torch
import torch.nn as nn


def build_block(block_list: list, activation_list: list = ["LeakyReLU", 0.2]) -> nn:
    """

    :param norm_list:
    :param activation_list:
    :param block_list: Conv: ["Conv", in_channels, out_channels, kernel_size, padding, stride]
                       MaxPool2d: ["MaxPool" kernel_size, stride]
    :return: nn
    """
    block = nn.Sequential()
    for i, block_info in enumerate(block_list):
        block_type = block_info[0]
        if block_type == "FC":
            block.add_module("FC{}".format(i), nn.Linear(block_info[1], block_info[2]))

        if block_type == "Conv":
            block.add_module("Conv{}".format(i), nn.Conv2d(in_channels=block_info[1],
                                                           out_channels=block_info[2],
                                                           kernel_size=block_info[3],
                                                           padding=block_info[4],
                                                           stride=block_info[5]))
        if block_type == "MaxPool":
            block.add_module("MaxPool{}".format(i), nn.MaxPool2d(kernel_size=block_info[1],
                                                                 padding=block_info[2],
                                                                 stride=block_info[3]))

        if block_type == "AvgPool":
            block.add_module("AvgPool{}".format(i), nn.AvgPool2d(kernel_size=block_info[1],
                                                                 stride=block_info[2]))

        if block_type == "BatchNorm":
            block.add_module("BatchNorm{}".format(i), nn.BatchNorm2d(num_features=block_info[1]))
            block.add_module("{}{}".format(activation_list[0], i), build_activation(activation_list))

        if block_type == "UpNearest":
            block.add_module("{}{}".format(block_info[0], i), nn.UpsamplingNearest2d(scale_factor=block_info[1]))

        if block_type == "ConvTranpose":
            block.add_module("{}{}".format(block_info[0], i), nn.ConvTranspose2d(in_channels=block_info[1],
                                                                                 out_channels=block_info[2],
                                                                                 kernel_size=block_info[3],
                                                                                 stride=block_info[4]))

        if block_type == "FPN":
            block.add_module("{}{}".format(block_info[0], i), FPN(neck_config=block_info[1],
                                                                  channels=block_info[2],
                                                                  activation_list=activation_list))
        if block_type == "ASF":
            block.add_module("{}{}".format(block_info[0], i), AdaptiveScaleFusion(N=block_info[1],
                                                                                  C=block_info[2]))

        if block_type == "FPEM":
            block.add_module("{}{}".format(block_info[0], i), FeaturePyramidEnhancement(input_channels=block_info[1],
                                                                                        pre_conv=block_info[2]))

        if block_type == "FFM":
            block.add_module("{}{}".format(block_info[0], i), FeatureFusion())

    return block


def build_activation(activation_list: list) -> nn:
    """
    get the activation layer by the name
    :param activation_list:
    :return: nn
    """
    activation_name = activation_list[0]
    if activation_name == "ReLU":
        return nn.ReLU()
    elif activation_name == "LeakyReLU":
        return nn.LeakyReLU(*activation_list[1:])
    elif activation_name == "Sigmoid":
        return nn.Sigmoid()
    elif activation_name == "Tanh":
        return nn.Tanh()
    else:
        return nn.Identity()


class AdaptiveScaleFusion(nn.Module):
    def __init__(self, N=4, C=1024):
        """
        the Adaptive Scale Fusion Module of DBNet++
        :param N: the number of feature maps from different layers
        :param C: the channels of feature maps
        """
        super(AdaptiveScaleFusion, self).__init__()
        self.N = N
        self.C = C
        self.SA = SpatialAttention(input_channel=C, output_channel=N)
        self.conv1 = nn.Conv2d(self.N*self.C, self.C, 3, 1, 1)

    def forward(self, x):
        """
        :param x: shape is [B, N*C, H, W]
        :return: shape is [B, N*C, H, W]
        """
        B, N_C, H, W = x.shape
        x_part1 = self.conv1(x)
        weights = self.SA(x_part1) #weights shape is [B, N, H, W]
        weights = weights.unsqueeze(dim=2) #weights shape is [B, N, 1, H, W]
        x = x.reshape(B, self.N, self.C, H, W)
        output = x*weights
        output = output.reshape(B, self.N*self.C, H, W)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, input_channel=4, output_channel=4):
        """
        the Spatial Attention Module of DBNet++
        :param input_channel: the input channels of SA
        :param output_channel: the output channels of SA
        """
        super(SpatialAttention, self).__init__()
        self.conv_part1 = nn.Sequential(nn.Conv2d(1, 1, 1, 1, 0),
                                        nn.ReLU(),
                                        nn.Conv2d(1, 1, 3, 1, 1),
                                        nn.Sigmoid())
        self.conv_part2 = nn.Sequential(nn.Conv2d(input_channel, output_channel, 3, 1, 1),
                                        nn.Sigmoid())

    def forward(self, x:torch.FloatTensor):
        x_part1 = torch.mean(input=x, dim=1)
        x_part1 = x_part1.unsqueeze(dim=1)
        x_part1 = self.conv_part1(x_part1)
        x = x_part1 + x
        output = self.conv_part2(x)
        return output


class FPN(nn.Module):
    def __init__(self, neck_config, channels, hidden_channel=256, activation_list=None):
        """

        :param neck_config:
        :param channels:
        :param activation_list:
        """
        super(FPN, self).__init__()
        self.neck = {}
        for block_name, block_list in neck_config.items():
            self.neck[block_name] = build_block(block_list, activation_list=activation_list)

        self.pre_layer = [nn.Conv2d(channels[i], hidden_channel, kernel_size=3, padding=1, stride=1) for i in range(0, 3)]

    def forward(self, f):
        f2, f3, f4, f5 = f
        p5 = self.neck["P5"](f5)
        f4 = self.pre_layer[2](f4)
        p5_up = self.neck["P5_up"](p5)
        p4 = self.neck["P4"](p5_up + f4)
        f3 = self.pre_layer[1](f3)
        p5_up = self.neck["P4_up"](p5_up)
        p4_up = self.neck["P4_up"](p4)
        p3 = self.neck["P3"](p4_up + f3)
        f2 = self.pre_layer[0](f2)
        p5_up = self.neck["P3_up"](p5_up)
        p4_up = self.neck["P3_up"](p4_up)
        p3_up = self.neck["P3_up"](p3)
        p2 = self.neck["P2"](p3_up + f2)

        p5 = p5_up
        p4 = p4_up
        p3 = p3_up
        return p2, p3, p4, p5


class FeaturePyramidEnhancement(nn.Module):
    def __init__(self, input_channels=None, pre_conv=True):
        """
        Feature Pyramid Enhancement Module of PANNet
        :param input_channels:
        """
        super(FeaturePyramidEnhancement, self).__init__()
        if input_channels is None:
            input_channels = [256, 512, 1024, 2048]

        self.input_channels = input_channels
        self.upsample =  nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_block_upsample = []
        self.conv_block_downsample = []
        self.is_preconv = pre_conv
        if self.is_preconv:
            self.pre_conv = [nn.Conv2d(input_channel, 128, 1, 1, 0) for input_channel in input_channels]

        for input_channel in input_channels[1:]:
            self.conv_block_upsample.append(nn.Sequential(nn.Conv2d(128,128,
                                                                    groups=128,
                                                                    kernel_size=3,
                                                                    padding=1,
                                                                    stride=1),
                                                          nn.Conv2d(128, 128, 1, 1, 0),
                                                          nn.BatchNorm2d(128),
                                                          nn.ReLU()))
            self.conv_block_downsample.append(nn.Sequential(nn.Conv2d(128, 128,
                                                                    groups=128,
                                                                    kernel_size=3,
                                                                    padding=1,
                                                                    stride=2),
                                                          nn.Conv2d(128, 128, 1, 1, 0),
                                                          nn.BatchNorm2d(128),
                                                          nn.ReLU()))

    def forward(self, x):
        assert len(self.input_channels) == len(x)
        f2, f3, f4, f5 = x
        if self.is_preconv:
            f2 = self.pre_conv[0](f2)
            f3 = self.pre_conv[1](f3)
            f4 = self.pre_conv[2](f4)
            f5 = self.pre_conv[3](f5)

        # Up-scale enhancement
        p5 = f5
        p5_up = self.upsample(p5)
        p4 = self.conv_block_upsample[-1](p5_up+f4)
        p4_up = self.upsample(p4)
        p3 = self.conv_block_upsample[-2](p4_up+f3)
        p3_up = self.upsample(p3)
        p2 = self.conv_block_upsample[-3](p3_up+f2)

        # Down-scale enhancement
        o2 = p2
        o3 = self.conv_block_downsample[-1](p3_up+o2)
        o4 = self.conv_block_downsample[-2](p4_up+o3)
        o5 = self.conv_block_downsample[-3](p5_up+o4)
        return o2, o3, o4, o5


class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.upsample_2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_8x = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, x):
        o2, o3, o4, o5 = x
        o5 = self.upsample_8x(o5)
        o4 = self.upsample_4x(o4)
        o3 = self.upsample_2x(o3)
        output = torch.cat([o2, o3, o4, o5], dim=1)
        return output

if __name__ == "__main__":
    a = torch.rand((1, 256, 160, 160))
    b = torch.rand((1, 512, 80, 80))
    c = torch.rand((1, 1024, 40, 40))
    d = torch.rand((1, 2048, 20, 20))
    model = FeaturePyramidEnhancement()
    o2, o3, o4, o5 = model([a,b,c,d])
    print(o2.shape)