from collections import OrderedDict
import torch
import torch.nn as nn
from ObjectDetection.Utils.NeckLayer import *

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

        if block_type == "Residual":
            block.add_module("{}{}".format(block_info[0], i), Residual(input_channel=block_info[1],
                                                                       mid_channel=block_info[2],
                                                                       kernel_size=block_info[3],
                                                                       padding=block_info[4],
                                                                       stride=block_info[5],
                                                                       activation_list=activation_list))
        if block_type == "ConvSet_block":
            block.add_module("{}{}".format(block_info[0], i), ConvSet_block(input_channel=block_info[1],
                                                                            output_channel=block_info[2],
                                                                            n_block=block_info[3],
                                                                            activation_list=activation_list))
        if block_type == "UpNearest":
            block.add_module("{}{}".format(block_info[0], i), nn.UpsamplingNearest2d(scale_factor=block_info[1]))

        if block_type == "DarkNet_block":
            block.add_module("{}{}".format(block_info[0], i), DarkNet_block(input_channel=block_info[1],
                                                                            n_block=block_info[2],
                                                                            activation_list=activation_list))
        if block_type == "CSPDarkNet_block":
            block.add_module("{}{}".format(block_info[0], i), CSPDarkNet_block(input_channel=block_info[1],
                                                                               output_channel=block_info[2],
                                                                               n_block=block_info[3],
                                                                               csp_mode=block_info[4],
                                                                               activation_list=activation_list))
        if block_type == "SPP":
            block.add_module("{}{}".format(block_info[0], i), SPP(input_channel=block_info[1]))
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
    if activation_name == "LeakyReLU":
        return nn.LeakyReLU(*activation_list[1:])
    if activation_name == "Sigmoid":
        return nn.Sigmoid()
    if activation_name == "Tanh":
        return nn.Tanh()

    return nn.ReLU()


class DarkNet_block(nn.Module):
    def __init__(self, input_channel, n_block, activation_list=["LeakyReLU", 0.2]):
        super(DarkNet_block, self).__init__()
        self.darknet_block_list = [["Conv", input_channel, input_channel//2, 1, 0, 1],
                                   ["BatchNorm", input_channel//2],
                                   ["Conv", input_channel//2, input_channel, 3, 1, 1],
                                   ["BatchNorm", input_channel],
                                   ["Residual", input_channel, input_channel*2, 3, 1, 1]]*n_block
        self.darknet_block = build_block(self.darknet_block_list, activation_list)

    def forward(self, x):
        return self.darknet_block(x)


class Residual(nn.Module):
    def __init__(self, input_channel, mid_channel, kernel_size, padding, stride, activation_list=["LeakyReLU", 0.2]):
        super(Residual, self).__init__()
        self.activation = build_activation(activation_list)
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=input_channel,
                                              out_channels=mid_channel,
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              stride=stride),
                                    nn.BatchNorm2d(mid_channel),
                                    self.activation)
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=mid_channel,
                                              out_channels=input_channel,
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              stride=stride),
                                    nn.BatchNorm2d(input_channel),
                                    self.activation)

    def forward(self, x):
        latent = self.block1(x)
        latent = self.block2(latent)
        output = latent + x
        return output


class ConvSet_block(nn.Module):
    def __init__(self, input_channel, output_channel, n_block=1, activation_list=["LeakyReLU", 0.2]):
        super(ConvSet_block, self).__init__()
        if n_block>1:
            assert input_channel == output_channel
        self.darknet_block_list = [["Conv", input_channel, input_channel // 2, 1, 0, 1],
                                   ["BatchNorm", input_channel // 2],
                                   ["Conv", input_channel // 2, input_channel, 3, 1, 1],
                                   ["BatchNorm", input_channel],
                                   ["Conv", input_channel, output_channel, 1, 0, 1],
                                   ["BatchNorm", output_channel],
                                   ["Conv", output_channel, output_channel * 2, 3, 1, 1],
                                   ["BatchNorm", output_channel * 2],
                                   ["Conv", output_channel * 2, output_channel, 1, 0, 1],
                                   ["BatchNorm", output_channel]] * n_block
        self.darknet_block = build_block(block_list=self.darknet_block_list, activation_list=activation_list)

    def forward(self, x):
        return self.darknet_block(x)


class CSPDarkNet_block(nn.Module):
    def __init__(self, input_channel, output_channel, n_block=1, activation_list=["LeakyReLU", 0.2],
                 csp_mode="fusion_last"):
        """
        In CSPDarknet Block, The input x is divided into two parts along the channel direction.
        The Paper url:
        https://openaccess.thecvf.com/content_CVPRW_2020/html/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.html
        :param input_channel: the number of input channel
        :param output_channel: the output of output channel
        :param activation_list: the setting of activation layer
        :param csp_mode: fusion_last or fusion_first, the detail can be found in paper of CSPNet
        """
        super(CSPDarkNet_block, self).__init__()
        self.csp_mode = csp_mode
        assert output_channel > input_channel // 2
        if self.csp_mode == "fusion_last":
            self.transition_layer = nn.Conv2d(input_channel // 2, output_channel - input_channel // 2,
                                              kernel_size=1)
        else:
            self.transition_layer = nn.Conv2d(input_channel, output_channel, kernel_size=1)
        self.darknet_block = DarkNet_block(input_channel // 2, n_block, activation_list=activation_list)

    def forward(self, x):
        B, C, H, W = x.shape
        x_part1 = x[:, :C // 2, :, :]
        x_part2 = x[:, C // 2:, :, :]
        if self.csp_mode == "fusion_last":
            x_part2 = self.darknet_block(x_part2)
            x_part2 = self.transition_layer(x_part2)
            output = torch.cat([x_part1, x_part2], dim=1)
        else:
            x_part2 = self.darknet_block(x_part2)
            output = torch.cat([x_part1, x_part2], dim=1)
            output = self.transition_layer(output)
        return output


if __name__ == "__main__":
    x = torch.rand((3, 512, 12, 12))
    csp_darnet_block = CSPDarkNet_block(512, 512)
    output = csp_darnet_block(x)
    print(output.shape)
