from collections import OrderedDict
import torch
import torch.nn as nn


def build_block(block_list: list, activation_list: list = ["LeakyReLU", 0.2]) -> nn:
    """
    build the basic block for all neural network
    :param activation_list: configuration of convolutional layer
    :param block_list: Conv: ["Conv", in_channels, out_channels, kernel_size, padding, stride]
                       MaxPool2d: ["MaxPool" kernel_size, padding, stride]
                       FC: ["FC", in_channels, out_channels]
                       AvgPool: ["AvgPool", kernel_size, stride]
                       BatchNorm: ["BatchNorm", num_feature]
                       BottleNeck: [input_channel,  mode=1, neck_type activation_list, mid_channel,  stride]
                                   [input_channel,  mode=2, neck_type (activation_list)]
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

        if block_type == "BottleNeck":
            block.add_module("{}{}".format(block_type, i), BottleNeck(*block_info[1:]))
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


class BottleNeck(nn.Module):
    def __init__(self, input_channel, mode=1, neck_type=2, activation_list=["ReLU"], mid_channel=None, stride=None):
        """
        BottleNeck block of ResNet
        reference: https://zhuanlan.zhihu.com/p/353235794?ivk_sa=1024320u
        :param input_channel: the input channel
        :param neck_type: 1 means bottleNeck for resnet 18 and 32, 2 means bottleNeck for resnet 50, 101 and 152
        :param mid_channel: only work for mode 1
        :param stride: only work for mode 1
        :param mode: 1 means BTNK1 in refernce, 2 means BTNK2 in reference
        :param activation_list: configuration of activation
        """
        super(BottleNeck, self).__init__()
        self.neck_type = neck_type
        self.mode = mode
        if self.mode == 1:
            assert mid_channel is not None
            assert stride is not None
            self.conv_part1_list = [["Conv", input_channel, mid_channel, 1, 0, stride],
                                    ["BatchNorm", mid_channel],
                                    ["Conv", mid_channel, mid_channel, 3, 1, 1],
                                    ["BatchNorm", mid_channel]]
            if neck_type == 2:
                self.conv_part1_list.extend([["Conv", mid_channel, mid_channel * 4, 1, 0, 1],
                                             ["BatchNorm", mid_channel * 4]])

                self.conv_part2_list = [["Conv", input_channel, mid_channel * 4, 1, 0, stride]]
            else:
                self.conv_part2_list = [["Conv", input_channel, mid_channel, 1, 0, stride]]

            self.conv_part1 = build_block(self.conv_part1_list, activation_list)
            self.conv_part2 = build_block(self.conv_part2_list, activation_list)
        else:
            self.conv_part1_list = [["Conv", input_channel, input_channel, 1, 0, 1],
                                    ["BatchNorm", input_channel],
                                    ["Conv", input_channel, input_channel, 3, 1, 1],
                                    ["BatchNorm", input_channel]]
            if neck_type == 2:
                self.conv_part1_list.extend([["Conv", input_channel, input_channel, 1, 0, 1],
                                             ["BatchNorm", input_channel]])

            self.conv_part1 = build_block(self.conv_part1_list, activation_list)
            self.conv_part2 = nn.Identity()

    def forward(self, x):
        output_part1 = self.conv_part1(x)
        output_part2 = self.conv_part2(x)
        output = output_part1 + output_part2
        return output_part1
