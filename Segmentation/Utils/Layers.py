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


class FPN(nn.Module):
    def __init__(self, neck_config, channels, activation_list=None):
        super(FPN, self).__init__()
        self.neck = {}
        for block_name, block_list in neck_config.items():
            self.neck[block_name] = build_block(block_list, activation_list=activation_list)

        self.pre_layer = [nn.Conv2d(channels[i], 256, kernel_size=3, padding=1, stride=1) for i in range(0, 3)]

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