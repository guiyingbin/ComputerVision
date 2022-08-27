from collections import OrderedDict

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
                                                                 stride=block_info[2]))

        if block_type == "AvgPool":
            block.add_module("AvgPool{}".format(i), nn.AvgPool2d(kernel_size=block_info[1],
                                                                 stride=block_info[2]))

        if block_type == "BatchNorm":
            block.add_module("BatchNorm{}".format(i), nn.BatchNorm2d(num_features=block_info[1]))
            block.add_module("{}{}".format(activation_list[0], i), build_activation(activation_list))

        if block_type == "Residual":
            block.add_module("{}{}".format(block_info[0],i), Residual(input_channel=block_info[1],
                                                                      mid_channel=block_info[2],
                                                                      kernel_size=block_info[3],
                                                                      padding=block_info[4],
                                                                      stride=block_info[5],
                                                                      activation_list=activation_list))
        if block_type == "Darknet_block":
            block.add_module("{}{}".format(block_info[0],i), Darknet_block(input_channel=block_info[1],
                                                                           output_channel=block_info[2],
                                                                           activation_list=activation_list))
        if block_type == "UpNearest":
            block.add_module("{}{}".format(block_info[0], i), nn.UpsamplingNearest2d(scale_factor=block_info[1]))

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


class IntermediateLayerGetter(nn.ModuleDict):
    """ get the output of certain layers """

    def __init__(self, model, return_layers):
        # 判断传入的return_layers是否存在于model中
        if not set(return_layers).issubset([name for name, _ in model.named_modules()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}  # 构造dict
        layers = OrderedDict()
        # 将要从model中获取信息的最后一层之前的模块全部复制下来
        for name, module in model.named_modules():
            if name== "":
                continue
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)  # 将所需的网络层通过继承的方式保存下来
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 将所需的值以k,v的形式保存到out中
        for name, module in self.named_modules():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


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
        output = latent+x
        return output


class Darknet_block(nn.Module):
    def __init__(self, input_channel, output_channel, activation_list=["LeakyReLU", 0.2]):
        super(Darknet_block, self).__init__()
        self.darknet_block_list = [["Conv", input_channel, input_channel//2, 1, 0, 1],
                                   ["BatchNorm", input_channel//2],
                                   ["Conv", input_channel//2, input_channel, 3, 1, 1],
                                   ["BatchNorm", input_channel],
                                   ["Conv", input_channel, input_channel // 2, 1, 0, 1],
                                   ["BatchNorm", input_channel // 2],
                                   ["Conv", input_channel // 2, input_channel, 3, 1, 1],
                                   ["BatchNorm", input_channel],
                                   ["Conv", input_channel, output_channel, 1, 0, 1],
                                   ["BatchNorm", output_channel]]
        self.darknet_block = build_block(block_list=self.darknet_block_list, activation_list=activation_list)

    def forward(self, x):
        return self.darknet_block(x)
