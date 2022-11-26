from collections import OrderedDict
import torch
import torch.nn as nn
from Segmentation.Utils.Layers import FPN


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

        elif block_type == "Conv":
            block.add_module("Conv{}".format(i), nn.Conv2d(in_channels=block_info[1],
                                                           out_channels=block_info[2],
                                                           kernel_size=block_info[3],
                                                           padding=block_info[4],
                                                           stride=block_info[5]))
        elif block_type == "MaxPool":
            block.add_module("MaxPool{}".format(i), nn.MaxPool2d(kernel_size=block_info[1],
                                                                 padding=block_info[2],
                                                                 stride=block_info[3]))

        elif block_type == "AvgPool":
            block.add_module("AvgPool{}".format(i), nn.AvgPool2d(kernel_size=block_info[1],
                                                                 stride=block_info[2]))

        elif block_type == "BatchNorm":
            block.add_module("BatchNorm{}".format(i), nn.BatchNorm2d(num_features=block_info[1]))
            block.add_module("{}{}".format(activation_list[0], i), build_activation(activation_list))
            """
            The following are non-basic modules. When building non-basic modules, 
            if you use build_block to build, do not form loop nesting
            """
        elif block_type == "Residual":
            block.add_module("{}{}".format(block_info[0], i), Residual(input_channel=block_info[1],
                                                                       mid_channel=block_info[2],
                                                                       activation_list=activation_list))
        elif block_type == "ConvSet_block":
            block.add_module("{}{}".format(block_info[0], i), ConvSet_block(input_channel=block_info[1],
                                                                            output_channel=block_info[2],
                                                                            n_block=block_info[3],
                                                                            activation_list=activation_list))
        elif block_type == "UpNearest":
            block.add_module("{}{}".format(block_info[0], i), nn.UpsamplingNearest2d(scale_factor=block_info[1]))

        elif block_type == "DarkNet_block":
            block.add_module("{}{}".format(block_info[0], i), DarkNet_block(input_channel=block_info[1],
                                                                            n_block=block_info[2],
                                                                            activation_list=activation_list))
        elif block_type == "CSPDarkNet_block":
            block.add_module("{}{}".format(block_info[0], i), CSPDarkNet_block(input_channel=block_info[1],
                                                                               output_channel=block_info[2],
                                                                               n_block=block_info[3],
                                                                               csp_mode=block_info[4],
                                                                               activation_list=activation_list))
        elif block_type == "SPP":
            block.add_module("{}{}".format(block_info[0], i), SPP(input_channel=block_info[1],
                                                                  activation_list=activation_list))

        elif block_type == "SPP_CSP_Conv":
            block.add_module("{}{}".format(block_info[0], i), SPP_CSP_Conv(input_channel=block_info[1],
                                                                           activation_list=activation_list))

        elif block_type == "CSP1_block":
            block.add_module("{}{}".format(block_info[0], i), CSP1_block(input_channel=block_info[1],
                                                                         output_channel=block_info[2],
                                                                         n_block=block_info[3],
                                                                         activation_list=activation_list))
        elif block_type == "CSP2_block":
            block.add_module("{}{}".format(block_info[0], i), CSP2_block(input_channel=block_info[1],
                                                                         output_channel=block_info[2],
                                                                         n_block=block_info[3],
                                                                         activation_list=activation_list))
        elif block_type == "Focus":
            block.add_module("{}{}".format(block_info[0], i), Focus(input_channel=block_info[1],
                                                                    output_channel=block_info[2],
                                                                    activation_list=activation_list))

        elif block_type == "PAN":
            block.add_module("{}{}".format(block_info[0], i), PAN(block_info[1], block_info[2]))

        elif block_type == "ELAN":
            block.add_module("{}{}".format(block_info[0], i), ELAN(block_info[1], block_info[2]))

        elif block_type == "MP_block":
            block.add_module("{}{}".format(block_info[0], i), MP_block(block_info[1], block_info[2]))

        elif block_type == "RepConv":
            block.add_module("{}{}".format(block_info[0], i), RepConv(block_info[1]))
        elif block_type == "FPN_3":
            block.add_module("{}{}".format(block_info[0], i), FPN_3Layer(neck_config=block_info[1],
                                                                         channels=block_info[2],
                                                                         activation_list=activation_list))
        else:
            block.add_module("Identity{}".format(i), nn.Identity())
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
    elif activation_name == "Mish":
        return Mish()
    elif activation_name == "Silu":
        return Silu()
    else:
        return nn.Identity()


class Silu(nn.Module):
    def __init__(self):
        """
        the SiLU activation function
        x * sigmoid(x)
        """
        super(Silu, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def __init__(self):
        """
        the Mish activation function
        x * torch.tanh(log(1+e^x))
        """
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.exp(x)))


class SPP_CSP_Conv(nn.Module):
    def __init__(self, input_channel, activation_list=["Silu"]):
        """
        the SPP_CSP_block in yolo V7
        :param input_channel:
        :param output_channel:
        :param activation_list:
        """
        super(SPP_CSP_Conv, self).__init__()
        conv_part1_list = [["Conv", input_channel, input_channel // 2, 1, 0, 1],
                           ["BatchNorm", input_channel // 2]]
        conv_part2_list = [["Conv", input_channel, input_channel // 2, 1, 0, 1],
                           ["BatchNorm", input_channel // 2],
                           ["Conv", input_channel // 2, input_channel // 2, 3, 1, 1],
                           ["BatchNorm", input_channel // 2],
                           ["Conv", input_channel // 2, input_channel // 2, 1, 0, 1],
                           ["BatchNorm", input_channel // 2],
                           ["SPP", input_channel // 2],
                           ["Conv", input_channel // 2, input_channel // 2, 3, 1, 1]]

        conv_part3_list = [["Conv", input_channel, input_channel // 2, 3, 1, 1],
                           ["BatchNorm", input_channel // 2]]
        self.conv_part1 = build_block(conv_part1_list, activation_list)
        self.conv_part2 = build_block(conv_part2_list, activation_list)
        self.conv_part3 = build_block(conv_part3_list, activation_list)

    def forward(self, x):
        output_part1 = self.conv_part1(x)
        output_part2 = self.conv_part2(x)
        output = torch.cat([output_part1, output_part2], dim=1)
        output = self.conv_part3(output)
        return output


class MP_block(nn.Module):
    def __init__(self, input_channel, mode=1, activation_list=["Silu"]):
        """
        the MP block of yolo v7
        reference:https://blog.csdn.net/u012863603/article/details/126118799
        :param input_channel: the input channel
        :param mode: If mode is equal, then the number of output channels is input_channel, otherwise it is 2*input_channel
        :param activation_list: the configuration of activation layer
        """
        super(MP_block, self).__init__()

        if mode == 1:
            conv_part1_list = [["MaxPool", 3, 1, 2],
                               ["Conv", input_channel, input_channel // 2, 1, 0, 1]]
            conv_part2_list = [["Conv", input_channel, input_channel // 2, 1, 0, 1],
                               ["Conv", input_channel // 2, input_channel // 2, 3, 1, 2]]
        else:
            conv_part1_list = [["MaxPool", 3, 1, 2],
                               ["Conv", input_channel, input_channel, 1, 0, 1]]
            conv_part2_list = [["Conv", input_channel, input_channel, 1, 0, 1],
                               ["Conv", input_channel, input_channel, 3, 1, 2]]

        self.conv_part1 = build_block(conv_part1_list, activation_list)
        self.conv_part2 = build_block(conv_part2_list, activation_list)

    def forward(self, x):
        output_part1 = self.conv_part1(x)
        output_part2 = self.conv_part2(x)
        output = torch.cat([output_part1, output_part2], dim=1)
        return output


class ELAN(nn.Module):
    def __init__(self, input_channel, mode=None, activation_list=["Silu"]):
        """
        The ELAN block of yoloV7
        :param input_channel: the input channel of image
        :param mode: if mode==W, the ELAN-W block will be applied
        :param activation_list: activation layer configuration
        """
        super(ELAN, self).__init__()
        self.mode = mode
        conv_part1_list = [["Conv", input_channel, input_channel // 2, 1, 0, 1],
                           ["BatchNorm", input_channel // 2]]
        if self.mode == "W":
            conv_part2_list = [
                [["Conv", input_channel, input_channel // 2, 1, 0, 1],
                 ["BatchNorm", input_channel // 2]],
                [["Conv", input_channel // 2, input_channel // 4, 3, 1, 1],
                 ["BatchNorm", input_channel // 4]],
                [["Conv", input_channel // 4, input_channel // 4, 3, 1, 1],
                 ["BatchNorm", input_channel // 4]],
                [["Conv", input_channel // 4, input_channel // 4, 3, 1, 1],
                 ["BatchNorm", input_channel // 4]],
                [["Conv", input_channel // 4, input_channel // 4, 3, 1, 1],
                 ["BatchNorm", input_channel // 4]]
            ]

            conv_part3_list = [["Conv", 2 * input_channel, input_channel // 2, 1, 0, 1],
                               ["BatchNorm", input_channel // 2]]
        else:
            conv_part2_list = [
                [["Conv", input_channel, input_channel // 2, 1, 0, 1],
                 ["BatchNorm", input_channel // 2]],
                [["Conv", input_channel // 2, input_channel // 2, 3, 1, 1],
                 ["BatchNorm", input_channel // 2],
                 ["Conv", input_channel // 2, input_channel // 2, 3, 1, 1],
                 ["BatchNorm", input_channel // 2]],
                [["Conv", input_channel // 2, input_channel // 2, 3, 1, 1],
                 ["BatchNorm", input_channel // 2],
                 ["Conv", input_channel // 2, input_channel // 2, 3, 1, 1],
                 ["BatchNorm", input_channel // 2]]
            ]

            conv_part3_list = [["Conv", 2 * input_channel, 2 * input_channel, 1, 0, 1],
                               ["BatchNorm", 2 * input_channel]]
        self.conv_part1 = build_block(conv_part1_list, activation_list)
        self.conv_part2 = [build_block(block_list, activation_list) for block_list in conv_part2_list]
        self.conv_part3 = build_block(conv_part3_list, activation_list)

    def forward(self, x):
        output_part1 = self.conv_part1(x)
        output_part2 = [output_part1]
        input_part2 = x
        for conv_block in self.conv_part2:
            temp_output = conv_block(input_part2)
            output_part2.append(temp_output)
            input_part2 = temp_output

        output = torch.cat(output_part2, dim=1)
        output = self.conv_part3(output)
        return output


class DarkNet_block(nn.Module):
    def __init__(self, input_channel, n_block, activation_list=["LeakyReLU", 0.2]):
        """
        the basic block of DarkNet
        :param input_channel:
        :param n_block:
        :param activation_list:
        """
        super(DarkNet_block, self).__init__()
        self.darknet_block_list = [["Conv", input_channel, input_channel // 2, 1, 0, 1],
                                   ["BatchNorm", input_channel // 2],
                                   ["Conv", input_channel // 2, input_channel, 3, 1, 1],
                                   ["BatchNorm", input_channel],
                                   ["Residual", input_channel, input_channel * 2, 1]] * n_block
        self.darknet_block = build_block(self.darknet_block_list, activation_list)

    def forward(self, x):
        return self.darknet_block(x)


class Residual(nn.Module):
    def __init__(self, input_channel, mid_channel, n_block=1, activation_list=["LeakyReLU", 0.2]):
        """
        the Residual Block of yoloV3
        :param input_channel:
        :param mid_channel:
        :param kernel_size:
        :param padding:
        :param stride:
        :param activation_list:
        """
        super(Residual, self).__init__()
        self.activation = build_activation(activation_list)
        self.block_list = [["Conv", input_channel, mid_channel, 3, 1, 1],
                           ["BatchNorm", mid_channel],
                           ["Conv", mid_channel, input_channel, 1, 0, 1],
                           ["BatchNorm", input_channel]] * n_block
        self.block = build_block(self.block_list, activation_list=activation_list)

    def forward(self, x):
        latent = self.block(x)
        output = latent + x
        return output


class ConvSet_block(nn.Module):
    def __init__(self, input_channel, output_channel, n_block=1, activation_list=["LeakyReLU", 0.2]):
        super(ConvSet_block, self).__init__()
        if n_block > 1:
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


class CSP1_block(nn.Module):
    def __init__(self, input_channel, output_channel, n_block=1, activation_list=["LeakyReLU", 0.2]):
        """
        CSP1_X in yoloV5s.
        reference:https://zhuanlan.zhihu.com/p/272726675
        :param input_channel:
        :param output_channel:
        :param n_block:
        :param activation_list:
        """
        super(CSP1_block, self).__init__()
        csp1_block1_list = [["Conv", input_channel, output_channel // 2, 1, 0, 1],
                            ["BatchNorm", output_channel // 2]]
        csp1_block2_list = [["BatchNorm", output_channel],
                            ["Conv", output_channel, output_channel, 1, 0, 1],
                            ["BatchNorm", output_channel]]
        self.csp1_block1 = build_block(csp1_block1_list, activation_list=activation_list)
        self.csp1_block2 = build_block(csp1_block2_list, activation_list=activation_list)

        self.residual = Residual(output_channel // 2, output_channel, n_block=n_block,
                                 activation_list=activation_list)
        self.conv1 = nn.Conv2d(output_channel // 2, output_channel // 2, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(input_channel, output_channel // 2, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x_part1 = x
        x_part2 = x
        x_part1 = self.csp1_block1(x_part1)
        x_part1 = self.residual(x_part1)
        x_part1 = self.conv1(x_part1)

        x_part2 = self.conv2(x_part2)
        output = torch.cat([x_part1, x_part2], dim=1)
        output = self.csp1_block2(output)
        return output


class CSP2_block(nn.Module):
    def __init__(self, input_channel, output_channel, n_block=1, activation_list=["LeakyReLU", 0.2]):
        """
        CSP2_X in yoloV5s.
        reference:https://zhuanlan.zhihu.com/p/272726675
        :param input_channel:
        :param output_channel:
        :param n_block:
        :param activation_list:
        """
        super(CSP2_block, self).__init__()
        csp2_block1_list = [["Conv", input_channel, output_channel // 2, 1, 0, 1],
                            ["BatchNorm", output_channel // 2]]
        csp2_block2_list = [["Conv", output_channel // 2, output_channel // 2, 1, 0, 1],
                            ["BatchNorm", output_channel // 2],
                            ["Conv", output_channel // 2, output_channel // 2, 3, 1, 1],
                            ["BatchNorm", output_channel // 2]] * n_block

        csp2_block3_list = [["BatchNorm", output_channel],
                            ["Conv", output_channel, output_channel, 1, 0, 1],
                            ["BatchNorm", output_channel]]
        self.csp2_block1 = build_block(csp2_block1_list, activation_list=activation_list)
        self.csp2_block2 = build_block(csp2_block2_list, activation_list=activation_list)
        self.csp3_block3 = build_block(csp2_block3_list, activation_list=activation_list)
        self.conv1 = nn.Conv2d(output_channel // 2, output_channel // 2, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(input_channel, output_channel // 2, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x_part1 = x
        x_part2 = x
        x_part1 = self.csp2_block1(x_part1)
        x_part1 = self.csp2_block2(x_part1)
        x_part1 = self.conv1(x_part1)
        x_part2 = self.conv2(x_part2)
        output = torch.cat([x_part1, x_part2], dim=1)
        output = self.csp3_block3(output)
        return output


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


class Focus(nn.Module):
    def __init__(self, input_channel, output_channel, activation_list=["LeakyReLU", 0.2]):
        """
        Focus block of yoloV5
        reference:https://blog.csdn.net/weixin_55073640/article/details/122539858
        """
        super(Focus, self).__init__()
        self.activation = build_activation(activation_list)
        self.CBL = nn.Sequential(nn.Conv2d(input_channel * 4, output_channel, kernel_size=3, padding=1, stride=1),
                                 nn.BatchNorm2d(output_channel),
                                 self.activation)

    def forward(self, x):
        sliced_x = torch.cat([x[:, :, 0::2, 0::2],
                              x[:, :, 1::2, 0::2],
                              x[:, :, 1::2, 1::2],
                              x[:, :, 0::2, 1::2]], dim=1)

        return self.CBL(sliced_x)


class SPP(nn.Module):
    def __init__(self, input_channel, activation_list=["LeakyReLU", 0.2]):
        """
        SPP block of yolo V4
        :param input_channel:
        """
        super(SPP, self).__init__()
        self.max_pool1 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=9, padding=4, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=13, padding=6, stride=1)
        self.act_layer = build_activation(activation_list)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel * 4, out_channels=input_channel,
                      kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(input_channel),
            self.act_layer
        )

    def forward(self, x):
        x1 = self.max_pool1(x)
        x2 = self.max_pool2(x)
        x3 = self.max_pool3(x)
        x = torch.cat([x, x1, x2, x3], dim=1)
        output = self.conv1(x)
        return output


class RepConv(nn.Module):
    def __init__(self, input_channel):
        super(RepConv, self).__init__()
        self.conv_part1 = nn.Sequential(nn.Conv2d(input_channel,
                                                  input_channel,
                                                  kernel_size=1,
                                                  padding=0,
                                                  stride=1),
                                        nn.BatchNorm2d(input_channel))
        self.conv_part2 = nn.Sequential(nn.Conv2d(input_channel,
                                                  input_channel,
                                                  kernel_size=1,
                                                  padding=0,
                                                  stride=1),
                                        nn.BatchNorm2d(input_channel))
        self.part3 = nn.BatchNorm2d(input_channel)

    def forward(self, x):
        if self.training:
            x1 = self.conv_part1(x)
            x2 = self.conv_part2(x)
            x3 = self.part3(x)
            x = x1 + x2 + x3
        else:
            x = self.conv_part1(x)
        return x


class PAN(nn.Module):
    def __init__(self, channels=[128, 256, 1024], conv_type="Conv"):
        """
        PAN_Net
        :param channels: the channel of output layer
        :param conv_type: The type of convolution to use after downsampling and concatenation,
                          including 'Conv', 'CSP2_block', "ELAN_MP"
        """
        super(PAN, self).__init__()
        self.down_sample = []
        self.conv_set = []
        for i in range(len(channels) - 1):
            if conv_type == "ELAN_MP":
                downsample_block = [["MP_block", channels[i], 2]]
            else:
                downsample_block = [["Conv", channels[i], channels[i + 1], 3, 1, 2],
                                    ["BatchNorm", channels[i + 1]]]

            if conv_type == "CSP2_block":
                convset_block = [["CSP2_block", channels[i + 1] * 2, channels[i + 1], 1]]

            elif conv_type == "ELAN_MP":
                convset_block = [["ELAN", channels[i] * 2 * 2, "W"]]

            else:
                convset_block = [["Conv", channels[i + 1] * 2, channels[i + 1], 1, 0, 1],
                                 ["BatchNorm", channels[i + 1]]]

            self.down_sample.append(build_block(downsample_block))
            self.conv_set.append(build_block(convset_block))

    def forward(self, x: list):
        """
        :param x: [C4, C5, C6 ..]
        :return: [C4, C5, C6 ..]
        """
        n = len(x)
        assert n == len(self.down_sample) + 1
        output = [x[0]]
        for i in range(n - 1):
            C_i = x[i]
            C_i_downsample = self.down_sample[i](C_i)
            temp = torch.cat([x[i + 1], C_i_downsample], dim=1)
            output.append(self.conv_set[i](temp))
        return output


class FPN_3Layer(FPN):
    def __init__(self, neck_config, channels, hidden_channel=256, activation_list=None):
        super(FPN_3Layer, self).__init__(neck_config, channels, hidden_channel, activation_list)

    def forward(self, f):
        f2, f3, f4 = f
        p4 = self.neck["P4"](f4)
        f3 = self.pre_layer[1](f3)
        p4_up = self.neck["P4_up"](p4)
        p3 = self.neck["P3"](p4_up + f3)
        f2 = self.pre_layer[0](f2)
        p4_up = self.neck["P3_up"](p4_up)
        p3_up = self.neck["P3_up"](p3)
        p2 = self.neck["P2"](p3_up + f2)
        p4 = p4_up
        p3 = p3_up
        return p2, p3, p4


if __name__ == "__main__":
    x = torch.rand((3, 512, 12, 12))
    csp1_block = ELAN(512, mode="W")
    print(csp1_block(x).shape)
