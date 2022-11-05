from collections import OrderedDict
import math
import torch
from typing import Optional
import torch.nn as nn
from Segmentation.Utils.Layers import FPN
import numpy as np
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from torch.nn.modules import transformer
from torch import nn as nn, Tensor

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

        if block_type == "UpNearest":
            block.add_module("{}{}".format(block_type, i), nn.UpsamplingNearest2d(scale_factor=block_info[1]))

        if block_type == "FPN":
            block.add_module("{}{}".format(block_type, i), FPN(neck_config=block_info[1],
                                                               channels=block_info[2],
                                                               hidden_channel=block_info[3],
                                                               activation_list=activation_list))

        if block_type == "RepBlock":
            block.add_module("{}{}".format(block_type, i), RepBlock(in_channels=block_info[1],
                                                                    output_channels=block_info[2],
                                                                    stride=block_info[3],
                                                                    identity=block_info[4],
                                                                    activation_list=activation_list))
        if block_type == "DenseBlock":
            block.add_module("{}{}".format(block_type, i), DenseBlock(in_channels=block_info[1],
                                                                      output_channels=block_info[2],
                                                                      n_block=block_info[3],
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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, output_channels, activation_list=["ReLU"]):
        super(ConvBlock, self).__init__()
        self.act = build_activation(activation_list)
        self.model = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   self.act,
                                   nn.Conv2d(in_channels, output_channels, 1, 1, 0),
                                   nn.BatchNorm2d(output_channels),
                                   self.act,
                                   nn.Conv2d(output_channels, output_channels, 3, 1, 1)
                                   )

    def forward(self, x):
        return self.model(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, output_channels, n_block, activation_list=["ReLU"]):
        """
        The basic Block of DenseNet
        :param in_channels:
        :param output_channels:
        """
        super(DenseBlock, self).__init__()
        self.model = []
        self.model.append(ConvBlock(in_channels, output_channels, activation_list))
        for i in range(n_block - 1):
            self.model.append(ConvBlock(in_channels + output_channels * (i + 1),
                                        output_channels,
                                        activation_list))

    def forward(self, x):
        output = [x]
        for layer in self.model:
            temp = torch.cat(output, dim=1)
            latent = layer(temp)
            output.append(latent)
        return torch.cat(output, dim=1)


class RepBlock(nn.Module):
    def __init__(self, in_channels, output_channels, stride, identity=True, activation_list=["ReLU"]):
        """
        The basic block of RepVGG
        :param in_channels: input channels
        :param output_channels: output channels
        :param stride:
        :param identity: whether to use the residual operation
        """
        super(RepBlock, self).__init__()
        self.con_3x3 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=output_channels,
                                 kernel_size=3,
                                 padding=1,
                                 stride=stride)
        self.con_1x1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=output_channels,
                                 kernel_size=1,
                                 padding=0,
                                 stride=stride)

        self.identity = identity
        self.act = build_activation(activation_list)
        self.bn = nn.Sequential(nn.BatchNorm2d(output_channels),
                                self.act)

    def forward(self, x):
        o1 = self.bn(self.con_3x3(x))
        if self.training:
            o2 = self.bn(self.con_1x1(x))
            if not self.identity:
                return o2 + o1
            x = self.bn(x)
            return o1 + o2 + x
        else:
            return o1


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
        return output


class PositionEncoding(nn.Module):
    def __init__(self, d_model, n_position=256):
        """
        The position encoding module of transformer block
        :param d_model: the output dimension of input embedding(B, T, D), usually it is equal to the D
        :param n_position: the max length of the Step T of input embedding
        """
        super(PositionEncoding, self).__init__()
        self.d_model = d_model
        self.n_position = n_position
        self.get_sinusoid_encoding_table()
        self.sinusoid_table = self.get_sinusoid_encoding_table()

    def get_position_angel_vec(self, postion):
        return postion / torch.pow(10000, (torch.arange(0, self.d_model) // 2) / (self.d_model / 2))

    def get_sinusoid_encoding_table(self):
        """
        :return: sinusoid_table, and shape is (n_position, d_model)
        """
        sinusoid_table = self.get_position_angel_vec(torch.arange(0, self.n_position).reshape(-1, 1))
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        return x + self.sinusoid_table[:, :x.shape[1]]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        latent = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        if mask is not None:
            assert latent.shape == mask.shape
            latent = latent.masked_fill(mask, 1e-9)

        output = latent.softmax(dim=-1) @ v
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_model):
        """
        multi head attention module of transformer
        the detail can be found in Paper: Attention is all your need
        :param n_head:
        :param d_k:
        :param d_v:
        :param d_model:
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.scaleDotAttention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.q_linear = nn.Linear(d_model, n_head * d_k)
        self.k_linear = nn.Linear(d_model, n_head * d_k)
        self.v_linear = nn.Linear(d_model, n_head * d_v)
        self.layer_norm = nn.LayerNorm(d_model)

        self.linear = nn.Linear(n_head * d_v, d_model)

    def forward(self, q, k, v, mask=None):
        """

        :param q: shape is [B, q_len, d_model]
        :param k: shape is [B, k_len, d_model]
        :param v: shape is [B, v_len, d_model]
        :param mask: shape is [B, q_len, k_len]
        :return:
        """
        B, q_len, _ = q.shape
        B, k_len, _ = k.shape
        B, v_len, _ = v.shape

        q_ = self.q_linear(q).reshape(B, q_len, self.n_head, self.d_k)
        k_ = self.k_linear(k).reshape(B, k_len, self.n_head, self.d_k)
        v_ = self.v_linear(v).reshape(B, v_len, self.n_head, self.d_v)

        q_ = q_.permute(2, 0, 1, 3).contiguous().reshape(-1, q_len, self.d_k)
        k_ = k_.permute(2, 0, 1, 3).contiguous().reshape(-1, k_len, self.d_k)
        v_ = v_.permute(2, 0, 1, 3).contiguous().reshape(-1, v_len, self.d_v)

        latent = self.scaleDotAttention(q_, k_, v_, mask)
        latent = latent.reshape(self.n_head, B, v_len, self.d_v).permute(1, 2, 0, 3).contiguous()
        latent = latent.reshape(B, v_len, -1)
        output = self.linear(latent) + q
        output = self.layer_norm(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        """
        Positionwise Feed Forward Module of
        :param input_channel:
        :param hidden_channel:
        """
        super(FeedForward, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channel,
                               out_channels=hidden_channel,
                               kernel_size=1,
                               padding=0,
                               stride=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_channel,
                               out_channels=input_channel,
                               kernel_size=1,
                               padding=0,
                               stride=1)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(input_channel)

    def forward(self, x):
        x_ = x.transpose(1, 2)
        latent = self.conv1(x_)
        latent = self.activation(latent)
        latent = self.conv2(latent)
        latent = latent.transpose(1, 2)
        output = self.norm(latent + x)
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_model):
        super(TransformerEncoderBlock, self).__init__()
        self.MHA = MultiHeadAttention(n_head, d_k, d_v, d_model)
        self.FF = FeedForward(d_model, d_model * 4)

    def forward(self, encoded_input, mask=None):
        output = self.MHA(encoded_input, encoded_input, encoded_input, mask)
        output = self.FF(output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, n_block=2, n_head=8, d_k=64, d_v=64, d_model=512, n_position=256):
        super(TransformerEncoder, self).__init__()
        self.positionalEncoding = PositionEncoding(d_model=d_model,
                                                   n_position=n_position)
        self.EncoderLayers = nn.ModuleList([TransformerEncoderBlock(n_head=n_head,
                                                                    d_k=d_k,
                                                                    d_v=d_v,
                                                                    d_model=d_model)] * n_block)

    def forward(self, x, mask=None):
        x = self.positionalEncoding(x)
        for encoderLayer in self.EncoderLayers:
            x = encoderLayer(x, mask)
        return x


class ParallelVisualAttention(nn.Module):
    def __init__(self, d_model=512, n_max_len=25, n_position=256):
        """
        The Parallel Visual Attention Module of SRN
        :param d_model: the output dimension of Transformer Encoder
        :param n_max_len: the max length of character in single image
        :param n_position:
        """
        super(ParallelVisualAttention, self).__init__()
        self.n_max_len = n_max_len
        self.d_model = d_model
        self.n_position = n_position
        self.f_o = nn.Embedding(n_max_len, d_model)

        self.w_o = nn.Linear(n_max_len, n_position)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_e = nn.Linear(d_model, n_max_len)

        self.activation = nn.Tanh()

    def forward(self, v):
        """
        :param v: (B, n_position, d_model)
        :return: (B, n_max_len, d_model)
        """
        O_t = torch.arange(0, self.n_max_len, dtype=torch.long, device=v.device).unsqueeze(0).repeat(v.shape[0], 1)
        O_t = self.f_o(O_t)  # B, n_max_len, d_model
        O_t = O_t.transpose(1, 2).contiguous()
        O_t = self.w_o(O_t)  # B, d_model, n_position
        O_t = O_t.transpose(1, 2) + self.w_v(v)  # B, n_position, d_model
        O_t = self.w_e(self.activation(O_t))
        e_t = O_t.transpose(1, 2).contiguous()
        a_t = e_t.softmax(-2)
        g_t = a_t @ v
        return g_t


class GlobalSemanticReasoning(nn.Module):
    def __init__(self, n_class=37, n_block=4, n_position=25, d_model=512):
        super(GlobalSemanticReasoning, self).__init__()
        self.transformer_unit = TransformerEncoder(n_block=n_block, n_position=n_position)
        self.fc = nn.Sequential(nn.Linear(d_model, n_class),
                                nn.Softmax(dim=2))
        self.embedding = nn.Embedding(n_class, d_model)

    def forward(self, G):
        """
        :param g_t: the output of PVA module, and its shape is (B, n_max_len, d_model)
        :return:
        """
        g_t = self.fc(G)
        e_t = self.embedding(g_t.argmax(-1))
        s_t = self.transformer_unit(e_t)
        return g_t, s_t


class BidirectionalLanguageModelBlock(nn.Module):
    def __init__(self, n_class=37, n_position=25, d_model=512, n_head=8):
        """
        Bidirectional Language Model in DPAN
        :param n_class:
        :param n_block:
        :param n_position:
        :param d_model:
        """
        super(BidirectionalLanguageModelBlock, self).__init__()
        self.MHA_cross = MultiHeadAttention(n_head=n_head, d_k=n_position, d_v=n_position, d_model=d_model)
        self.FF = FeedForward(input_channel=d_model, hidden_channel=d_model * 4)
        self.pe = PositionEncoding(d_model=d_model, n_position=n_position)
        self.fc = nn.Sequential(nn.Linear(d_model, n_class),
                                nn.Softmax(dim=2))
        self.embbedding = nn.Embedding(n_class, d_model)

    def forward(self, G, mask=None):
        k = self.pe(G)
        q = self.embbedding(self.fc(G).argmax(-1))
        output = self.MHA_cross(q, k, k, mask)
        output = self.FF(output)
        return output


class BidirectionalLanguageModel(nn.Module):
    def __init__(self, n_class=37, n_position=25, d_model=512, n_head=8, n_block=4):
        """
        Bidirectional Language Model in DPAN
        :param n_class:
        :param n_block:
        :param n_position:
        :param d_model:
        """
        super(BidirectionalLanguageModel, self).__init__()
        self.model = nn.ModuleList([BidirectionalLanguageModelBlock(n_head=n_head,
                                                                    n_class=n_class,
                                                                    n_position=n_position,
                                                                    d_model=d_model)] * n_block)

    def forward(self, x, mask=None):
        for encoderLayer in self.model:
            x = encoderLayer(x, mask)
        return x


class ParallelContextualAttention(nn.Module):
    def __init__(self, n_class=37, n_block=4, n_position=25, d_model=512):
        """
        unofficial implemention of Parallel Contextual Attention Module in DPAN
        :param n_class:
        :param n_block:
        :param n_position:
        :param d_model:
        """
        super(ParallelContextualAttention, self).__init__()
        self.projecton = nn.Linear(d_model, d_model)
        self.blm = BidirectionalLanguageModel(n_class=n_class, n_block=n_block,
                                              n_position=n_position, d_model=d_model)

        self.fc = nn.Sequential(nn.Linear(d_model, n_class),
                                nn.Softmax(dim=2))
        self.embbedding = nn.Embedding(n_class, d_model)

    def forward(self, G, mask=None):
        g_t = self.projecton(G)
        s_t = self.blm(g_t, mask)
        qc = self.embbedding(self.fc(s_t).argmax(-1))
        return s_t, qc


class DPANDecoder(nn.Module):
    def __init__(self, d_model=512, n_max_len=25, n_position=256, n_class=37, n_block=4):
        """
        The Parallel Positional Attention Module and Parallel Contextual Attention Module of DPAN
        :param d_model:
        :param n_max_len:
        :param n_position:
        :param n_class:
        :param n_block:
        """
        super(DPANDecoder, self).__init__()
        self.PVA = ParallelVisualAttention(d_model, n_max_len, n_position)
        self.PCA = ParallelContextualAttention(n_class=n_class, n_block=n_block, n_position=n_max_len, d_model=d_model)
        self.linear = nn.Sequential(nn.Linear(d_model, n_class),
                                    nn.Softmax(dim=2))
        self.SDP = ScaledDotProductAttention(temperature=np.power(d_model, 0.5))
        self.fc1 = nn.Sequential(nn.Linear(d_model, n_class),
                                 nn.Softmax(dim=2))
        self.fc2 = nn.Sequential(nn.Linear(d_model, n_class),
                                 nn.Softmax(dim=2))

    def forward(self, x, mask=None):
        G1 = self.PVA(x)
        pred1 = self.fc1(G1)
        Qc, s_l = self.PCA(G1)
        pred2 = self.fc2(Qc)
        G2 = self.SDP(Qc, x, x, mask)
        pred3 = self.linear(G2 + s_l)
        return pred1, pred2, pred3


class SRNDecoder(nn.Module):
    def __init__(self, d_model=512, n_max_len=25, n_position=256, n_class=37, n_block=4):
        super(SRNDecoder, self).__init__()
        self.PVA = ParallelVisualAttention(d_model, n_max_len, n_position)
        self.GRS = GlobalSemanticReasoning(n_class, n_block, n_max_len, d_model)
        self.linear = nn.Linear(d_model, n_class)

    def forward(self, x):
        G = self.PVA(x)
        g_t, s_t = self.GRS(G)
        F = self.linear(G + s_t)
        return g_t, s_t, F

"""
Source: https://github.com/chenjun2hao/SRN.pytorch
"""
class TPS_SpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

        return batch_I_r


class LocalizationNetwork(nn.Module):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1,
                      bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):
        """
        input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
        return batch_C_prime


class GridGenerator(nn.Module):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # F+3 x F+3
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # n x F+3

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(
            batch_size, 3, 2).float()), dim=1)  # batch_size x F+3 x 2
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, batch_max_length=25):
        super(Attention, self).__init__()
        self.batch_max_length = batch_max_length
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_()
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text=None):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = self.batch_max_length + 2  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0))

        if self.training:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha

"""
source:https://github.com/baudm/parseq
"""
class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)

"""
source:https://github.com/baudm/parseq
"""
class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, tgt_mask: Optional[Tensor],
                       tgt_key_padding_mask: Optional[Tensor]):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)[0]
        return query, content

"""
source:https://github.com/baudm/parseq
"""
class PARSeq_Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last)
        query = self.norm(query)
        return query


class PARSeq_Encoder(SwinTransformer):
    """
    The Encoder of PARSeq.
    In the original paper of PARSeq, the encoder is ViT. Here, we replace it with SWin Transformer
    """
    def __init__(self, img_size,
                 patch_size=(4, 8),
                 in_chans=3,
                 window_size=4,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.):

        super(PARSeq_Encoder, self).__init__(img_size=img_size,
                                             patch_size=patch_size,
                                             in_chans=in_chans,
                                             window_size=window_size,
                                             embed_dim=embed_dim,
                                             depths=depths,
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             drop_rate=drop_rate,
                                             attn_drop_rate=attn_drop_rate,
                                             drop_path_rate=drop_path_rate,
                                             num_classes=0)

    def forward(self, x):
        return self.forward_features(x)

class Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=False)  # these disable the classifier head

    def forward(self, x):
        # Return all tokens
        return self.forward_features(x)
if __name__ == "__main__":
    # dense_block = DenseBlock(32, 32, 6)
    # x = torch.randn((3, 32, 224, 224))
    # print(dense_block(x).shape)
    from torchsummary import summary
    encoder = PARSeq_Encoder(img_size=(32, 256),
                             patch_size=(2, 4),
                             embed_dim=96,
                             depths=(2, 2),
                             num_heads=(3, 6))
    x = torch.randn((1, 3, 32, 256))
    # decoder = PARSeq
    print(encoder(x).shape)
    print(summary(encoder, input_size=(3, 32, 256), batch_size=1, device="cpu"))
    vit_encoder = Encoder(img_size=(32, 256), patch_size=[4, 8], embed_dim=192)
    print(vit_encoder(x).shape)
    print(summary(vit_encoder, input_size=(3, 32, 256), batch_size=1, device="cpu"))