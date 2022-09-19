from collections import OrderedDict
import torch
import torch.nn as nn
from Segmentation.Utils.Layers import FPN
import numpy as np


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

        if block_type == "FPN":
            block.add_module("{}{}".format(block_type, i), FPN(neck_config=block_info[1],
                                                               channels=block_info[2],
                                                               hidden_channel=block_info[3],
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
        self.MHA_cross = MultiHeadAttention(n_head = n_head, d_k=n_position, d_v=n_position, d_model=d_model)
        self.FF = FeedForward(input_channel=d_model, hidden_channel=d_model*4)
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


if __name__ == "__main__":
    dk = 128
    dv = 256
    dmodel = 512
    TE = TransformerEncoder()
    PVA = ParallelVisualAttention()
    GRS = GlobalSemanticReasoning()

    PCA= ParallelContextualAttention()
    x = torch.rand((7, 256, dmodel))
    dpandecoder = DPANDecoder()
    # output = TE(x)
    # output = PVA(x)
    # pred1, s_t = PCA(output)
    qc, s_t, f = dpandecoder(x)
    print(s_t.shape)
