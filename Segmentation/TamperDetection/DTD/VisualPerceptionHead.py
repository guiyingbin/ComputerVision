import torch.nn as nn
import torch
from CommonLayer import LayerNorm, ConvNextBlock, DownSampleBlock


class VisualPerceptionHead(nn.Module):
    def __init__(self, dim=None):
        """
        该模块输入为3*H*W大小图片，输出为Fv (Cv*H/8*W/8) Ff0 (C0*H/4*W/4)
        采用7层卷积blocks作为图像特征输出
        这里采用ConvNext block, 原文中采用的也是ConvNext
        """
        super().__init__()
        if dim is None:
            dim = [96, 192]
        self.n = len(dim)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=dim[0], kernel_size=4, stride=4),
            LayerNorm(dim[0], eps=1e-6),
            ConvNextBlock(dim=dim[0])
        )
        self.downsampler = []
        for i in range(self.n-1):
            self.downsampler.append(DownSampleBlock(dim[i], dim[i+1]))

        self.convblocks = []
        for i in range(self.n-1):
            self.convblocks.append(ConvNextBlock(dim[i+1]))

    def forward(self, x):
        x = self.layer1(x)
        output = [x]
        for i in range(self.n-1):
            x = self.downsampler[i](x)
            x = self.convblocks[i](x)
            output.append(x)
        return output



if __name__ == "__main__":
    a = torch.rand(size=(1, 3, 512, 512))
    model = VisualPerceptionHead()
    print(model(a)[-1].shape)