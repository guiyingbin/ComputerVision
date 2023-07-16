import torch.nn as nn
import torch
from timm.models.swin_transformer_v2 import SwinTransformerV2
import math

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class MultiModalityModeling(nn.Module):
    def __init__(self, img_size, in_chans, out_chans, window_size, path_size=1, depths=(2, 2, 6), embed_dim=192):
        """
        原文给的并未说明采用了采用的结构，直接载入了某个权重模型，
        在这里进行了尝试，如果要满足原文的输入和输出大小且为swinv2的话
        patch_size 大小只能是1
        """
        super().__init__()
        assert len(depths)>=3
        self.swin = SwinTransformerV2(img_size=img_size,
                                    in_chans=out_chans,
                                    window_size=window_size,
                                    patch_size=path_size,
                                    depths=depths,
                                    embed_dim=192)
        self.scse = SCSEModule(in_channels=in_chans)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=in_chans,
                                               out_channels=out_chans,
                                               kernel_size=1),
                                     nn.BatchNorm2d(out_chans),
                                     nn.ReLU())
    def forward(self, x):
        x = self.scse(x)
        x_1 = self.conv1_1(x)
        x_1_ = self.swin.patch_embed(x_1)
        x_1_ = self.swin.layers[0](x_1_)
        x_2 = self.swin.layers[1](x_1_)
        x_3 = self.swin.layers[2](x_2)
        x_2 = x_2.permute(0, 3, 1, 2).contiguous()
        x_3 = x_3.permute(0, 3, 1, 2).contiguous()
        return x_1, x_2, x_3

if __name__ == "__main__":
    mm = MultiModalityModeling(img_size=64, in_chans=1024, out_chans=256, window_size=4, path_size=1)
    a = torch.rand(size=(1, 1024, 64, 64))
    a0, a1, a2 = mm(a)
    print(a0.shape)
    print(a1.shape)
    print(a2.shape)

