import torch.nn as nn
from CommonLayer import ConvNextBlock, AddCoords
import torch

class FrequencyPerceptionHead(nn.Module):
    def __init__(self, out_dim=256, n_embed=256, embed_dim=64):
        """
        在原文这里增加quant table的输入，但是该方式只是针对JPEG图片的压缩方式，
        PNG图片并不采用该方式，所以在此处剔除
        同时，原文采用DCT这种频率信息，DCT这种信息无论是PNG还是JPEG修改后，都会不一样，所以保留下来也是因为参考这篇论文
        Passive detection of doctored jpeg image via block artifact grid extraction
        """
        super().__init__()
        self.f_emedding = nn.Embedding(n_embed, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim+3, out_channels=out_dim, kernel_size=8, stride=8),
            ConvNextBlock(dim=out_dim),
            ConvNextBlock(dim=out_dim),
            ConvNextBlock(dim=out_dim)
        )
        self.add_coord = AddCoords() # 此处的position encoding是直接将坐标index与图片沿通道数拼接

    def forward(self, x):
        x = self.f_emedding(x)
        x = x.permute(0, 3, 1, 2)
        x_pos = self.add_coord(x)
        return self.conv(x_pos)

if __name__ == "__main__":
    a = torch.randint(low=1, high=255, size=(1, 512, 512))
    model = FrequencyPerceptionHead()
    print(model(a).shape)

