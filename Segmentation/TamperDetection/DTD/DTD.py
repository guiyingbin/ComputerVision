import torch.nn as nn
from VisualPerceptionHead import VisualPerceptionHead as VPH
from FrequencyPerceptionHead import FrequencyPerceptionHead as FPH
from MultiModeModeling import MultiModalityModeling as MMM
from MultiIterativeDecoder import MultiViewIterativeDecoder as MID
from segmentation_models_pytorch.base import modules as md
import torch


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = md.Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class DTD(nn.Module):
    def __init__(self, img_size=512, vph_dim=[96, 192],
                 fph_emb_dim=64, fph_n_embed=256,
                 mmm_win_size=8,
                 mmm_swin_depths=[2, 2, 6],
                 mid_enc_channels = [96, 192, 384, 768],
                 mid_dec_channels = [384, 192, 96, 96],
                 classes=1):
        super().__init__()
        self.vph = VPH(dim=vph_dim)
        self.fph = FPH(out_dim=vph_dim[0]*2,
                       n_embed=fph_n_embed,
                       embed_dim=fph_emb_dim)
        self.mmm = MMM(img_size=img_size // 8,
                       in_chans=vph_dim[1] + vph_dim[0]*2,
                       out_chans=vph_dim[0]*2,
                       window_size=mmm_win_size,
                       depths=mmm_swin_depths)
        self.mid = MID(encoder_channels=mid_enc_channels,
                       decoder_channels=mid_dec_channels)
        self.seg_head = SegmentationHead(in_channels=vph_dim[0],
                                         out_channels=classes,
                                         upsampling=2)

    def forward(self, x, dct):
        feats = self.vph(x)
        f0, fv = feats[0], feats[1]
        fd = self.fph(dct)
        f1, f2, f3 = self.mmm(torch.cat([fv, fd], dim=1))
        decode_feats = self.mid(*[f0, f1, f2, f3])
        output = self.seg_head(decode_feats)
        return output


if __name__ == "__main__":
    a = torch.rand(size=(1, 3, 512, 512))
    dct = torch.randint(low=1, high=256, size=(1, 512, 512))
    model = DTD()
    print(model(a, dct).shape)
