import torch.nn as nn
import torch
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from MultiModeModeling import MultiModalityModeling
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch import UnetPlusPlus
class DTD_Modified(nn.Module):
    def __init__(self):
        super().__init__()
        self.vph = get_encoder("efficientnet-b1", in_channels=3, depth=2, weights="imagenet")
        self.fph = get_encoder("timm-mobilenetv3_small_075", in_channels=16, depth=2, weights="imagenet")
        self.mmm = MultiModalityModeling(img_size=(128, 128), in_chans=self.vph.out_channels[-1]+self.fph.out_channels[-1],
                                         path_size=1, embed_dim=96, window_size=8, depths=(2, 2, 6), out_chans=192)
        self.fuse = UnetPlusPlusDecoder(n_blocks=4, encoder_channels=[3, 32, 192, 384, 768], decoder_channels=[256, 128, 64, 32])
        self.segmentation_head = SegmentationHead(in_channels=32, out_channels=1)
        self.f_emedding = nn.Embedding(21, 16)
    def forward(self, img, dct):
        feats = self.vph(img)
        f0, fv = feats[1], feats[-1]
        dct = self.f_emedding(dct)
        dct = dct.permute(0, 3, 1, 2)
        feats_fre = self.fph(dct)
        fd = feats_fre[-1]
        f1, f2, f3 = self.mmm(torch.cat([fv, fd], dim=1))
        decoder_output = self.fuse(*[img, f0, f1, f2, f3])
        pred_mask = self.segmentation_head(decoder_output)
        return pred_mask

if __name__=="__main__":
    model = DTD_Modified()
    img = torch.rand(size=(2, 3, 512, 512))
    dct = torch.randint(low=0, high=20, size=(2, 512, 512))
    print(model(img, dct.long()))
    # vph = get_encoder("efficientnet-b1", in_channels=3, depth=5, weights="imagenet")
    # for x in vph(img):
    #     print(x.shape)