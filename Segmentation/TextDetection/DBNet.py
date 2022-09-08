"""
DBNet
@author: guiyingbin
@time: 2022/09/08
"""

from Segmentation.TextDetection.PSENet import pseNet
from Segmentation.Config.TextDetectionConfig import dbnet_cfg
import torch
import torch.nn.functional as F


class dbNet(pseNet):
    def __init__(self, cfg=dbnet_cfg):
        super(dbNet, self).__init__(cfg)
        self.k = cfg.k

    def differentiable_binarization(self,binary, threshold):
        return 1/(1+torch.exp(-self.k*(binary-threshold)))

    def forward(self, imgs):
        _, _, H, W = imgs.shape
        p2, p3, p4, p5 = self.backbone(imgs)
        for neck_block in self.neck:
            p2, p3, p4, p5 = neck_block([p2, p3, p4, p5])
        output = torch.cat([p2, p3, p4, p5], dim=1)
        binary = self.head["prob_head"](output)
        threshold = self.head["threshold_head"](output)
        output = self.differentiable_binarization(binary, threshold)
        return output

if __name__ == "__main__":
    dbnet = dbNet()
    imgs = torch.rand((1, 3, 640, 640))
    output = dbnet(imgs)
    print(output.shape)