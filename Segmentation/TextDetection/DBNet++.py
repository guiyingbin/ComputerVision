from Segmentation.TextDetection.DBNet import dbNet
from Segmentation.Config.TextDetectionConfig import dbnet_plusplus_cfg
import torch


class dbNet_plusplus(dbNet):
    def __init__(self, cfg=dbnet_plusplus_cfg):
        super(dbNet_plusplus, self).__init__(cfg)

    def forward(self, imgs):
        _, _, H, W = imgs.shape
        p2, p3, p4, p5 = self.backbone(imgs)
        for neck_block in self.neck:
            p2, p3, p4, p5 = neck_block([p2, p3, p4, p5])
        output = torch.cat([p2, p3, p4, p5], dim=1)
        output = self.head["pre_conv"](output)
        binary = self.head["prob_head"](output)
        threshold = self.head["threshold_head"](output)
        output = self.differentiable_binarization(binary, threshold)
        return binary, threshold, output


if __name__ == "__main__":
    dbnet = dbNet_plusplus()
    imgs = torch.rand((1, 3, 640, 640))
    binary, threshold, output = dbnet(imgs)
    print(binary.shape)
    print(threshold.shape)
    print(output.shape)