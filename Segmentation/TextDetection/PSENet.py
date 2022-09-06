import torch.nn as nn
import torch
from Segmentation.Config.TextDetectionConfig import psenet_cfg
from Segmentation.Utils.Layers import build_block
from Classification.ResNet import resnet
import torch.nn.functional as F


class pseNet(nn.Module):
    def __init__(self, cfg=psenet_cfg):
        super(pseNet, self).__init__()
        self.cfg = cfg
        self.backbone = self.build_backbone(cfg.backbone_type)
        self.neck = self.build_neck(cfg.neck_config)
        self.head = self.build_head(cfg.head_config)

        self.pre_layer = [nn.Conv2d(self.cfg.channels[i], 256, kernel_size=3, padding=1, stride=1) for i in range(0, 3)]

    def build_backbone(self, backbone_type):
        model = nn.Sequential()
        if "resnet" in backbone_type:
            model = resnet(model_type=backbone_type, activation_list=self.cfg.backbone_activation_list)
        return model

    def build_neck(self, neck_config):
        neck = {}
        for block_name, block_list in neck_config.items():
            neck[block_name] = build_block(block_list, activation_list=self.cfg.neck_activation_list)

        return neck

    def build_head(self, head_config):
        return build_block(head_config["head"], activation_list=self.cfg.head_activation_list)

    def forward(self, imgs):
        _, _, H, W = imgs.shape
        f2, f3, f4, f5 = self.backbone(imgs)
        p5 = self.neck["P5"](f5)
        f4 = self.pre_layer[2](f4)
        p5_up = self.neck["P5_up"](p5)
        p4 = self.neck["P4"](p5_up+f4)
        f3 = self.pre_layer[1](f3)
        p5_up = self.neck["P4_up"](p5_up)
        p4_up = self.neck["P4_up"](p4)
        p3 = self.neck["P3"](p4_up+f3)
        f2 = self.pre_layer[0](f2)
        p5_up = self.neck["P3_up"](p5_up)
        p4_up = self.neck["P3_up"](p4_up)
        p3_up = self.neck["P3_up"](p3)
        p2 = self.neck["P2"](p3_up+f2)

        p5 = p5_up
        p4 = p4_up
        p3 = p3_up
        output = torch.cat([p2, p3, p4, p5], dim=1)
        output = self.head(output)
        output = F.interpolate(output, size=(H, W))
        return output


if __name__ == "__main__":
    psenet = pseNet()
    imgs = torch.rand((1, 3, 640, 640))
    output = psenet(imgs)
    print(output.shape)