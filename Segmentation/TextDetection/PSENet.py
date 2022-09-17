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

    def build_backbone(self, backbone_type):
        model = nn.Sequential()
        if "resnet" in backbone_type:
            model = resnet(model_type=backbone_type, activation_list=self.cfg.backbone_activation_list)
        return model

    def build_neck(self, neck_config):
        neck = []
        for neck_name, neck_block_config in neck_config.items():
            if neck_name.startswith("FPN"):
                neck_block_list = [["FPN", neck_block_config, self.cfg.channels]]
            neck.append(build_block(neck_block_list, activation_list=self.cfg.neck_activation_list))

        return neck

    def build_head(self, head_config):
        head = {}
        for head_name, head_block_list in head_config.items():
            head[head_name] = build_block(head_config[head_name], activation_list=self.cfg.head_activation_list)
        return head

    def forward(self, imgs):
        _, _, H, W = imgs.shape
        p2, p3, p4, p5 = self.backbone(imgs)
        for neck_block in self.neck:
            p2, p3, p4, p5 = neck_block([p2, p3, p4, p5])
        output = torch.cat([p2, p3, p4, p5], dim=1)
        output = self.head["head"](output)
        output = F.interpolate(output, size=(H, W))
        return output


if __name__ == "__main__":
    psenet = pseNet()
    imgs = torch.rand((1, 3, 640, 640))
    output = psenet(imgs)
    print(output.shape)