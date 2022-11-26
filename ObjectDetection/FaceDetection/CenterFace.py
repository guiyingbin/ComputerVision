import torch.nn as nn
from Classification.MobileNet.MobileNetV3 import mobileNetV3
from ObjectDetection.Config.FaceDetectionConfig import centerface_cfg
from ObjectDetection.Utils.Layers import build_block
import torch
import torch.nn.functional as F


class centerFace(nn.Module):
    def __init__(self, cfg=centerface_cfg):
        super(centerFace, self).__init__()
        self.cfg = cfg
        self.backbone = self.build_backbone(model_name=self.cfg.backbone_type)
        self.neck = self.build_neck(self.cfg.neck_config)
        self.head = self.build_head(self.cfg.head_config)

    def build_backbone(self, model_name, pretrained=False):
        return mobileNetV3(model_name=model_name, pretrained=pretrained, out_features=True,
                           output_layer_index=[2, 3, 4])

    def build_neck(self, neck_config):
        neck = []
        for neck_name, neck_block_config in neck_config.items():
            if neck_name.startswith("FPN"):
                neck_block_list = [[neck_name, neck_block_config, self.cfg.channels]]
            else:
                neck_block_list = neck_block_config
            neck.append(build_block(neck_block_list, activation_list=self.cfg.neck_activation_list))

        return neck

    def build_head(self, head_config):
        head = {}
        for head_name, head_block_list in head_config.items():
            head[head_name] = build_block(head_config[head_name], activation_list=self.cfg.head_activation_list)
        return head

    def forward(self, imgs):
        _, _, H, W = imgs.shape
        p2, p3, p4 = self.backbone(imgs)
        for neck_block in self.neck:
            p2, p3, p4 = neck_block([p2, p3, p4])
        output = torch.cat([p2, p3, p4], dim=1)
        pred_class = self.head["class_head"](output)
        pred_box = self.head["bbox_head"](output)
        pred_keypoint = self.head["keypoint_head"](output)
        return pred_class, pred_box, pred_keypoint

if __name__ == "__main__":
    model = centerFace()
    imgs = torch.rand((1, 3, 640, 640))
    pred_class, pred_box, pred_keypoint = model(imgs)
    print(pred_class.shape)