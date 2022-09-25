import torch.nn as nn
from ObjectDetection.Config.TextDetectionConfig import east_cfg
from Classification.VGG import vgg
from Classification.Utils.Layers import build_block


class east(nn.Module):
    def __init__(self, cfg=east_cfg):
        """
        The EAST(Efficient and Accuracy Scene Text detection) model
        paper url: https://arxiv.org/pdf/1801.02765.pdf
        the model is mainly to detect multi-oriented text
        :param cfg:
        """
        super(east, self).__init__()
        self.cfg = cfg
        self.backbone = self.build_backbone(self.cfg.backbone_type, self.cfg.backbone_activation_list)
        self.neck = self.build_neck(self.cfg.neck_config, self.cfg.neck_activation_list)
        self.head = self.build_head(self.cfg.head_config, self.cfg.head_activation_list)

    def build_backbone(self, model_type, activation_list):
        if model_type.startswith("vgg"):
            model = vgg(model_type, activation_list)
        else:
            model = nn.Identity()
        return model

    def build_neck(self, model_config, activation_list):

        C5 = build_block(model_config["C5"], activation_list=activation_list)
        C5_up = build_block(model_config["C5_up"], activation_list=activation_list)

        C4 = build_block(model_config["C4"], activation_list=activation_list)
        C4_up = build_block(model_config["C4_up"], activation_list=activation_list)

        C3 = build_block(model_config["C3"], activation_list=activation_list)
        C3_up = build_block(model_config["C3_up"], activation_list=activation_list)

        C2 = build_block(model_config["C2"], activation_list=activation_list)
        neck = {"C2":C2,
                "C3_up":C3_up,
                "C3":C3,
                "C4_up":C4_up,
                "C4":C4,
                "C5_up":C5_up,
                "C5":C5}
        return neck

    def build_head(self, model_config, activation_list):
        score_head = build_block(model_config["score_head"], activation_list=activation_list)

        boxes_head = build_block(model_config["boxes_head"], activation_list=activation_list)

        angle_head = build_block(model_config["angle_head"], activation_list=activation_list)

        quad_head = build_block(model_config["quad_head"], activation_list=activation_list)

        head = {"score_head": score_head,
                "boxes_head": boxes_head,
                "angle_head": angle_head,
                "quad_head": quad_head}
        return head

    def forward(self, imgs):
        f2, f3, f4, f5 = self.backbone(imgs) #128, 256, 512, 512

        p5 = self.neck["C5"](f5)
        p5_up = self.neck["C5_up"](f5)
        p4 = self.neck["C4"](torch.cat([p5_up, f4], dim=1))
        p4_up = self.neck["C4_up"](p4)
        p3 = self.neck["C3"](torch.cat([p4_up, f3], dim=1))
        p3_up = self.neck["C3_up"](p3)
        p2 = self.neck["C2"](torch.cat([p3_up, f2], dim=1))

        score = self.head["score_head"](p2)
        boxes = self.head["boxes_head"](p2)
        angle = self.head["angle_head"](p2)
        quad = self.head["quad_head"](p2)
        return score, boxes, angle, quad


if __name__ == "__main__":
    import torch
    east_net = east()
    imgs = torch.rand((1, 3, 640, 640))
    score, boxes, angle, quad = east_net(imgs)
    print(score.shape)
    print(boxes.shape)
    print(angle.shape)
    print(quad.shape)



