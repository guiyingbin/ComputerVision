import torch
import torch.nn as nn
from Utils.Layers import build_activation, Darknet_block, build_block
from ObjectDetection.Yolo.YoloV2 import yoloV2
from ObjectDetection.Config.YoloConfig import yolov3_cfg


class yoloV3(yoloV2):
    def __init__(self, cfg):
        """
        implementation of yolo v3
        Compared to yolo v2, the improvements of yolo v3 are shown as following:
            1. Make a deeper backbone:Darknet53
            2. Grids with different sizes are used for objects of different sizes
        :param n_anchors: number of anchors
        :param n_class: number of objects
        :param img_size: the size of image, and the default value is 448
        :param activation_list: the setting of activation layer
        """
        super(yoloV3, self).__init__(cfg)

        self.neck = self.build_neck(cfg.neck_config)

    def build_neck(self, model_config):
        big_object_block1 = build_block(model_config["C4"], activation_list=self.activation_list)
        big_object_upsample = build_block(model_config["C4_up"], activation_list=self.activation_list)

        mid_object_block1 = build_block(model_config["C5"], activation_list=self.activation_list)
        mid_object_upsample = build_block(model_config["C5_up"], activation_list=self.activation_list)

        small_object_block1 = build_block(model_config["C6"], activation_list=self.activation_list)
        neck = {"C4": big_object_block1,
                "C4_up": big_object_upsample,
                "C5": mid_object_block1,
                "C5_up": mid_object_upsample,
                "C6": small_object_block1}
        return neck

    def build_head(self, model_config):
        # object detection layer for big object
        big_object_block2 = build_block(model_config["C4"], activation_list=self.activation_list)

        # object detection layer for mid object
        mid_object_block2 = build_block(model_config["C5"], activation_list=self.activation_list)

        # object detection layer for small object
        small_object_block2 = build_block(model_config["C6"], activation_list=self.activation_list)

        head = {"C4": big_object_block2,
                "C5": mid_object_block2,
                "C6": small_object_block2}
        return head

    def forward(self, imgs):
        output_C4, output_C5, output_C6 = self.backbone(imgs)

        # for big object
        latent_b = self.neck["C4"](output_C6)
        output_b = self.head["C4"](latent_b)

        # for mid object
        upsample_b = self.neck["C4_up"](latent_b)
        concat_bm = torch.cat([output_C5, upsample_b], dim=1)
        latent_m = self.neck["C5"](concat_bm)
        output_m = self.head["C5"](latent_m)

        # for small object
        upsample_m = self.neck["C5_up"](latent_m)
        concat_ms = torch.cat([output_C4, upsample_m], dim=1)
        latent_s = self.neck["C6"](concat_ms)
        output_s = self.head["C6"](latent_s)
        return output_b, output_m, output_s


if __name__ == "__main__":
    yolo = yoloV3(yolov3_cfg)
    anchors = torch.rand((14, 2)).numpy()
    yolo.get_anchors(anchors)

    print(yolo.anchor_bboxs)
    img = torch.rand((1, 3, 416, 416))
    output_b, output_m, output_s = yolo(img)
    print(output_b.shape)
    print(output_m.shape)
    print(output_s.shape)
