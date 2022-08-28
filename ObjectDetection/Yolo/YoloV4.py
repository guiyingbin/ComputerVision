from ObjectDetection.Yolo.YoloV3 import yoloV3
from ObjectDetection.Config.YoloConfig import yolov4_cfg
import torch

from ObjectDetection.Utils.Layers import build_block


class yoloV4(yoloV3):
    def __init__(self, cfg):
        super(yoloV4, self).__init__(cfg)

    def build_neck(self, model_config):
        big_object_block1 = build_block(model_config["C6"], activation_list=self.activation_list)
        big_object_upsample = build_block(model_config["C6_up"], activation_list=self.activation_list)

        mid_object_block1 = build_block(model_config["C5"], activation_list=self.activation_list)
        mid_object_upsample = build_block(model_config["C5_up"], activation_list=self.activation_list)

        small_object_block1 = build_block(model_config["C4"], activation_list=self.activation_list)
        pan = build_block(model_config["PAN"], activation_list=["LeakyReLU", 0.2])
        neck = {"C6": big_object_block1,
                "C6_up": big_object_upsample,
                "C5": mid_object_block1,
                "C5_up": mid_object_upsample,
                "C4": small_object_block1,
                "PAN": pan}
        return neck

    def forward(self, imgs):
        output_C4, output_C5, output_C6 = self.backbone(imgs)

        # for big object
        latent_b = self.neck["C6"](output_C6)
        # for mid object
        upsample_b = self.neck["C6_up"](latent_b)
        concat_bm = torch.cat([output_C5, upsample_b], dim=1)
        latent_m = self.neck["C5"](concat_bm)
        # for small object
        upsample_m = self.neck["C5_up"](latent_m)
        concat_ms = torch.cat([output_C4, upsample_m], dim=1)
        latent_s = self.neck["C4"](concat_ms)

        latent_s, latent_m, latent_b = self.neck["PAN"]([latent_s, latent_m, latent_b])
        output_b = self.head["C6"](latent_b)
        output_m = self.head["C5"](latent_m)
        output_s = self.head["C4"](latent_s)

        output = [output_b, output_m, output_s]
        return output

if __name__ == "__main__":
    yolo = yoloV4(yolov4_cfg)
    anchors = torch.rand((14, 2)).numpy()
    yolo.get_anchors(anchors)

    print(yolo.anchor_bboxs)
    img = torch.rand((1, 3, 416, 416))
    output_b, output_m, output_s = yolo(img)
    print(output_b.shape)
    print(output_m.shape)
    print(output_s.shape)