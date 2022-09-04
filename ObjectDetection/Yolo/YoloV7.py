from ObjectDetection.Yolo.YoloV5 import yoloV5
from ObjectDetection.Config.YoloConfig import yolov7_cfg
import torch
from ObjectDetection.Utils.Layers import build_block


class yoloV7(yoloV5):
    def __init__(self, cfg=yolov7_cfg):
        super(yoloV7, self).__init__(cfg)
        self.build_backbone(cfg.backbone_type, cfg.backbone_activation_list)
        self.build_neck(cfg.neck_config, cfg.neck_activation_list)
        self.build_head(cfg.head_config, cfg.head_activation_list)

        # The convolution operation required to enter the neck_layer
        prev_conv4_list = [["Conv", 512, 128, 1, 0, 1],
                           ["BatchNorm", 128]]
        prev_conv5_list = [["Conv", 1024, 256, 1, 0, 1],
                           ["BatchNorm", 256]]
        self.prev_conv4 = build_block(prev_conv4_list, activation_list=cfg.neck_activation_list)
        self.prev_conv5 = build_block(prev_conv5_list, activation_list=cfg.neck_activation_list)

    def forward(self, imgs):
        output_C4, output_C5, output_C6 = self.backbone(imgs)
        output_C4 = self.prev_conv4(output_C4)
        output_C5 = self.prev_conv5(output_C5)

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
    yolo = yoloV7()
    img = torch.rand((1, 3, 640, 640))
    output_b, output_m, output_s = yolo(img)
    print(output_b.shape)
    print(output_m.shape)
    print(output_s.shape)