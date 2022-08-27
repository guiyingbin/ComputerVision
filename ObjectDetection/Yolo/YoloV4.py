from ObjectDetection.Yolo.YoloV3 import yoloV3
from ObjectDetection.Config.YoloConfig import yolov4_cfg
import torch

class yoloV4(yoloV3):
    def __init__(self, cfg):
        super(yoloV4, self).__init__(cfg)


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