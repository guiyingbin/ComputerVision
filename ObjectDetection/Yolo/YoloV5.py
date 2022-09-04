from ObjectDetection.Yolo.YoloV4 import yoloV4
from ObjectDetection.Config.YoloConfig import yolov5s_cfg
import torch

class yoloV5(yoloV4):
    def __init__(self, cfg):
        """
        reference:https://zhuanlan.zhihu.com/p/172121380
        No paper
        :param cfg:
        """
        super(yoloV5, self).__init__(cfg)


if __name__ == "__main__":
    yolo = yoloV5(yolov5s_cfg)
    anchors = torch.rand((14, 2)).numpy()
    yolo.get_anchors(anchors)

    print(yolo.anchor_bboxs)
    img = torch.rand((1, 3, 608, 608))
    output_b, output_m, output_s = yolo(img)
    print(output_b.shape)
    print(output_m.shape)
    print(output_s.shape)