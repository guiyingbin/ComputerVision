import torch
import torch.nn as nn
from Utils.Layers import build_block
from ObjectDetection.Yolo.BaseYolo import BaseYolo

class yoloV1(BaseYolo):
    def __init__(self, n_anchors=2, S=7, n_class=20, mid_dim=4096):
        """
        implementation of yolo v1
        paper url: https://arxiv.org/pdf/1506.02640.pdf
        In yolo v1, the image is spilted into S*S grid, and there are B bounding boxes in each grid.
        Besides, each grid predicts C conditional class probability
        :param B: the number of bounding boxes of each grid
        :param S: the shape of grid of one image
        :param C: the number of different object
        """
        super(yoloV1, self).__init__()
        self.n_anchors = n_anchors
        self.n_class = n_class
        self.S = S
        backbone_config = get_yolo_backbone()
        self.backbone = self.build_backbone(backbone_config)

        head_config = {
            "head1":[["FC", 1024 * 7 * 7, mid_dim],
                     ["FC", mid_dim, self.S * self.S * (self.n_anchors * 5 + self.n_class)]]}
        self.head = self.build_head(head_config)

    def build_backbone(self, model_config):
        backbone = nn.Sequential()
        for block_name, block_list in model_config.items():
            backbone.add_module(block_name, build_block(block_list))
        return backbone

    def build_head(self, model_config):
        head = nn.Sequential()
        for block_name, block_list in model_config.items():
            head.add_module(block_name, build_block(block_list))
        return head

    def forward(self, imgs):
        B, C, H, W = imgs.shape
        assert H == 448 and W == 448

        latent = self.backbone(imgs)
        latent = latent.reshape((latent.shape[0], -1))
        output = self.head(latent)
        output = output.reshape((output.shape[0], self.S, self.S, -1))
        dx = torch.sigmoid(output[:, :, :, 0:self.B*5:5])
        dy = torch.sigmoid(output[:, :, :, 1:self.B*5:5])
        dw = torch.exp(output[:, :, :, 2:self.B*5:5])
        dh = torch.exp(output[:, :, :, 3:self.B*5:5])
        iou_pred = torch.sigmoid(output[:, :, :, 4:self.B*5:5])
        cls_pred = torch.sigmoid(output[:, :, :, self.B*5:])
        bbox_pred = torch.stack([dx, dy, dw, dh], dim=-1)
        return bbox_pred, iou_pred, cls_pred



def get_yolo_backbone():
    model_config = {
        "block1":
                [["Conv", 3, 192, 7, 3, 2],
                 ["MaxPool", 2, 2]],
        "block2":
                [["Conv", 192, 128, 3, 1, 1],
                 ["MaxPool", 2, 2]],
        "block3":
                [["Conv", 128, 256, 1, 0, 1],
                 ["Conv", 256, 256, 3, 1, 1],
                 ["Conv", 256, 512, 1, 0, 1],
                 ["Conv", 512, 512, 3, 1, 1],
                 ["MaxPool", 2, 2]],
        "block4":
                [["Conv", 512, 256, 1, 0, 1],
                 ["Conv", 256, 512, 3, 1, 1],
                 ["Conv", 512, 256, 1, 0, 1],
                 ["Conv", 256, 512, 3, 1, 1],
                 ["Conv", 512, 256, 1, 0, 1],
                 ["Conv", 256, 512, 3, 1, 1],
                 ["Conv", 512, 256, 1, 0, 1],
                 ["Conv", 256, 512, 3, 1, 1],
                 ["Conv", 512, 512, 1, 0, 1],
                 ["Conv", 512, 1024, 3, 1, 1],
                 ["MaxPool", 2, 2]],
        "block5":
                [["Conv", 1024, 512, 1, 0, 1],
                 ["Conv", 512, 1024, 3, 1, 1],
                 ["Conv", 1024, 512, 1, 0, 1],
                 ["Conv", 512, 1024, 3, 1, 1],
                 ["Conv", 1024, 1024, 3, 1, 1],
                 ["Conv", 1024, 1024, 3, 1, 2]],
        "block6":
                [["Conv", 1024, 1024, 3, 1, 1],
                 ["Conv", 1024, 1024, 3, 1, 1]]
    }
    return model_config
if __name__ == "__main__":
   model = yoloV1()
   img = torch.rand((1, 3,448,448))
   bbox_pred, iou_pred, cls_pred = model(img)
   print(bbox_pred.shape)
