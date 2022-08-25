import torch
import torch.nn as nn
from .DarkNet import darknet

class yoloV2(nn.Module):
    def __init__(self, n_anchors: int, n_class: int, img_size=448):
        """
        implementation of yolo v2
        paper url: https://arxiv.org/abs/1612.08242
        Compared to yolo v1, the improvements of yolo V2 are shown following:
            1. Batch Normaliazation is added behind all convolutional layer
            2. The resolution of finetuned backbone is 448*448, while that of yolo V1 is 224 * 224
            3. Use anchors boxes, and several objects can be predict in one grid
            4. Use kmeans to find best anchors boxes
        :param n_anchors: number of anchors
        :param n_class: number of objects
        :param img_size: the size of image, and the default value is 448
        """
        super(yoloV2, self).__init__()
        self.n_class = n_class
        self.n_anchors = n_anchors
        self.output_channel = self.n_anchors * (self.n_class + 5)
        self.darknet = darknet(output_channel=self.output_channel, input_size=img_size)

    def forward(self, images):
        latent = self.darknet(images)
        # (B, C, H, W) -> (B, H, W, C)
        latent = latent.permute(0, 2, 3, 1)
        # (B, H, W, C) -> (B, H*W, n_anchors, n_class+5)
        latent = latent.reshape(latent.shape[0], -1, self.n_anchors, self.n_class+5)
        dx = torch.sigmoid(latent[:, :, :, 0])
        dy = torch.sigmoid(latent[:, :, :, 1])
        dw = torch.exp(latent[:, :, :, 2])
        dh = torch.exp(latent[:, :, :, 3])
        iou_pred = torch.sigmoid(latent[:, :, :, 4])
        cls_pred = torch.softmax(latent[:, :, :, 5:], dim=-1)

        return dx, dy, dw, dh, iou_pred, cls_pred