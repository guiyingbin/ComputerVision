import torch
import torch.nn as nn
from ObjectDetection.Utils.Layers import build_block
from sklearn.cluster import KMeans
import numpy as np
from ObjectDetection.Utils.DarkNet import darknet
from ObjectDetection.Yolo.BaseYolo import BaseYolo
from ObjectDetection.Config.YoloConfig import yolov2_cfg

class yoloV2(BaseYolo):
    def __init__(self, cfg):
        """
        implementation of yolo v2
        paper url: https://arxiv.org/abs/1612.08242
        Compared to yolo v1, the improvements of yolo V2 are shown following:
            1. Batch Normaliazation is added behind all convolutional layer
            2. The resolution of finetuned backbone is 448*448, while that of yolo V1 is 224 * 224
            3. Use anchors boxes, and several objects can be predict in one grid
            4. Use kmeans to find best anchors boxes
        :param cfg: configuration
        """
        super(yoloV2, self).__init__()

        self.n_class = cfg.n_class
        self.img_size = cfg.img_size
        self.n_anchors = cfg.n_anchors
        self.activation_list = cfg.activation_list
        self.anchor_bboxs = np.array([])
        self.backbone = self.build_backbone(cfg.backbone_type, self.activation_list)
        self.head = self.build_head(cfg.head_config, self.activation_list)

    def build_backbone(self, model_type, activation_list):
        return darknet(model_type=model_type, activation_list=activation_list)

    def build_head(self, model_config, activation_list):
        head = nn.Sequential()
        for block_name, block_list in model_config.items():
            head.add_module(block_name, build_block(block_list, activation_list))
        return head

    def get_anchors(self, bbox_wh_data: np.array):
        """
        Get prior anchors by Kmeans.
        In the paper of yolo V2, the size of anchors is determined by Kmeans,
        :param bbox_wh_data: np.ndarray (w, h)
        :return: None
        """
        k_means = KMeans(n_clusters=self.n_anchors, random_state=10)
        k_means.fit(bbox_wh_data)
        self.anchor_bboxs = np.round(k_means.cluster_centers_, 1)

    def forward(self, imgs):
        latent = self.backbone(imgs)
        latent = self.head(latent)
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
        bbox_pred = torch.stack([dx, dy, dw, dh], dim=-1)
        return bbox_pred, iou_pred, cls_pred

if __name__ == "__main__":
    a = np.array([[1, 3], [3, 4], [5, 3], [4, 3], [4, 3],
                  [2, 4], [5, 4], [7, 6], [9, 8]])
    yolo = yoloV2(yolov2_cfg)
    yolo.get_anchors(a)
    print(yolo.anchor_bboxs)
    img = torch.rand((1, 3, 448, 448))
    bbox_pred, iou_pred, cls_pred = yolo(img)
    print(bbox_pred.shape)