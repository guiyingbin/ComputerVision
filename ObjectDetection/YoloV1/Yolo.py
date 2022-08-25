import torch
import torch.nn as nn
from Utils.Layers import build_block

class yoloV1(nn.Module):
    def __init__(self, B=2, S=7, C=20):
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
        self.B = B
        self.C = C
        self.S = S
        self.backbone = self.build_backbone()
        self.head = self.build_head()

    def build_head(self, mid_dim=4096):
        head = nn.Sequential()
        head.add_module("fc", nn.Linear(1024*7*7, mid_dim))
        head.add_module("head", nn.Linear(mid_dim, self.S*self.S*(self.B*5+self.C)))
        return head

    def build_backbone(self):
        backbone = nn.Sequential()
        # Block1
        block_list1 = [["Conv", 3, 192, 7, 3, 2],
                       ["MaxPool", 2, 2]]
        block1 = build_block(block_list1)
        backbone.add_module("Block1", block1)

        # Block2
        block_list2 = [["Conv", 192, 128, 3, 1, 1],
                       ["MaxPool", 2, 2]]
        block2 = build_block(block_list2)
        backbone.add_module("Block2", block2)

        # Block3
        block_list3 = [["Conv", 128, 256, 1, 0, 1],
                       ["Conv", 256, 256, 3, 1, 1],
                       ["Conv", 256, 512, 1, 0, 1],
                       ["Conv", 512, 512, 3, 1, 1],
                       ["MaxPool", 2, 2]]
        block3 = build_block(block_list3)
        backbone.add_module("Block3", block3)

        # Block4
        block_list4 = [["Conv", 512, 256, 1, 0, 1],
                       ["Conv", 256, 512, 3, 1, 1],
                       ["Conv", 512, 256, 1, 0, 1],
                       ["Conv", 256, 512, 3, 1, 1],
                       ["Conv", 512, 256, 1, 0, 1],
                       ["Conv", 256, 512, 3, 1, 1],
                       ["Conv", 512, 256, 1, 0, 1],
                       ["Conv", 256, 512, 3, 1, 1],
                       ["Conv", 512, 512, 1, 0, 1],
                       ["Conv", 512, 1024, 3, 1, 1],
                       ["MaxPool", 2, 2]]
        block4 = build_block(block_list4)
        backbone.add_module("Block4", block4)

        # Block4
        block_list4 = [["Conv", 512, 256, 1, 0, 1],
                       ["Conv", 256, 512, 3, 1, 1],
                       ["Conv", 512, 256, 1, 0, 1],
                       ["Conv", 256, 512, 3, 1, 1],
                       ["Conv", 512, 256, 1, 0, 1],
                       ["Conv", 256, 512, 3, 1, 1],
                       ["Conv", 512, 256, 1, 0, 1],
                       ["Conv", 256, 512, 3, 1, 1],
                       ["Conv", 512, 512, 1, 0, 1],
                       ["Conv", 512, 1024, 3, 1, 1],
                       ["MaxPool", 2, 2]]
        block4 = build_block(block_list4)
        backbone.add_module("Block4", block4)

        # Block5
        block_list5 = [["Conv", 1024, 512, 1, 0, 1],
                       ["Conv", 512, 1024, 3, 1, 1],
                       ["Conv", 1024, 512, 1, 0, 1],
                       ["Conv", 512, 1024, 3, 1, 1],
                       ["Conv", 1024, 1024, 3, 1, 1],
                       ["Conv", 1024, 1024, 3, 1, 2]]
        block5 = build_block(block_list5)
        backbone.add_module("Block5", block5)

        # Block6
        block_list6 = [["Conv", 1024, 1024, 3, 1, 1],
                       ["Conv", 1024, 1024, 3, 1, 1]]
        block6 = build_block(block_list6)
        backbone.add_module("Block6", block6)
        return backbone

    def forward(self, x):
        B, C, H, W = x.shape
        assert H==448 and W==448

        latent = self.backbone(x)
        latent = latent.reshape((latent.shape[0], -1))
        output = self.head(latent)
        output = output.reshape((output.shape[0], self.S, self.S, -1))
        dx = torch.sigmoid(output[:, :, :, 0:self.B*5:5])
        dy = torch.sigmoid(output[:, :, :, 1:self.B*5:5])
        dw = torch.exp(output[:, :, :, 2:self.B*5:5])
        dh = torch.exp(output[:, :, :, 3:self.B*5:5])
        iou_pred = torch.sigmoid(output[:, :, :, 4:self.B*5:5])
        cls_pred = torch.sigmoid(output[:, :, :, 5:])
        return dx, dy, dw, dh, iou_pred, cls_pred

if __name__ == "__main__":
   model = yoloV1()
   img = torch.rand((1, 3,448,448))
   print(model(img).shape)
