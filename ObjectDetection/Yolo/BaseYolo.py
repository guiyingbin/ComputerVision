import torch.nn as nn


class BaseYolo(nn.Module):
    def __init__(self):
        super(BaseYolo, self).__init__()

    def get_anchors(self, *params, **kwargs):
        pass

    def preprocess(self, *params, **kwargs):
        pass

    def postproces(self, *params, **kwargs):
        pass

    def build_backbone(self, *params, **kwargs):
        pass

    def build_neck(self, *params, **kwargs):
        pass

    def build_head(self, *params, **kwargs):
        pass

    def forward(self, *params, **kwargs):
        pass
