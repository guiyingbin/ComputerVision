from Utils.Layers import build_block
import torch.nn as nn
from ObjectDetection.Config.OcrConfig import crnn_cfg
import torch


class crnn(nn.Module):
    def __init__(self, cfg):
        super(crnn, self).__init__()
        self.head = self.build_head(cfg.head_config, cfg.head_activation)
        self.backbone = self.build_backbone(cfg.backbone_config, cfg.backbone_activation)

    def build_head(self, head_config, activation_list):
        head = nn.Sequential()
        for block_name, block_list in head_config.items():
            head.add_module(block_name, build_block(block_list))
        return head

    def build_backbone(self, backbone_config, activation_list):
        backbone = nn.Sequential()
        for block_name, block_list in backbone_config.items():
            backbone.add_module(block_name, build_block(block_list))
        return backbone

    def forward(self, x):
        latent = self.backbone(x)
        B, C, H, W = latent.size()
        print(latent.size())
        assert H == 1
        latent = latent.reshape(B, C, H*W).permute(2, 0, 1)
        output = self.head(latent)
        return output

if __name__ == "__main__":
    crnn_model = crnn(crnn_cfg)
    x = torch.randn((3, 3, 32, 240))
    print(crnn_model(x).shape)