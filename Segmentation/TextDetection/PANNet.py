from Segmentation.TextDetection.PSENet import pseNet
from Segmentation.Config.TextDetectionConfig import pannet_cfg
from Segmentation.Utils.Layers import build_block

class panNet(pseNet):
    def __init__(self, cfg=pannet_cfg):
        super(panNet, self).__init__(cfg)
        self.cfg = cfg

    def build_neck(self, neck_config):
        neck = {}
        for i, (block_name, block_list) in enumerate(neck_config.items()):
            if block_name == "FPEM_Block":
                for block in block_list:
                    neck[block_name+str(i)] = build_block([block], self.cfg.neck_activation_list)
            else:
                neck[block_name] = build_block(block_list, self.cfg.neck_activation_list)
        return neck

    def forward(self, imgs):
        f2, f3, f4, f5 = self.backbone(imgs)
        p2, p3, p4, p5 = 0, 0, 0, 0

        for neck_name, neck_layer in self.neck.items():
            if neck_name.startswith("FPEM_Block"):
                f2, f3, f4, f5 = neck_layer([f2, f3, f4, f5])
                p2 += f2
                p3 += f3
                p4 += f4
                p5 += f5

        output = self.neck["FFM_Block"]([p2, p3, p4, p5])
        text_region = self.head["text_region_head"](output)
        kernel = self.head["kernel_head"](output)
        smiliarity = self.head["similarity_head"](output)
        return text_region, kernel, smiliarity


if __name__ == "__main__":
    import torch
    dbnet = panNet()
    imgs = torch.rand((1, 3, 640, 640))
    binary, threshold, output = dbnet(imgs)
    print(binary.shape)
    print(threshold.shape)
    print(output.shape)