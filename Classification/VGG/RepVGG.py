from Classification.VGG.VGG import vgg
from Classification.Config.VGGConfig import repvgg_cfg


class repVGG(vgg):
    def __init__(self, cfg=repvgg_cfg, model_type="repvgg_a0", activation_list=None):
        super(repVGG, self).__init__(cfg, model_type, activation_list)

    def build_model(self, model_config):
        model = self.build_multi_output_model(model_config, ["stage2", "stage3", "stage4"])
        return model


if __name__ == "__main__":
    import torch
    imgs = torch.rand((2, 3, 224, 224))
    model = repVGG()
    O2, O3, O4, O5 = model(imgs)
    print(O2.shape)
    print(O3.shape)
    print(O4.shape)
    print(O5.shape)