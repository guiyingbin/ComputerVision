from Classification.Config.VGGConfig import vgg_cfg
import torch.nn as nn
from Classification.Utils.Layers import build_block


class vgg(nn.Module):
    def __init__(self, cfg=vgg_cfg, model_type="vgg_16", activation_list=None):
        super(vgg, self).__init__()
        if activation_list is None:
            activation_list = ["ReLU"]
        self.activation_list = activation_list
        self.model = self.build_model(cfg[model_type])

    def build_model(self, model_config):
        model = self.build_multi_output_model(model_config, ["block2", "block3", "block4"])
        return model

    def build_multi_output_model(self, model_config: dict, output_point: list):
        model = []
        temp_model = nn.Sequential()
        for block_name, block_list in model_config.items():
            temp_model.add_module(block_name, build_block(block_list, self.activation_list))
            if block_name in output_point:
                model.append(temp_model)
                temp_model = nn.Sequential()
        model.append(temp_model)
        return model

    def forward(self, imgs):
        C2, C3, C4, C5 = self.model
        O2 = C2(imgs)
        O3 = C3(O2)
        O4 = C4(O3)
        O5 = C5(O4)
        return O2, O3, O4, O5


if __name__ == "__main__":
    import torch
    imgs = torch.rand((2, 3, 224, 224))
    model = vgg(model_type="vgg_19")
    O2, O3, O4, O5 = model(imgs)
    print(O2.shape)
    print(O3.shape)
    print(O4.shape)
    print(O5.shape)