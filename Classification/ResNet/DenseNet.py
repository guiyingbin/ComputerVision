from Classification.ResNet.ResNet import resnet
from Classification.Config.ResNetConfig import densenet_cfg


class denseNet(resnet):
    def __init__(self, cfg=densenet_cfg, model_type="densenet_121", activation_list=None):
        super(denseNet, self).__init__(cfg, model_type, activation_list)

    def build_model(self, model_config):
        model = self.build_multi_output_model(model_config, ["stage2", "stage3", "stage4"])
        return model


if __name__ == "__main__":
    import torch
    res = denseNet(model_type="densenet_264")
    img = torch.rand((1, 3, 640, 640))
    outputs = res(img)
    for output in outputs:
        print(output.shape)