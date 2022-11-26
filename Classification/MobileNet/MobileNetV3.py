import timm
import torch.nn as nn
from timm.models.mobilenetv3 import mobilenetv3_small_050


class mobileNetV3(nn.Module):
    def __init__(self, model_name="mobilenetv3_small_050", pretrained=False, out_features=True,
                 output_layer_index=[2, 3, 4]):
        super(mobileNetV3, self).__init__()
        if out_features:
            self.model = mobilenetv3_small_050(pretrained=pretrained, features_only=True)
        else:
            self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.output_layer_index = output_layer_index

    def forward(self, x):
        output = []
        pred = self.model(x)
        for index in self.output_layer_index:
            output.append(pred[index])
        return output


if __name__ == "__main__":
    import torch

    mb = mobileNetV3()
    x = torch.randn(size=(1, 3, 640, 640))
    for each in mb(x):
        print(each.shape)