import torch.nn as nn
from Segmentation.Config.ResNetConfig import resnet_cfg


class resnet(nn.Module):
    def __init__(self, model_type="resnet_50", activation_list=None):
        """
        ResNet Model
        :param model_type:
        :param activation_list:
        """
        super(resnet, self).__init__()
        if activation_list is None:
            activation_list = ["ReLU"]

        self.activation_list = activation_list
        self.model_type = model_type

        model_config = resnet_cfg[model_type]
        self.resnet_model = self.build_resnet(model_config)

    def build_resnet(self, model_config):
        model = self.build_multi_output_model(model_config, [])
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

    def forward(self, img):
        P3, P4, P5, P6 = self.darknet_model
        output1 = P3(img)
        output2 = P4(output1)
        output3 = P5(output2)
        output4 = P6(output3)
        return output1, output2, output3, output4