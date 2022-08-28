from Utils.Layers import build_block
import torch.nn as nn
import torch
from ObjectDetection.Config.DarkNetConfig import darknet_cfg

class darknet(nn.Module):
    def __init__(self, model_type="darknet_19", activation_list=None):
        """
        The Darknet Model
        :param model_type: the version of darknet
        :param activation_list: the setting of activation layer
        """
        super(darknet, self).__init__()

        if activation_list is None:
            activation_list = ["LeakyReLU", 0.2]

        self.activation_list = activation_list
        self.model_type = model_type

        model_config = darknet_cfg[model_type]
        self.darknet_model = self.build_darknet(model_config)


    def forward(self, img):
        if self.model_type == "darknet_19":
            output = self.darknet_model(img)
            return output
        elif self.model_type in ["darknet_53", "cspdarknet_53"]:
            C4, C5, C6 = self.darknet_model
            output1 = C4(img)
            output2 = C5(output1)
            output3 = C6(output2)
            return output1, output2, output3

    def build_darknet(self, model_config):
        if self.model_type == "darknet_19":
            model = nn.Sequential()
            for block_name, block_list in model_config.items():
                model.add_module(block_name, build_block(block_list, self.activation_list))

        elif self.model_type in ["darknet_53", "cspdarknet_53"]:
            model = []
            temp_model = nn.Sequential()
            for block_name, block_list in model_config.items():
                if block_name in ["block7", "block9"]:
                    model.append(temp_model)
                    temp_model = nn.Sequential()
                temp_model.add_module(block_name, build_block(block_list, self.activation_list))
            model.append(temp_model)

        return model


if __name__ == "__main__":

    model = darknet(model_type="cspdarknet_53")
    img = torch.rand((1, 3, 416, 416))
    output1, output2, output3 = model(img)
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)
