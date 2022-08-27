from Utils.Layers import build_block
import torch.nn as nn
import torch

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

        if self.model_type == "darknet_19":
            model_config = get_darenet19_config()
            self.darknet_model = self.build_darknet(model_config)
        elif self.model_type == "darknet_53":
            model_config = get_darenet53_config()
            self.darknet_model_list = self.build_darknet(model_config)


    def forward(self, img):
        if self.model_type == "darknet_19":
            output = self.darknet_model(img)
            return output
        elif self.model_type == "darknet_53":
            C4, C5, C6 = self.darknet_model_list
            output1 = C4(img)
            output2 = C5(output1)
            output3 = C6(output2)
            return output1, output2, output3

    def build_darknet(self, model_config):
        if self.model_type == "darknet_19":
            model = nn.Sequential()
            for block_name, block_list in model_config.items():
                model.add_module(block_name, build_block(block_list, self.activation_list))

        elif self.model_type == "darknet_53":
            model = []
            temp_model = nn.Sequential()
            for block_name, block_list in model_config.items():
                if block_name in ["block7", "block9"]:
                    model.append(temp_model)
                    temp_model = nn.Sequential()
                temp_model.add_module(block_name, build_block(block_list, self.activation_list))
            model.append(temp_model)

        return model


def get_darenet19_config():
    model_config = {
        "block1":
            [["Conv", 3, 32, 3, 1, 1],
             ["BatchNorm", 32],
             ["MaxPool", 2, 2]],
        "block2":
            [["Conv", 32, 64, 3, 1, 1],
             ["BatchNorm", 64],
             ["MaxPool", 2, 2]],
        "block3":
            [["Conv", 64, 128, 3, 1, 1],
             ["BatchNorm", 128],
             ["Conv", 128, 64, 1, 0, 1],
             ["BatchNorm", 64],
             ["Conv", 64, 128, 3, 1, 1],
             ["BatchNorm", 128],
             ["MaxPool", 2, 2]],
        "block4":
            [["Conv", 128, 256, 3, 1, 1],
             ["BatchNorm", 256],
             ["Conv", 256, 128, 1, 0, 1],
             ["BatchNorm", 128],
             ["Conv", 128, 256, 3, 1, 1],
             ["BatchNorm", 256],
             ["MaxPool", 2, 2]],
        "block5":
            [["Conv", 256, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["Conv", 512, 256, 1, 0, 1],
             ["BatchNorm", 256],
             ["Conv", 256, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["Conv", 512, 256, 1, 0, 1],
             ["BatchNorm", 256],
             ["Conv", 256, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["MaxPool", 2, 2]],
        "block6":
            [["Conv", 512, 1024, 3, 1, 1],
             ["BatchNorm", 1024],
             ["Conv", 1024, 512, 1, 0, 1],
             ["BatchNorm", 512],
             ["Conv", 512, 1024, 3, 1, 1],
             ["BatchNorm", 1024],
             ["Conv", 1024, 512, 1, 0, 1],
             ["BatchNorm", 512],
             ["Conv", 512, 1024, 3, 1, 1],
             ["BatchNorm", 1024]]}

    return model_config


def get_darenet53_config():
    model_config = {
        "block1":
            [["Conv", 3, 32, 3, 1, 1],
             ["BatchNorm", 32],
             ["Conv", 32, 64, 3, 1, 2]],
        "block2":
            [["Conv", 64, 32, 1, 0, 1],
             ["BatchNorm", 32],
             ["Conv", 32, 64, 3, 1, 1],
             ["BatchNorm", 64],
             ["Residual", 64, 128, 3, 1, 1]],
        "block3":
            [["Conv", 64, 128, 3, 1, 2],
             ["BatchNorm", 128]],
        "block4":
            [["Conv", 128, 64, 1, 0, 1],
             ["BatchNorm", 64],
             ["Conv", 64, 128, 3, 1, 1],
             ["BatchNorm", 128],
             ["Residual", 128, 256, 3, 1, 1]] * 2,
        "block5":
            [["Conv", 128, 256, 3, 1, 2],
             ["BatchNorm", 256]],
        "block6":
            [["Conv", 256, 128, 1, 0, 1],
             ["BatchNorm", 128],
             ["Conv", 128, 256, 3, 1, 1],
             ["BatchNorm", 256],
             ["Residual", 256, 512, 3, 1, 1]] * 8,
        "block7":
            [["Conv", 256, 512, 3, 1, 2],
             ["BatchNorm", 512]],
        "block8":
            [["Conv", 512, 256, 1, 0, 1],
             ["BatchNorm", 256],
             ["Conv", 256, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["Residual", 512, 1024, 3, 1, 1]] * 8,
        "block9":
            [["Conv", 512, 1024, 3, 1, 2],
             ["BatchNorm", 1024]],
        "block10":
            [["Conv", 1024, 512, 1, 0, 1],
             ["BatchNorm", 512],
             ["Conv", 512, 1024, 3, 1, 1],
             ["BatchNorm", 1024],
             ["Residual", 1024, 2048, 3, 1, 1]] * 4
        }
    return model_config


if __name__ == "__main__":
    from Utils.Layers import IntermediateLayerGetter
    model = darknet(model_type="darknet_53")

    img = torch.rand((1, 3, 416, 416))
    output1, output2, output3 = model(img)
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)
