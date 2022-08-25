from Utils.Layers import build_block
import torch.nn as nn
import torch


class darknet(nn.Module):
    def __init__(self, model_type="darknet_19", input_size=448, output_channel=1000,
                 activation_list=None):
        super(darknet, self).__init__()
        if activation_list is None:
            activation_list = ["LeakyReLU", 0.2]
        self.activation_list = activation_list
        if model_type == "darknet_19":
            self.model_config = get_darenet19_config(input_size=input_size, output_channel=output_channel)

        self.darknet_model = nn.Sequential()
        self.build_darknet()

    def forward(self, img):
        output = self.darknet_model(img)
        return output.softmax(1)

    def build_darknet(self):
        # block1

        for block_name, block_list in self.model_config.items():
            self.darknet_model.add_module(block_name, build_block(block_list, self.activation_list))


def get_darenet19_config(input_size=448, output_channel=1000):
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
             ["BatchNorm", 1024]],
        "block7":
            [["Conv", 1024, output_channel, 1, 0, 1],
             ["BatchNorm", output_channel],
             ["AvgPool", input_size // 32, 1]]}

    return model_config


if __name__ == "__main__":
    darknet = darknet()
    img = torch.rand((1, 3, 448, 448))
    print(darknet(img).shape)
