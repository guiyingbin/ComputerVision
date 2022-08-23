from Utils.Layers import build_block
import torch.nn as nn
import torch

class darknet(nn.Module):
    def __init__(self, model_type="darknet_19"):
        super(darknet, self).__init__()

        if model_type=="darknet_19":
            self.model_config = {
                                "block1":
                                    [["Conv", 3, 32, 3, 1, 1],
                                     ["MaxPool", 2, 2]],
                                "block2":
                                    [["Conv", 32, 64, 3, 1, 1],
                                     ["MaxPool", 2, 2]],
                                "block3":
                                    [["Conv", 64, 128, 3, 1, 1],
                                     ["Conv", 128, 64, 1, 0, 1],
                                     ["Conv", 64, 128, 3, 1, 1],
                                     ["MaxPool", 2, 2]],
                                "block4":
                                    [["Conv", 128, 256, 3, 1, 1],
                                     ["Conv", 256, 128, 1, 0, 1],
                                     ["Conv", 128, 256, 3, 1, 1],
                                     ["MaxPool", 2, 2]],
                                "block5":
                                    [["Conv", 256, 512, 3, 1, 1],
                                     ["Conv", 512, 256, 1, 0, 1],
                                     ["Conv", 256, 512, 3, 1, 1],
                                     ["Conv", 512, 256, 1, 0, 1],
                                     ["Conv", 256, 512, 3, 1, 1],
                                     ["MaxPool", 2, 2]],
                                "block6":
                                    [["Conv", 512, 1024, 3, 1, 1],
                                     ["Conv", 1024, 512, 1, 0, 1],
                                     ["Conv", 512, 1024, 3, 1, 1],
                                     ["Conv", 1024, 512, 1, 0, 1],
                                     ["Conv", 512, 1024, 3, 1, 1]],
                                "block7":
                                    [["Conv", 1024, 1000, 1, 0, 1],
                                     ["AvgPool", 7, 1]]}

        self.darknet_model = nn.Sequential()
        self.build_darknet()


    def forward(self, img):
        output = self.darknet_model(img)
        return output.softmax(1)

    def build_darknet(self):
        # block1

        for block_name, block_list in self.model_config.items():
            self.darknet_model.add_module(block_name, build_block(block_list))

if __name__ == "__main__":
    darknet = darknet()
    img = torch.rand((1, 3, 224, 224))
    print(darknet(img))