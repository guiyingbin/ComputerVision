from Classification.FaceRecognition.DeepFace import deepFace
import torch.nn as nn
import torch


class deepID(deepFace):
    def __init__(self, n_convnet=60, n_class=20):
        """
        In DeepID, 60 ConvNets is trained.Features are extracted from 60 face patches with ten regions, three scales, and RGB or gray
        channels.  Each ConvNets extracts two 160-dimensional DeepID vectors from a particular patch and its horizontally flipped counterpart
        """
        self.n_convnet = n_convnet
        self.n_class = n_class
        super(deepID, self).__init__()
        self.model = self.build_model()
        self.recognition_head = nn.Linear(self.n_convnet*160*2, self.n_class)

    def build_model(self):
        model = []
        for i in range(self.n_convnet):
            model.append(nn.Sequential(nn.Conv2d(3, 20, 4, 1, 0),
                                       nn.MaxPool2d(3, 2, 1),
                                       nn.Conv2d(20, 40, 3, 1, 0),
                                       nn.MaxPool2d(3, 2, 1),
                                       nn.Conv2d(40, 60, 3, 1, 0),
                                       nn.MaxPool2d(2, 2, 0),
                                       nn.Conv2d(60, 160, 2, 1, 0),
                                       nn.AdaptiveAvgPool2d((1, 1))))
        return model

    def forward(self, imgs):
        """
        :param imgs: list, [Batch patch1(B, C, H, W....,]
        :return:
        """
        assert len(imgs) == self.n_convnet
        output = []
        for i, img in enumerate(imgs):
            B, C, H, W = img.shape
            model = self.model[i]
            output.append(model(img).reshape(B, -1))
            flip_img = torch.flip(img, dims=[3])
            output.append(model(flip_img).reshape(B, -1))

        return output

    def get_recognition_pred(self, imgs):
        features = self.forward(imgs)
        features = torch.cat(features, dim=1)
        cls_logit = self.recognition_head(features)
        return cls_logit


if __name__ == "__main__":
    import torch
    img = torch.rand((3, 3, 39, 39))
    x = [img]*60
    df = deepID(n_convnet=60)
    print(df.get_recognition_pred(x).shape)
