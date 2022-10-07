import torch.nn as nn


class deepFace(nn.Module):
    def __init__(self):
        """
        Implemention of DeepFace
        The contribution of DeepFace:
        1. An effective DNN(a simple network)
        2. An effective face alignment method
        """
        super(deepFace, self).__init__()
        self.model = self.build_model()

    def face_alignment(self, x):
        """
        In th paper of DeepFace, the 2D alignment is just like face detection
        3D alignment is to align faces undergoing out of-plane rotations, which is achieved by using
        a second SVR to localize additional 67 fiducial points x2d in the 2D-aligned crop,
        :param x:
        :return:
        """
        # x2d = SVR(x)
        # x3d = SVR(x)
        # x = T(x)  T is a piece-wise affine transformation, and x is result of frontalization
        return x

    def build_model(self):
        model = nn.Sequential(nn.Conv2d(3, 32, 11, 1, 0),
                              nn.ReLU(),
                              nn.Conv2d(32, 32, 3, 2, 1),
                              nn.ReLU(),
                              nn.Conv2d(32, 16, 9, 1, 0),
                              nn.ReLU(),
                              nn.Conv2d(16, 16, 9, 1, 0),
                              nn.ReLU(),
                              nn.Conv2d(16, 16, 7, 2, 0),
                              nn.ReLU(),
                              nn.Conv2d(16, 16, 5, 1, 0))
        return model

    def forward(self, x):
        x = self.face_alignment(x)
        output = self.model(x)
        return output

if __name__ == "__main__":
    import torch
    x = torch.rand((1, 3, 152, 152))
    df = deepFace()
    print(df(x).shape)