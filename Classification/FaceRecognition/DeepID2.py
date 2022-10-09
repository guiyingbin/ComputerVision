from Classification.FaceRecognition.DeepID import deepID
import torch

class deepID2(deepID):
    def __init__(self, n_convnet=200):
        super(deepID2, self).__init__(n_convnet)
        self.m = torch.rand(size=(1,), requires_grad=True)

    def get_verification_pred(self, imgs1, imgs2):
        features1 = self.forward(imgs1)
        features2 = self.forward(imgs2)
        features1 = torch.cat(features1, dim=1)
        features2 = torch.cat(features2, dim=1)
        return features1, features2

    def get_loss(self, imgs1, imgs2, y):
        idx_1 = torch.where(y == 1.0)
        idx_0 = torch.where(y == -1.0)
        f1, f2 = self.get_verification_pred(imgs1, imgs2)
        verf = 0.5 * torch.sqrt(torch.pow((f1[idx_1]-f2[idx_1]),2)).sum() + \
            0.5 * torch.clamp_min_(self.m-torch.sqrt(torch.pow((f1[idx_0]-f2[idx_0]), 2)).sum(), 0)
        return verf

if __name__ == "__main__":
    img2 = torch.rand((3, 3, 3, 640, 640))
    img1 = torch.rand((3, 3, 3, 640, 640))
    y = torch.FloatTensor([1, 1, -1])
    di2 = deepID2(3)
    print(di2.get_loss(img1,img2,y))