import torch
import torch.nn as nn
import numpy as np

class SRM_filter(nn.Module):
    def __init__(self):
        super().__init__()
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = np.asarray([[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])
        self.layer = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.layer.weight = nn.Parameter(torch.from_numpy(filters).float())
        for param in self.layer.parameters():
            param.requires_grad = False
    def forward(self, img):
        return self.layer(img)


if __name__ == "__main__":
    import cv2
    model = SRM_filter()
    # img = cv2.imread(r"C:\Users\Guiyingbin\Desktop\f644203371b0f046698fd893cee9b595_copy1.jpg")
    img = cv2.imread(r"C:\Users\Guiyingbin\Downloads\f644203371b0f046698fd893cee9b595.jpeg")
    img_tensor = torch.from_numpy(img[np.newaxis, :, :, :].transpose(0, 3, 1, 2))
    noise = model(img_tensor.float())

    noise_img = noise[0].numpy().transpose(1, 2, 0)
    cv2.imwrite("temp1.jpg", noise_img)