import os
import cv2
from torch.utils.data import Dataset
from DataSet.Utils.tools import resize_img
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


class baiduOcr(Dataset):
    def __init__(self, img_size, img_dir, dataframe, transform=None):
        super(baiduOcr, self).__init__()
        self.img_size = img_size
        self.img_dir = img_dir
        self.total_size = dataframe.shape[0]
        # = pd.read_csv(os.path.join(self.root_path, self.train_path), encoding='gbk')
        self.imgs_path = dataframe["name"]
        self.transform = transform
        self.normalize = A.Compose([A.Normalize(),
                                   ToTensorV2()])

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_path))[:, :, ::-1]
        img = resize_img(img, self.img_size)
        if self.transform:
            img = self.transform(image=img)["image"]
        img = self.normalize(image=img)["image"]
        return img, idx

