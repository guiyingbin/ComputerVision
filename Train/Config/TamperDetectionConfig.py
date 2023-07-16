import torch
from Segmentation.TamperDetection.DTD.DTD import DTD
from OcrConfig import base_train_cfg


class dtd_train_cfg(base_train_cfg):
    epoch = 10
    lr = 0.01
    seed = 42
    batch_size = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DTD()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    img_size = (512, 512)
    roots = ""
