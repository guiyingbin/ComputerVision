"""
This is configuration of training process of Ocr
@author:guiyingbin
@time:2022/09/03
"""
import torch

from DataSet.Utils.strConverter import textConverter
from ObjectDetection.Config.OcrConfig import crnn_model_cfg
from ObjectDetection.Ocr.CRNN import crnn

class base_train_cfg:
    epoch = None
    lr = None
    model = None
    optimizer = None
    criterion = None
    scheduler = None
    data_path = None

class crnn_train_cfg(base_train_cfg):
    epoch = 10
    lr = 0.01
    seed = 42
    batch_size = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = crnn(crnn_model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CTCLoss()
    img_size = (32, 240)
    data_path = r"D:\ocr_data\train_label.csv"
    imgs_dir = r"D:\ocr_data\train_images"
    text_converter = textConverter(crnn_model_cfg)