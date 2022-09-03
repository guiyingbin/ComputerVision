import torch
import pandas as pd
from Train.Config.OcrConfig import crnn_train_cfg
from Utils.BaseClass import baseTrainer
from DataSet.Ocr.baiduOcr import baiduOcr
from torch.utils.data import DataLoader
from tqdm import tqdm

from Utils.Tools import AverageMeter

torch.nn.CTCLoss()


class Trainer(baseTrainer):
    def __init__(self, cfg=crnn_train_cfg):
        super(Trainer, self).__init__()
        self.device = cfg.device
        self.model = cfg.model.to(self.device)
        self.optimzer = cfg.optimizer
        self.criterion = cfg.criterion.to(self.device)
        self.set_seed(cfg.seed)
        self.dataframe = pd.read_csv(cfg.data_path, encoding="gbk").sample(frac=1, random_state=cfg.seed)
        self.converter = cfg.text_converter
        self.cfg = cfg

    def train_one_epoch(self, dataloader, is_train=True):
        losses = AverageMeter()
        pbar = tqdm(dataloader, total=len(dataloader))
        for imgs, idxs in pbar:
            imgs = imgs.to(self.device)
            label_str = self.dataframe.iloc[idxs.type(torch.int32).tolist(), 1].values
            label_list = []
            target_lengths = []
            for text in label_str:
                encoded_text = self.converter.encode(text)
                label_list.extend(encoded_text)
                target_lengths.append(len(encoded_text))
            target_lengths = torch.IntTensor(target_lengths)
            labels = torch.IntTensor(label_list)

            if is_train:
                preds = self.model(imgs)
                T, N, C = preds.shape
                input_lengths = torch.IntTensor([T]*N)
                loss = self.criterion(preds.cpu().log_softmax(-1), labels, input_lengths, target_lengths)
                losses.update(loss.item(), n=N)
                # print("train loss:{}".format(loss.item()))
            else:
                with torch.no_grad():
                    preds = self.model(imgs)
                    T, N, C = preds.shape
                    input_lengths = torch.IntTensor([T * N])
                    loss = self.criterion(preds.cpu(), labels, input_lengths, target_lengths)
                    losses.update(loss.item(), n=N)
                # print("val loss:{}".format(loss.item()))
            pbar.set_postfix(loss=losses.get_avg())
    def train_fold(self, fold):
        print("=============fold:{}============".format(fold))
        train_dataset = baiduOcr(img_size=self.cfg.img_size,
                                 img_dir=self.cfg.imgs_dir,
                                 dataframe=self.dataframe)
        train_dataloader = DataLoader(train_dataset, batch_size=self.cfg.batch_size)
        self.train_one_epoch(train_dataloader)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_fold(fold=0)
