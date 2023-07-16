import torch
import pandas as pd
from Train.Config.TamperDetectionConfig import dtd_train_cfg
from Utils.BaseClass import baseTrainer
from DataSet.TamperDetection.dtd_dataset import DocTamperDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from Utils.Tools import AverageMeter


class Trainer(baseTrainer):
    def __init__(self, cfg=dtd_train_cfg):
        super(Trainer, self).__init__()
        self.device = cfg.device
        self.model = cfg.model.to(self.device)
        self.optimzer = cfg.optimizer
        self.criterion = cfg.criterion.to(self.device)
        self.set_seed(cfg.seed)
        self.cfg = cfg

    def train_one_epoch(self, dataloader, is_train=True):
        losses = AverageMeter()
        pbar = tqdm(dataloader, total=len(dataloader))
        self.optimzer.zero_grad()
        for imgs, dcts, masks in pbar:
            imgs = imgs.to(self.device) # [B, 3, H, W]
            dcts = dcts.to(self.device) #[B, H, W]
            masks = masks.to(self.device).unsqueeze(dim=1) # [B, 1, H, W

            if is_train:
                preds = self.model(imgs, dcts)
                loss = self.criterion(preds, masks)
                losses.update(loss.item(), n=preds.shape[0])
                # print("train loss:{}".format(loss.item()))
            else:
                with torch.no_grad():
                    preds = self.model(imgs)
                    loss = self.criterion(preds, masks)
                    losses.update(loss.item(), n=preds.shape[0])
                # print("val loss:{}".format(loss.item()))
            if is_train:
                loss.backward()
                self.optimzer.step()
            pbar.set_postfix(loss=losses.get_avg())

    def train_fold(self, fold):
        print("=============fold:{}============".format(fold))
        train_dataset = DocTamperDataset(roots=self.cfg.roots)
        train_dataloader = DataLoader(train_dataset, batch_size=self.cfg.batch_size)
        self.train_one_epoch(train_dataloader)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_fold(fold=0)
