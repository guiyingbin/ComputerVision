import torch
import numpy as np
import random

class baseConverter():
    def __init__(self):
        pass

    def encode(self, text):
        pass

    def decode(self, text_label):
        pass


class baseTrainer():
    def __init__(self):
        pass

    def train_one_epoch(self, dataloader, is_train):
        pass

    def train_fold(self, fold):
        pass

    def predict(self):
        pass

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True

