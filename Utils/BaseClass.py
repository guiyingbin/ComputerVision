import torch
import numpy as np
import random
import torch.nn as nn

class base_cfg:
    n_anchors = 5
    n_class = 5
    img_size = 448
    output_channel = n_anchors * (n_class + 5)
    backbone_type = "darknet_53"
    head_config = {}
    neck_config = {}
    activation_list = ["LeakyReLU", 0.2]


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
        torch.backends.cudnn.deterministic = True

class baseTextRecognizer(torch.nn.Module):
    def __init__(self, cfg):
        super(baseTextRecognizer, self).__init__()
        self.cfg = cfg
        self.transformation = self.build_transformation(transformation_name=self.cfg.transformation_name)
        self.feature_extractor = self.build_feature_extractor(feature_extractor_config=self.cfg.feature_extractor_config)
        self.sequence_modeling = self.build_sequence_modeling(sequence_config=self.cfg.sequence_config)
        self.prediction = self.build_prediction_layer(predict_config=self.cfg.predict_config)

    def build_transformation(self, transformation_name):
        pass

    def build_feature_extractor(self, feature_extractor_config):
        pass

    def build_sequence_modeling(self, sequence_config):
        pass

    def build_prediction_layer(self, predict_config):
        pass
