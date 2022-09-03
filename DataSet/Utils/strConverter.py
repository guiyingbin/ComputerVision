from ObjectDetection.Config.OcrConfig import crnn_model_cfg
from Utils.BaseClass import baseConverter


class textConverter(baseConverter):
    def __init__(self, cfg):
        super().__init__()
        self.alphabet_dict = {cfg.alphabet[i]:i for i in range(len(cfg.alphabet))}
        self.alphabet = cfg.alphabet

    def encode(self, text):
        text_label = [self.alphabet_dict[each] for each in text]
        return text_label

    def decode(self, text_label):
        text = [self.alphabet[each] for each in text_label]
        return text


if __name__ == "__main__":
    convert = textConverter(crnn_model_cfg)
    text = "132433这是我的电话"
    print(convert.encode(text))
    label = [1,3443, 3323, 323, 543, 2334, 4332, 4342, 4333, 4323]
    print(convert.decode(label))