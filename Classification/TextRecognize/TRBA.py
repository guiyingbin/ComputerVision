import timm
from torchsummary import summary
import torch.nn as nn
from Classification.Utils.Layers import TPS_SpatialTransformerNetwork, BidirectionalLSTM, Attention
from Classification.Config.TextRecognizeConfig import trba_cfg
from Classification.ResNet.DenseNet import denseNet
from Classification.Utils.tools import AttnLabelConverter


class TRBA(nn.Module):
    def __init__(self, cfg=trba_cfg):
        super(TRBA, self).__init__()
        self.cfg = cfg
        with open(self.cfg.character_path, "r") as f:
            content = f.readlines()
        character = "".join([c.strip() for c in content])
        self.label_converter = AttnLabelConverter(character)
        self.cfg.n_class = len(self.label_converter.character)

        self.transformation = self.build_transorformation(n_fiducial=self.cfg.n_fiducial,
                                                          I_size=self.cfg.input_size,
                                                          R_size=self.cfg.input_size,
                                                          I_channels=self.cfg.input_channels)

        self.feat = self.build_feature_extraction(model_name=self.cfg.model_name)

        self.seq = self.build_sequence_model(input_size=self.cfg.seq_input_size,
                                             hidden_size=self.cfg.hidden_size)

        self.pred = self.build_prediction(input_size=self.cfg.hidden_size,
                                          hidden_size=self.cfg.hidden_size,
                                          num_classes=self.cfg.n_class,
                                          batch_max_length=self.cfg.max_length)



    def build_transorformation(self, n_fiducial, I_size, R_size, I_channels=3):
        return TPS_SpatialTransformerNetwork(F=n_fiducial, I_size=I_size, I_r_size=R_size, I_channel_num=I_channels)

    def build_feature_extraction(self, model_name="densenet_121"):
        model = denseNet(cfg=self.cfg.densenet_cfg, model_type=model_name)
        return model

    def build_sequence_model(self, input_size, hidden_size):
        SequenceModeling = nn.Sequential(
            BidirectionalLSTM(input_size, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        return SequenceModeling

    def build_prediction(self, input_size, hidden_size, num_classes, batch_max_length):
        return Attention(input_size, hidden_size, num_classes, batch_max_length=batch_max_length)

    def forward(self, batch_imgs, text=None):
        if not self.training:
            with torch.no_grad():
                batch_imgs = self.transformation(batch_imgs)
                latent = self.feat(batch_imgs)[-1]
                B, C, H, W = latent.shape
                latent = latent.permute(0, 3, 1, 2).reshape(B, W, -1)
                latent = self.seq(latent)
                output = self.pred(latent)
            return output
        else:
            encode_text, _ = self.label_converter.encode(text, batch_max_length=self.cfg.max_length)
            batch_imgs = self.transformation(batch_imgs)
            latent = self.feat(batch_imgs)[-1]
            B, C, H, W = latent.shape
            latent = latent.permute(0, 3, 1, 2).reshape(B, W, -1)
            latent = self.seq(latent)
            output = self.pred(latent, text=encode_text)
            return output

if __name__ == "__main__":
    import torch
    model = TRBA()
    #model = denseNet(cfg=trba_cfg.densenet_cfg, model_type="densenet_121")
    print(model)
    batch_img = torch.rand((2, 3, 60, 256))
    model.train()
    print(model(batch_img, text=["中国第二省份","meiliguojibie"]).shape)
    #print(summary(TPS, input_size=(3, 60, 256), batch_size=1, device="cpu"))
    #print(summary(model, input_size=(3, 60, 256), batch_size=1, device="cpu"))