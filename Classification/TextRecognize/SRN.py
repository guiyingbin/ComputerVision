import torch.nn as nn
from Utils.BaseClass import baseTextRecognizer
from Classification.ResNet import resnet
from Classification.Utils.Layers import build_block, TransformerEncoder, SRNDecoder
from Classification.Config.TextRecognizeConfig import srn_cfg


class srn(baseTextRecognizer):
    def __init__(self, cfg=srn_cfg):
        """
        The Semantic reasoning network, the
        :param cfg:
        """
        super(srn, self).__init__(cfg)

    def build_transformation(self, transformation_name):
        if transformation_name is not None:
            # TODO 加入STN
            pass
        return nn.Identity()

    def build_feature_extractor(self, feature_extractor_config):

        feature_extractor = {"backbone": None,
                             "neck": []}

        # 载入 backbone
        if feature_extractor_config["backbone"].startswith("resnet"):
            feature_extractor["backbone"] = resnet(model_type=feature_extractor_config["backbone"])

        for name, neck_config in feature_extractor_config["neck"].items():
            if name.startswith("FPN"):
                neck_block_list = [["FPN", neck_config, self.cfg.channels, self.cfg.sequence_config["d_model"]]]
                feature_extractor["neck"].append(build_block(block_list=neck_block_list,
                                                             activation_list=self.cfg.neck_activation_list))

        return feature_extractor

    def build_sequence_modeling(self, sequence_config):
        sequence_model = None
        if sequence_config["name"] == "TransformerEncoder":
            sequence_model = TransformerEncoder(sequence_config["n_block"],
                                                sequence_config["n_head"],
                                                sequence_config["d_k"],
                                                sequence_config["d_v"],
                                                sequence_config["d_model"],
                                                sequence_config["n_position"],)
        return sequence_model

    def build_prediction_layer(self, predict_config):
        predict_model = None
        if predict_config["name"] == "SRN":
            predict_model = SRNDecoder(d_model=predict_config["d_model"],
                                       n_max_len=predict_config["n_max_len"],
                                       n_position=predict_config["n_position"],
                                       n_class=predict_config["n_class"],
                                       n_block=predict_config["n_block"])
        return predict_model

    def forward(self, imgs):
        imgs = self.transformation(imgs)
        latent_feature = self.feature_extractor["backbone"](imgs)
        if isinstance(latent_feature, tuple):
            for neck_layers in self.feature_extractor["neck"]:
                latent_feature = neck_layers(latent_feature)
            latent_feature = latent_feature[1]
        else:
            latent_feature = self.feature_extractor["neck"](latent_feature)

        b, c, h, w = latent_feature.shape
        latent_feature = latent_feature.reshape(b, c, -1).transpose(1, 2).contiguous()
        latent_feature = self.sequence_modeling(latent_feature)

        g_t, s_t, f_t = self.prediction(latent_feature)
        return g_t, s_t, f_t


if __name__ == "__main__":
    import torch
    SRN = srn()
    imgs = torch.rand((1,3, 64, 256))
    g_t, s_t, f_t = SRN(imgs)
    print(g_t.shape)
