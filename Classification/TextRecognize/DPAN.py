from Classification.TextRecognize.SRN import srn
from Classification.Utils.Layers import DPANDecoder
from Classification.Config.TextRecognizeConfig import dpan_cfg

class dpan(srn):
    def __init__(self, cfg=dpan_cfg):
        """
        I think DPAN has very little innovation and it feels like the network is basically the same as SRN
        :param cfg:
        """
        super(dpan, self).__init__(cfg)

    def build_prediction_layer(self, predict_config):
        predict_model = None
        if predict_config["name"] == "DPAN":
            predict_model = DPANDecoder(d_model=predict_config["d_model"],
                                        n_max_len=predict_config["n_max_len"],
                                        n_position=predict_config["n_position"],
                                        n_class=predict_config["n_class"],
                                        n_block=predict_config["n_block"])
        return predict_model

if __name__ == "__main__":
    import torch
    DPAN = dpan()
    imgs = torch.rand((1,3, 64, 256))
    g_t, s_t, f_t = DPAN(imgs)
    print(g_t.shape)