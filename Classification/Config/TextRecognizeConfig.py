
class srn_cfg():
    transformation_name = None
    backbone_activation_list = ["ReLU"]
    neck_activation_list = ["ReLU"]
    head_activation_list = ["ReLU"]
    channels = [256, 512, 1024, 2048]
    feature_extractor_config = {
        "backbone": "resnet_50",
        "neck": {
            "FPN1": {
                "P5":
                    [["Conv", channels[3], 512, 1, 0, 1]],
                "P5_up":
                    [["UpNearest", 2]],
                "P4":
                    [["Conv", 512, 512, 1, 0, 1]],
                "P4_up":
                    [["UpNearest", 2]],
                "P3":
                    [["Conv", 512, 512, 1, 0, 1]],
                "P3_up":
                    [["UpNearest", 2]],
                "P2":
                    [["Conv", 512, 512, 1, 0, 1]]
            }
        }
    }
    sequence_config = {
        "name":"TransformerEncoder",
        "n_head":8,
        "d_k":64,
        "d_v":64,
        "d_model":512,
        "n_position":1024,
        "n_block":2
    }
    predict_config = {
        "name":"SRN",
        "n_class":37,
        "n_head": 8,
        "d_k": 64,
        "d_v": 64,
        "d_model": 512,
        "n_position": 1024,
        "n_block": 4,
        "n_max_len": 25
    }


class dpan_cfg(srn_cfg):
    predict_config = {
        "name": "DPAN",
        "n_class": 37,
        "n_head": 8,
        "d_k": 64,
        "d_v": 64,
        "d_model": 512,
        "n_position": 1024,
        "n_block": 4,
        "n_max_len": 25
    }