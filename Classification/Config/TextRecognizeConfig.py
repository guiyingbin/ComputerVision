
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

class trba_cfg():
    densenet_cfg = {
        "densenet_121": {
            "stage1": [["Conv", 3, 32, 7, 3, 2],
                       ["BatchNorm", 32],
                       ["MaxPool", 3, 1, (2, 2)]],
            "stage2": [["DenseBlock", 32, 32, 6],
                       ["Conv", 224, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage3": [["DenseBlock", 32, 32, 12],
                       ["Conv", 416, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage4": [["DenseBlock", 32, 32, 24],
                       ["Conv", 800, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage5": [["DenseBlock", 32, 32, 16]],
        },
        "densenet_169": {
            "stage1": [["Conv", 3, 32, 7, 3, 2],
                       ["BatchNorm", 32],
                       ["MaxPool", 3, 1, (2, 2)]],
            "stage2": [["DenseBlock", 32, 32, 6],
                       ["Conv", 224, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage3": [["DenseBlock", 32, 32, 12],
                       ["Conv", 416, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage4": [["DenseBlock", 32, 32, 32],
                       ["Conv", 1056, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage5": [["DenseBlock", 32, 32, 32]],
        },
        "densenet_201": {
            "stage1": [["Conv", 3, 32, 7, 3, 2],
                       ["BatchNorm", 32],
                       ["MaxPool", 3, 1, 2]],
            "stage2": [["DenseBlock", 32, 32, 6],
                       ["Conv", 224, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage3": [["DenseBlock", 32, 32, 12],
                       ["Conv", 416, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage4": [["DenseBlock", 32, 32, 48],
                       ["Conv", 1568, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage5": [["DenseBlock", 32, 32, 32]],
        },
        "densenet_264": {
            "stage1": [["Conv", 3, 32, 7, 3, 2],
                       ["BatchNorm", 32],
                       ["MaxPool", 3, 1, 2]],
            "stage2": [["DenseBlock", 32, 32, 6],
                       ["Conv", 224, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage3": [["DenseBlock", 32, 32, 12],
                       ["Conv", 416, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage4": [["DenseBlock", 32, 32, 64],
                       ["Conv", 2080, 32, 1, 0, 1],
                       ["MaxPool", 3, 1, (2, 1)]],
            "stage5": [["DenseBlock", 32, 32, 48]],
        }
    }
    character_path = r"E:\Dataset\OCR_Dataset\char_std_5990.txt"
    max_length = 64
    input_size = (60, 256)
    model_name = "densenet_121"
    hidden_size = 256
    seq_input_size = 544 * 2
    n_class = 5990
    n_fiducial = 20
    input_channels = 3