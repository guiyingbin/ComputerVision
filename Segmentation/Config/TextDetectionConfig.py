"""
Configuration of text detection model
@author: guiyingbin
@time: 2022/09/08
"""

from Utils.BaseClass import base_cfg

class psenet_cfg(base_cfg):
    pretrained = False
    n_anchors = 5
    n_class = 5
    output_channel = n_anchors*(n_class+5)
    backbone_type = "resnet_50"
    channels = [64, 128, 256, 512] if backbone_type in ["resnet_18", "resnet_34"] else [256, 512, 1024, 2048]
    backbone_activation_list = ["ReLU"]
    neck_activation_list = ["ReLU"]
    head_activation_list = ["ReLU"]
    neck_config = {
        "FPN":{
            "P5":
                [["Conv", channels[3], 256, 1, 0, 1]],
            "P5_up":
                [["UpNearest", 2]],
            "P4":
                [["Conv", 256, 256, 1, 0, 1]],
            "P4_up":
                [["UpNearest", 2]],
            "P3":
                [["Conv", 256, 256, 1, 0, 1]],
            "P3_up":
                [["UpNearest", 2]],
            "P2":
                [["Conv", 256, 256, 1, 0, 1]]
        }

    }
    head_config = {
        "head":
            [["Conv", 1024, 1024, 3, 1, 1],
             ["BatchNorm", 1024],
             ["Conv", 1024, n_class, 1, 0, 1]]
    }


class dbnet_cfg(psenet_cfg):
    n_class = 1
    k=50
    head_config = {
        "prob_head":
            [["Conv", 1024, 1024, 3, 1, 1],
             ["BatchNorm", 1024],
             ["ConvTranpose", 1024, 1024, 2, 2],
             ["BatchNorm", 1024],
             ["ConvTranpose", 1024, n_class, 2, 2],
             ["BatchNorm", n_class]],
        "threshold_head":
            [["Conv", 1024, 1024, 3, 1, 1],
             ["BatchNorm", 1024],
             ["ConvTranpose", 1024, 1024, 2, 2],
             ["BatchNorm", 1024],
             ["ConvTranpose", 1024, n_class, 2, 2],
             ["BatchNorm", n_class]]
    }


class dbnet_plusplus_cfg(dbnet_cfg):
    n_class = 1
    head_config = {
        "pre_conv":
            [["ASF", 4, 256]],
        "prob_head":
            [["Conv", 1024, 1024, 3, 1, 1],
             ["BatchNorm", 1024],
             ["ConvTranpose", 1024, 1024, 2, 2],
             ["BatchNorm", 1024],
             ["ConvTranpose", 1024, n_class, 2, 2],
             ["BatchNorm", n_class]],
        "threshold_head":
            [["Conv", 1024, 1024, 3, 1, 1],
             ["BatchNorm", 1024],
             ["ConvTranpose", 1024, 1024, 2, 2],
             ["BatchNorm", 1024],
             ["ConvTranpose", 1024, n_class, 2, 2],
             ["BatchNorm", n_class]]
    }


class pannet_cfg(psenet_cfg):
    backbone_type = "resnet_18"
    channels = [64, 128, 256, 512] if backbone_type in ["resnet_18", "resnet_34"] else [256, 512, 1024, 2048]
    n_Fr = 5
    n_class = 1
    neck_config = {
        "FPEM_Block1":[["FPEM", channels, True]]+[["FPEM",channels, False]]*(n_Fr-1),
        "FFM_Block": [["FFM"]]
    }
    head_config = {
        "text_region_head":
            [["Conv", 512, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["ConvTranpose", 512, 512, 2, 2],
             ["BatchNorm", 512],
             ["ConvTranpose", 512, n_class, 2, 2],
             ["BatchNorm", n_class]],
        "kernel_head":
            [["Conv", 512, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["ConvTranpose", 512, 512, 2, 2],
             ["BatchNorm", 512],
             ["ConvTranpose", 512, n_class, 2, 2],
             ["BatchNorm", n_class]],
        "similarity_head":
            [["Conv", 512, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["ConvTranpose", 512, 512, 2, 2],
             ["BatchNorm", 512],
             ["ConvTranpose", 512, n_class, 2, 2],
             ["BatchNorm", n_class]]
    }