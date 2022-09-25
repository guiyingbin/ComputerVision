from Utils.BaseClass import base_cfg


class east_cfg(base_cfg):
    backbone_type = "vgg_16"
    backbone_activation_list = ["ReLU"]
    neck_activation_list = ["ReLU"]
    head_activation_list = ["ReLU"]
    neck_config = {
        "C5":
            [[None]],
        "C5_up":
            [["UpNearest", 2]],
        "C4":
            [["Conv", 1024, 256, 1, 0, 1],
             ["Conv", 256, 256, 3, 1, 1]],
        "C4_up":
            [["UpNearest", 2]],
        "C3":
            [["Conv", 512, 128, 1, 0, 1],
             ["Conv", 128, 128, 3, 1, 1]],
        "C3_up":
            [["UpNearest", 2]],
        "C2":
            [["Conv", 256, 64, 1, 0, 1],
             ["Conv", 64, 64, 3, 1, 1],
             ["Conv", 64, 64, 3, 1, 1]]
    }
    head_config = {
        "score_head": [["Conv", 64, 1, 1, 0, 1]],
        "boxes_head": [["Conv", 64, 4, 1, 0, 1]],
        "angle_head": [["Conv", 64, 1, 1, 0, 1]],
        "quad_head": [["Conv", 64, 8, 1, 0, 1]],
    }

