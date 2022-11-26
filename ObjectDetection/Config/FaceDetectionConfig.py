from Utils.BaseClass import base_cfg


class centerface_cfg(base_cfg):
    n_class = 4
    n_key_point = 5
    n_box_param = 4
    backbone_type = "mobilenetv3_small_050"
    channels = [16, 24, 288]
    backbone_activation_list = ["ReLU"]
    neck_activation_list = ["ReLU"]
    head_activation_list = ["ReLU"]
    neck_config = {
        "FPN_3":{
            "P4":
                [["Conv", channels[-1], 256, 1, 0, 1]],
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
        "class_head":
            [["Conv", 768, 768, 3, 1, 1],
             ["BatchNorm", 768],
             ["Conv", 768, n_class, 1, 0, 1]],
        "bbox_head":
            [["Conv", 768, 768, 3, 1, 1],
             ["BatchNorm", 768],
             ["Conv", 768, n_box_param, 1, 0, 1]],
        "keypoint_head":
            [["Conv", 768, 768, 3, 1, 1],
             ["BatchNorm", 768],
             ["Conv", 768, n_key_point, 1, 0, 1]],
    }