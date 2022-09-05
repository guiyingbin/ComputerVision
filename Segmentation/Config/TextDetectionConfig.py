from Utils.BaseClass import base_cfg

class psenet_cfg(base_cfg):
    n_anchors = 5
    n_class = 5
    output_channel = n_anchors*(n_class+5)
    backbone_type = "resnet_50"
    channels = [64, 128, 256, 512] if backbone_type in ["resnet_18", "resnet_34"] else [256, 512, 1024, 2048]
    backbone_activation_list = ["ReLU"]
    neck_activation_list = ["ReLU"]
    head_activation_list = ["ReLU"]
    neck_config = {
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
            [["Conv", 256, 256, 1, 0, 1]],
    }
    head_config = {
        "head":
            [["Conv", 1024, 1024, 3, 1, 1],
             ["BatchNorm", 1024],
             ["Conv", 1024, n_class, 1, 0, 1]]
    }