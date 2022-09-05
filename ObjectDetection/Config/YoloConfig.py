from Utils.BaseClass import base_cfg

class yolov2_cfg(base_cfg):
    n_anchors = 5
    n_class = 5
    output_channel = n_anchors * (n_class + 5)
    backbone_type = "darknet_19"
    head_config = {
        "head1":
            [["Conv", 1024, output_channel, 1, 0, 1]]
        }
    activation_list = ["LeakyReLU", 0.2]

class yolov3_cfg(base_cfg):
    n_anchors = 5
    n_class = 5
    output_channel = n_anchors*(n_class+5)
    backbone_type = "darknet_53"
    neck_config = {
        "C6":
            [["ConvSet_block", 1024, 1024, 1]],
        "C6_up":
            [["Conv", 1024, 256, 1, 0, 1],
             ["BatchNorm", 256],
             ["UpNearest", 2]],
        "C5":
            [["ConvSet_block", 768, 256, 1]],
        "C5_up":
            [["Conv", 256, 128, 1, 0, 1],
             ["BatchNorm", 128],
             ["UpNearest", 2]],
        "C4":
            [["ConvSet_block", 384, 128, 1]]
    }
    head_config = {
        "C6":
            [["Conv", 1024, 2048, 3, 1, 1],
             ["BatchNorm", 2048],
             ["Conv", 2048, output_channel, 1, 0, 1]],
        "C5":
            [["Conv", 256, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["Conv", 512, output_channel, 1, 0, 1]],
        "C4":
            [["Conv", 128, 256, 3, 1, 1],
             ["BatchNorm", 256],
             ["Conv", 256, output_channel, 1, 0, 1]]
    }


class yolov4_cfg(yolov3_cfg):
    backbone_type = "cspdarknet_53"
    neck_config = {
        "C6":
            [["SPP", 1024],
             ["ConvSet_block", 1024, 1024, 1]],
        "C6_up":
            [["Conv", 1024, 256, 1, 0, 1],
             ["BatchNorm", 256],
             ["UpNearest", 2]],
        "C5":
            [["ConvSet_block", 768, 256, 1]],
        "C5_up":
            [["Conv", 256, 128, 1, 0, 1],
             ["BatchNorm", 128],
             ["UpNearest", 2]],
        "C4":
            [["ConvSet_block", 384, 128, 1]],
        "PAN":
            [["PAN", [128, 256, 1024], "Conv"]]
    }
    activation_list = ["Mish"]

class yolov5s_cfg(yolov4_cfg):
    backbone_type = "cspnet_yolo5s"
    neck_config = {
        "C6":
            [["CSP2_block", 512, 512, 1],
             ["Conv", 512, 1024, 3, 1, 1],
             ["BatchNorm", 1024]],
        "C6_up":
            [["Conv", 1024, 256, 1, 0, 1],
             ["BatchNorm", 256],
             ["UpNearest", 2]],
        "C5":
            [["CSP2_block", 512, 512, 1],
             ["Conv", 512, 256, 1, 0, 1]],
        "C5_up":
            [["Conv", 256, 128, 1, 0, 1],
             ["BatchNorm", 128],
             ["UpNearest", 2]],
        "C4":
            [["CSP2_block", 256, 128, 1]],
        "PAN":
            [["PAN", [128, 256, 1024], "CSP2_block"]]
    }


class yolov7_cfg(base_cfg):
    n_anchors = 5
    n_class = 5
    output_channel = n_anchors * (n_class + 5)
    backbone_type = "elannet_yolo7"
    backbone_activation_list = ["Silu"]
    neck_activation_list = ["Silu"]
    head_activation_list = ["Silu"]
    neck_config = {
        "C6":
            [["SPP_CSP_Conv", 1024]],
        "C6_up":
            [["Conv", 512, 256, 1, 0, 1],
             ["BatchNorm", 256],
             ["UpNearest", 2]],
        "C5":
            [["ELAN", 512, "W"]],
        "C5_up":
            [["Conv", 256, 128, 1, 0, 1],
             ["BatchNorm", 128],
             ["UpNearest", 2]],
        "C4":
            [["ELAN", 256, "W"]],
        "PAN":
            [["PAN", [128, 256, 1024], "ELAN_MP"]]
    }
    head_config = {
        "C6":
            [["RepConv", 512],
             ["Conv", 512, output_channel, 1, 0, 1],
             ["BatchNorm", output_channel]],
        "C5":
            [["RepConv", 256],
             ["Conv", 256, output_channel, 1, 0, 1],
             ["BatchNorm", output_channel]],
        "C4":
            [["RepConv", 128],
             ["Conv", 128, output_channel, 1, 0, 1],
             ["BatchNorm", output_channel]]
    }