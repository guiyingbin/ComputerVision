class base_cfg:
    n_anchors = 5
    n_class = 5
    img_size = 448
    output_channel = n_anchors * (n_class + 5)
    backbone_type = "darknet_53"
    head_config = {}
    neck_config = {}
    activation_list=["LeakyReLU", 0.2]

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
        "C4":
            [["Darknet_block", 1024, 1024]],
        "C4_up":
            [["Conv", 1024, 256, 1, 0, 1],
             ["BatchNorm", 256],
             ["UpNearest", 2]],
        "C5":
            [["Darknet_block", 768, 256]],
        "C5_up":
            [["Conv", 256, 128, 1, 0, 1],
             ["BatchNorm", 128],
             ["UpNearest", 2]],
        "C6":
            [["Darknet_block", 384, 128]]
    }
    head_config = {
        "C4":
            [["Conv", 1024, 2048, 3, 1, 1],
             ["BatchNorm", 2048],
             ["Conv", 2048, output_channel, 1, 0, 1]],
        "C5":
            [["Conv", 256, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["Conv", 512, output_channel, 1, 0, 1]],
        "C6":
            [["Conv", 128, 256, 3, 1, 1],
             ["BatchNorm", 256],
             ["Conv", 256, output_channel, 1, 0, 1]]
    }