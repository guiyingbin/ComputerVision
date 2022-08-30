
class crnn_cfg:
    n_hidden_layer = 100
    n_class = 37
    head_config = {
        "head":
            [["BLSTM", 512, n_hidden_layer, n_hidden_layer],
             ["BLSTM", n_hidden_layer, n_hidden_layer, n_class]]
    }
    backbone_config = {
        "block1":
            [["Conv", 3, 64, 3, 1, 1],
             ["BatchNorm", 64],
             ["MaxPool", 2, 0, 2]],
        "block2":
            [["Conv", 64, 128, 3, 1, 1],
             ["BatchNorm", 128],
             ["MaxPool", 2, 0, 2]],
        "block3":
            [["Conv", 128, 256, 3, 1, 1],
             ["BatchNorm", 256],
             ["Conv", 256, 256, 3, 1, 1],
             ["BatchNorm", 256],
             ["MaxPool", (2,2), (0, 1), (2, 1)]],
        "block4":
            [["Conv", 256, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["Conv", 512, 512, 3, 1, 1],
             ["BatchNorm", 512],
             ["MaxPool", (2,2), (0,1), (2,1)]],
        "block5":
            [["Conv", 512, 512, 2, 0, 1],
             ["BatchNorm", 512]]
    }
    head_activation = ["ReLU"]
    backbone_activation = ["ReLU"]