vgg_cfg = {
    "vgg_16": {
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["Conv", 64, 128, 3, 1, 1],
                   ["BatchNorm", 128],
                   ["Conv", 128, 128, 3, 1, 1],
                   ["BatchNorm", 128],
                   ["MaxPool", 3, 1, 2]],
        "block3": [["Conv", 128, 256, 3, 1, 1],
                   ["BatchNorm", 256],
                   ["Conv", 256, 256, 3, 1, 1],
                   ["BatchNorm", 256],
                   ["Conv", 256, 256, 3, 1, 1],
                   ["BatchNorm", 256],
                   ["MaxPool", 3, 1, 2]],
        "block4": [["Conv", 256, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["MaxPool", 3, 1, 2]],
        "block5": [["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["MaxPool", 3, 1, 2]]},
    "vgg_19": {
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["Conv", 64, 128, 3, 1, 1],
                   ["BatchNorm", 128],
                   ["Conv", 128, 128, 3, 1, 1],
                   ["BatchNorm", 128],
                   ["MaxPool", 3, 1, 2]],
        "block3": [["Conv", 128, 256, 3, 1, 1],
                   ["BatchNorm", 256],
                   ["Conv", 256, 256, 3, 1, 1],
                   ["BatchNorm", 256],
                   ["Conv", 256, 256, 3, 1, 1],
                   ["BatchNorm", 256],
                   ["Conv", 256, 256, 3, 1, 1],
                   ["BatchNorm", 256],
                   ["MaxPool", 3, 1, 2]],
        "block4": [["Conv", 256, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["MaxPool", 3, 1, 2]],
        "block5": [["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["Conv", 512, 512, 3, 1, 1],
                   ["BatchNorm", 512],
                   ["MaxPool", 3, 1, 2]]},
}

repvgg_cfg = {
    "repvgg_a0": {
        "stage1": [["RepBlock", 3, 48, 2, False]],
        "stage2": [["RepBlock", 48, 48, 1, True],
                   ["RepBlock", 48, 48, 2, False]],
        "stage3": [["RepBlock", 48, 96, 1, False]] + [["RepBlock", 96, 96, 1, True]] * 2 +
                  [["RepBlock", 96, 96, 2, False]],
        "stage4": [["RepBlock", 96, 192, 1, False]] + [["RepBlock", 192, 192, 1, True]] * 12 +
                  [["RepBlock", 192, 192, 2, False]],
        "stage5": [["RepBlock", 192, 1280, 2, False]]
    },
    "repvgg_a1": {
        "stage1": [["RepBlock", 3, 64, 2, False]],
        "stage2": [["RepBlock", 64, 64, 1, True],
                   ["RepBlock", 64, 64, 2, False]],
        "stage3": [["RepBlock", 64, 128, 1, False]] + [["RepBlock", 128, 128, 1, True]] * 2 +
                  [["RepBlock", 128, 128, 2, False]],
        "stage4": [["RepBlock", 128, 256, 1, False]] + [["RepBlock", 256, 256, 1, True]] * 12 +
                  [["RepBlock", 256, 256, 2, False]],
        "stage5": [["RepBlock", 256, 1280, 2, False]]
    },
    "repvgg_a2": {
        "stage1": [["RepBlock", 3, 96, 2, False]],
        "stage2": [["RepBlock", 96, 96, 1, True],
                   ["RepBlock", 96, 96, 2, False]],
        "stage3": [["RepBlock", 96, 192, 1, False]] + [["RepBlock", 192, 192, 1, True]] * 2 +
                  [["RepBlock", 192, 192, 2, False]],
        "stage4": [["RepBlock", 192, 384, 1, False]] + [["RepBlock", 384, 384, 1, True]] * 12 +
                  [["RepBlock", 384, 384, 2, False]],
        "stage5": [["RepBlock", 384, 1408, 2, False]]
    },
    "repvgg_b0": {
        "stage1": [["RepBlock", 3, 64, 2, False]],
        "stage2": [["RepBlock", 64, 64, 1, True]]*3+[["RepBlock", 64, 64, 2, False]],
        "stage3": [["RepBlock", 64, 128, 1, False]] + [["RepBlock", 128, 128, 1, True]] * 4 +
                  [["RepBlock", 128, 128, 2, False]],
        "stage4": [["RepBlock", 128, 256, 1, False]] + [["RepBlock", 256, 256, 1, True]] * 14 +
                  [["RepBlock", 256, 256, 2, False]],
        "stage5": [["RepBlock", 256, 1280, 2, False]]
    },

}
