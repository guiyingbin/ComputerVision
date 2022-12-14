resnet_cfg = {
    "resnet_18": {
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["BottleNeck", 64, 1, 1, ["ReLU"], 64, 1]],
        "block3": [["BottleNeck", 64, 2, 1]] * 1,
        "block4": [["BottleNeck", 64, 1, 1, ["ReLU"], 128, 2]],
        "block5": [["BottleNeck", 128, 2, 1]] * 1,
        "block6": [["BottleNeck", 128, 1, 1, ["ReLU"], 256, 2]],
        "block7": [["BottleNeck", 256, 2, 1]] * 1,
        "block8": [["BottleNeck", 256, 1, 1, ["ReLU"], 512, 2]],
        "block9": [["BottleNeck", 512, 2, 1]] * 1
    },
    "resnet_34": {
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["BottleNeck", 64, 1, 1, ["ReLU"], 64, 1]],
        "block3": [["BottleNeck", 64, 2, 1]] * 2,
        "block4": [["BottleNeck", 64, 1, 1, ["ReLU"], 128, 2]],
        "block5": [["BottleNeck", 128, 2, 1]] * 3,
        "block6": [["BottleNeck", 128, 1, 1, ["ReLU"], 256, 2]],
        "block7": [["BottleNeck", 256, 2, 1]] * 5,
        "block8": [["BottleNeck", 256, 1, 1, ["ReLU"], 512, 2]],
        "block9": [["BottleNeck", 512, 2, 1]] * 2
    },
    "resnet_50": {
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["BottleNeck", 64, 1, 2, ["ReLU"], 64, 1]],
        "block3": [["BottleNeck", 256, 2, 2]] * 2,
        "block4": [["BottleNeck", 256, 1, 2, ["ReLU"], 128, 2]],
        "block5": [["BottleNeck", 512, 2, 2]] * 3,
        "block6": [["BottleNeck", 512, 1, 2, ["ReLU"], 256, 2]],
        "block7": [["BottleNeck", 1024, 2, 2]] * 5,
        "block8": [["BottleNeck", 1024, 1, 2, ["ReLU"], 512, 2]],
        "block9": [["BottleNeck", 2048, 2, 2]] * 2
    },
    "resnet_101": {
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["BottleNeck", 64, 1, 2, ["ReLU"], 64, 1]],
        "block3": [["BottleNeck", 256, 2, 2]] * 2,
        "block4": [["BottleNeck", 256, 1, 2, ["ReLU"], 128, 2]],
        "block5": [["BottleNeck", 512, 2, 2]] * 3,
        "block6": [["BottleNeck", 512, 1, 2, ["ReLU"], 256, 2]],
        "block7": [["BottleNeck", 1024, 2, 2]] * 22,
        "block8": [["BottleNeck", 1024, 1, 2, ["ReLU"], 512, 2]],
        "block9": [["BottleNeck", 2048, 2, 2]] * 2
    },
    "resnet_152": {
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["BottleNeck", 64, 1, 2, ["ReLU"], 64, 1]],
        "block3": [["BottleNeck", 256, 2, 2]] * 2,
        "block4": [["BottleNeck", 256, 1, 2, ["ReLU"], 128, 2]],
        "block5": [["BottleNeck", 512, 2, 2]] * 7,
        "block6": [["BottleNeck", 512, 1, 2, ["ReLU"], 256, 2]],
        "block7": [["BottleNeck", 1024, 2, 2]] * 35,
        "block8": [["BottleNeck", 1024, 1, 2, ["ReLU"], 512, 2]],
        "block9": [["BottleNeck", 2048, 2, 2]] * 2
    }
}
densenet_cfg = {
    "densenet_121": {
        "stage1": [["Conv", 3, 32, 7, 3, 2],
                   ["BatchNorm", 32],
                   ["MaxPool", 3, 1, 2]],
        "stage2": [["DenseBlock", 32, 32, 6],
                   ["Conv", 224, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage3": [["DenseBlock", 32, 32, 12],
                   ["Conv", 416, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage4": [["DenseBlock", 32, 32, 24],
                   ["Conv", 800, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage5": [["DenseBlock", 32, 32, 16]],
    },
    "densenet_169": {
        "stage1": [["Conv", 3, 32, 7, 3, 2],
                   ["BatchNorm", 32],
                   ["MaxPool", 3, 1, 2]],
        "stage2": [["DenseBlock", 32, 32, 6],
                   ["Conv", 224, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage3": [["DenseBlock", 32, 32, 12],
                   ["Conv", 416, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage4": [["DenseBlock", 32, 32, 32],
                   ["Conv", 1056, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage5": [["DenseBlock", 32, 32, 32]],
    },
    "densenet_201": {
        "stage1": [["Conv", 3, 32, 7, 3, 2],
                   ["BatchNorm", 32],
                   ["MaxPool", 3, 1, 2]],
        "stage2": [["DenseBlock", 32, 32, 6],
                   ["Conv", 224, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage3": [["DenseBlock", 32, 32, 12],
                   ["Conv", 416, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage4": [["DenseBlock", 32, 32, 48],
                   ["Conv", 1568, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage5": [["DenseBlock", 32, 32, 32]],
    },
    "densenet_264": {
        "stage1": [["Conv", 3, 32, 7, 3, 2],
                   ["BatchNorm", 32],
                   ["MaxPool", 3, 1, 2]],
        "stage2": [["DenseBlock", 32, 32, 6],
                   ["Conv", 224, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage3": [["DenseBlock", 32, 32, 12],
                   ["Conv", 416, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage4": [["DenseBlock", 32, 32, 64],
                   ["Conv", 2080, 32, 1, 0, 1],
                   ["MaxPool", 3, 1, 2]],
        "stage5": [["DenseBlock", 32, 32, 48]],
    }
}
