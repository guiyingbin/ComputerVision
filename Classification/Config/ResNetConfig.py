resnet_cfg = {
    "resnet_18":{
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["BottleNeck", 64, 1, 1, ["ReLU"], 64, 1]],
        "block3": [["BottleNeck", 64, 2, 1]]*1,
        "block4": [["BottleNeck", 64, 1, 1, ["ReLU"], 128, 2]],
        "block5": [["BottleNeck", 128, 2, 1]]*1,
        "block6": [["BottleNeck", 128, 1, 1, ["ReLU"], 256, 2]],
        "block7": [["BottleNeck", 256, 2, 1]]*1,
        "block8": [["BottleNeck", 256, 1, 1, ["ReLU"], 512, 2]],
        "block9": [["BottleNeck", 512, 2, 1]]*1
    },
    "resnet_34":{
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["BottleNeck", 64, 1, 1, ["ReLU"], 64, 1]],
        "block3": [["BottleNeck", 64, 2, 1]]*2,
        "block4": [["BottleNeck", 64, 1, 1, ["ReLU"], 128, 2]],
        "block5": [["BottleNeck", 128, 2, 1]]*3,
        "block6": [["BottleNeck", 128, 1, 1, ["ReLU"], 256, 2]],
        "block7": [["BottleNeck", 256, 2, 1]]*5,
        "block8": [["BottleNeck", 256, 1, 1, ["ReLU"], 512, 2]],
        "block9": [["BottleNeck", 512, 2, 1]]*2
    },
    "resnet_50":{
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["BottleNeck", 64, 1, 2, ["ReLU"], 64, 1]],
        "block3": [["BottleNeck", 256, 2, 2]]*2,
        "block4": [["BottleNeck", 256, 1, 2, ["ReLU"], 128, 2]],
        "block5": [["BottleNeck", 512, 2, 2]]*3,
        "block6": [["BottleNeck", 512, 1, 2, ["ReLU"], 256, 2]],
        "block7": [["BottleNeck", 1024, 2, 2]]*5,
        "block8": [["BottleNeck", 1024, 1, 2, ["ReLU"], 512, 2]],
        "block9": [["BottleNeck", 2048, 2, 2]]*2
    },
    "resnet_101":{
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["BottleNeck", 64, 1, 2, ["ReLU"], 64, 1]],
        "block3": [["BottleNeck", 256, 2, 2]]*2,
        "block4": [["BottleNeck", 256, 1, 2, ["ReLU"], 128, 2]],
        "block5": [["BottleNeck", 512, 2, 2]]*3,
        "block6": [["BottleNeck", 512, 1, 2, ["ReLU"], 256, 2]],
        "block7": [["BottleNeck", 1024, 2, 2]]*22,
        "block8": [["BottleNeck", 1024, 1, 2, ["ReLU"], 512, 2]],
        "block9": [["BottleNeck", 2048, 2, 2]]*2
    },
    "resnet_152":{
        "block1": [["Conv", 3, 64, 3, 1, 2],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["Conv", 64, 64, 3, 1, 1],
                   ["BatchNorm", 64],
                   ["MaxPool", 3, 1, 2]],
        "block2": [["BottleNeck", 64, 1, 2, ["ReLU"], 64, 1]],
        "block3": [["BottleNeck", 256, 2, 2]]*2,
        "block4": [["BottleNeck", 256, 1, 2, ["ReLU"], 128, 2]],
        "block5": [["BottleNeck", 512, 2, 2]]*7,
        "block6": [["BottleNeck", 512, 1, 2, ["ReLU"], 256, 2]],
        "block7": [["BottleNeck", 1024, 2, 2]]*35,
        "block8": [["BottleNeck", 1024, 1, 2, ["ReLU"], 512, 2]],
        "block9": [["BottleNeck", 2048, 2, 2]]*2
    }
}