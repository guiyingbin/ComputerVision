darknet_cfg = {
    "darknet_19": {
                "block1":
                    [["Conv", 3, 32, 3, 1, 1],
                     ["BatchNorm", 32],
                     ["MaxPool", 2, 0, 2]],
                "block2":
                    [["Conv", 32, 64, 3, 1, 1],
                     ["BatchNorm", 64],
                     ["MaxPool", 2, 0, 2]],
                "block3":
                    [["Conv", 64, 128, 3, 1, 1],
                     ["BatchNorm", 128],
                     ["Conv", 128, 64, 1, 0, 1],
                     ["BatchNorm", 64],
                     ["Conv", 64, 128, 3, 1, 1],
                     ["BatchNorm", 128],
                     ["MaxPool", 2, 0, 2]],
                "block4":
                    [["Conv", 128, 256, 3, 1, 1],
                     ["BatchNorm", 256],
                     ["Conv", 256, 128, 1, 0, 1],
                     ["BatchNorm", 128],
                     ["Conv", 128, 256, 3, 1, 1],
                     ["BatchNorm", 256],
                     ["MaxPool", 2, 0, 2]],
                "block5":
                    [["Conv", 256, 512, 3, 1, 1],
                     ["BatchNorm", 512],
                     ["Conv", 512, 256, 1, 0, 1],
                     ["BatchNorm", 256],
                     ["Conv", 256, 512, 3, 1, 1],
                     ["BatchNorm", 512],
                     ["Conv", 512, 256, 1, 0, 1],
                     ["BatchNorm", 256],
                     ["Conv", 256, 512, 3, 1, 1],
                     ["BatchNorm", 512],
                     ["MaxPool", 2, 0, 2]],
                "block6":
                    [["Conv", 512, 1024, 3, 1, 1],
                     ["BatchNorm", 1024],
                     ["Conv", 1024, 512, 1, 0, 1],
                     ["BatchNorm", 512],
                     ["Conv", 512, 1024, 3, 1, 1],
                     ["BatchNorm", 1024],
                     ["Conv", 1024, 512, 1, 0, 1],
                     ["BatchNorm", 512],
                     ["Conv", 512, 1024, 3, 1, 1],
                     ["BatchNorm", 1024]]},
    "darknet_53": {
            "block1":
                [["Conv", 3, 32, 3, 1, 1],
                 ["BatchNorm", 32],
                 ["Conv", 32, 64, 3, 1, 2]],
            "block2":
                [["DarkNet_block", 64, 1]],
            "block3":
                [["Conv", 64, 128, 3, 1, 2],
                 ["BatchNorm", 128]],
            "block4":
                [["DarkNet_block", 128, 2]],
            "block5":
                [["Conv", 128, 256, 3, 1, 2],
                 ["BatchNorm", 256]],
            "block6":
                [["DarkNet_block", 256, 8]],
            "block7":
                [["Conv", 256, 512, 3, 1, 2],
                 ["BatchNorm", 512]],
            "block8":
                [["DarkNet_block", 512, 8]],
            "block9":
                [["Conv", 512, 1024, 3, 1, 2],
                 ["BatchNorm", 1024]],
            "block10":
                [["DarkNet_block", 1024, 4]]
            },

    "cspdarknet_53": {
            "block1":
                [["Conv", 3, 32, 3, 1, 1],
                 ["BatchNorm", 32],
                 ["Conv", 32, 64, 3, 1, 2]],
            "block2":
                [["CSPDarkNet_block", 64, 64, 1, "fusion_last"]],
            "block3":
                [["Conv", 64, 128, 3, 1, 2],
                 ["BatchNorm", 128]],
            "block4":
                [["CSPDarkNet_block", 128, 128, 2, "fusion_last"]],
            "block5":
                [["Conv", 128, 256, 3, 1, 2],
                 ["BatchNorm", 256]],
            "block6":
                [["CSPDarkNet_block", 256, 256, 8, "fusion_last"]],
            "block7":
                [["Conv", 256, 512, 3, 1, 2],
                 ["BatchNorm", 512]],
            "block8":
                [["CSPDarkNet_block", 512, 512, 8, "fusion_last"]],
            "block9":
                [["Conv", 512, 1024, 3, 1, 2],
                 ["BatchNorm", 1024]],
            "block10":
                [["CSPDarkNet_block", 1024, 1024, 4, "fusion_last"]]
            },
    "cspnet_yolo5s": {
            "block1":
                [["Focus", 3, 32],
                 ["BatchNorm", 32],
                 ["Conv", 32, 64, 3, 1, 2]],
            "block2":
                [["CSP1_block", 64, 64, 1]],
            "block3":
                [["Conv", 64, 128, 3, 1, 2],
                 ["BatchNorm", 128]],
            "block4":
                [["CSP1_block", 128, 128, 3]],
            "block5":
                [["Conv", 128, 256, 3, 1, 2],
                 ["BatchNorm", 256]],
            "block6":
                [["CSP1_block", 256, 256, 3]],
            "block7":
                [["Conv", 256, 512, 3, 1, 2],
                 ["BatchNorm", 512]],
            "block8":
                [["SPP", 512]]
            },
}
