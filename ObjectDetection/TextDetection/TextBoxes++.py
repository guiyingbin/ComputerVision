import torch.nn as nn


class textBoxes_plus(nn.Module):
    def __init__(self):
        super(textBoxes_plus, self).__init__()
    """
    The paper url:https://arxiv.org/pdf/1801.02765.pdf
    The architecture of TextBoxes++ is a little different from EAST in backbone, and the detail can be found in paper.
    """