from Classification.Utils.SVTR_Layers import *
"""
Source:
    1.https://github.com/1079863482/paddle2torch_PPOCRv3/blob/main/rec/RecSVTR.py
    2.https://github.com/j-river/svtr-pytorch/blob/44c419c3365066d5d78d3ccbfc5ccc5e8c7b2633/svtr_model.py#L582
"""
class SVTR(nn.Module):
    def __init__(self, img_size, in_channels=3,
                 embed_dim=None,
                 depth=None,
                 num_heads=None,
                 out_channels=192,
                 max_length=50,
                 vab_size=6624):
        super(SVTR, self).__init__()
        if num_heads is None:
            num_heads = [2, 4, 8]
        if depth is None:
            depth = [3, 6, 3]
        if embed_dim is None:
            embed_dim = [64, 128, 256]
        self.backbone = SVTRNet(img_size=img_size,
                                in_channels=in_channels,
                                embed_dim=embed_dim,
                                out_channels=out_channels,
                                depth=depth,
                                num_heads=num_heads,
                                out_char_num=max_length)
        self.neck = Im2Seq()
        self.head = CTCHead(in_channels=out_channels,out_channels=vab_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x, _ = self.head(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary
    a = torch.rand(2, 3, 48, 512)
    svtr = SVTR(img_size=(48, 512))
    # print(summary(svtr, input_size=(3, 48, 512), batch_size=2, device="cpu"))
    out = svtr(a)
    print(out.size())