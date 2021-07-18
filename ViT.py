import torch
from torch import nn

class Patch_Embedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3,
                 embedding_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = (img_size//patch_size)**2
        self.linear_projection = nn.Conv2d(in_channel, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        B, C, H, W = img.shape
        assert H==W and C==3
        img = self.linear_projection(img)
        x = img.flatten(start_dim=2).transpose(1,2)
        return x

class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12, qkv_bias=False, qk_scale=None, atten_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim//self.num_heads
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(dim,dim)
        self.scale = qk_scale or head_dim**(-0.5)

    def forward(self, x):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qk = q @ k.transpose(-2, -1) * self.scale
        qk = qk.softmax(dim=-1)
        qk = self.atten_drop(qk)
        atten = (qk @ v).transpose(1, 2).reshape(B, N, C)
        atten = self.proj(atten)
        atten = self.proj_drop(atten)
        return atten

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act_layer = act_layer()
        self.drop = nn.Dropout(drop)
    def forward(self, x):

        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Encoder_Block(nn.Module):
    def __init__(self, embed_dim=768, norm_layer=nn.LayerNorm, num_heads=12, qkv_bias=False, qk_scale=None,
                             atten_drop=0, proj_drop=0, factor=4, mlp_drop=0, act_layer=nn.GELU):
        super().__init__()
        self.MSA = Attention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             atten_drop=atten_drop, proj_drop=proj_drop)
        self.MLP = MLP(in_features=embed_dim, out_features=embed_dim, hidden_features=embed_dim*factor,
                       act_layer=act_layer, drop=mlp_drop)
        self.norm_layer = norm_layer(embed_dim)
    def forward(self, x):
        x = self.norm_layer(x)
        x = x + self.MSA(x)
        x = self.norm_layer(x)
        x = x + self.MLP(x)
        return x

class Transformer_Encoder(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.encoder = nn.Sequential(*[
            Encoder_Block()
        for _ in range(depth)])

    def forward(self, x):
        return self.encoder(x)

class Vision_Transformer(nn.Module):
    def __init__(self,embed_dim=768, in_head=768, out_head=5, drop=0., depth=12):
        super().__init__()
        self.patch_embed = Patch_Embedding()
        self.transformer_encoder = Transformer_Encoder(depth)
        self.head = nn.Linear(in_head, out_head)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_token = 1
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+self.num_token, embed_dim))
        self.pos_dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token,x), dim=1)
        x = x+self.pos_embed
        x = self.pos_dropout(x)
        x = self.transformer_encoder(x)[:, 0]
        x = self.head(x)
        return x

if __name__ == '__main__':
    x = torch.randn((2, 3, 224, 224))
    vit = Vision_Transformer()
    x = vit(x)
    print(x)
    print(x.shape)

