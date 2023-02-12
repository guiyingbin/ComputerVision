import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer, PatchEmbed
from Classification.Utils.Layers import PARSeq_Encoder, DecoderLayer, PARSeq_Decoder, TokenEmbedding,PARSeq_EncoderV2
from typing import Sequence, Any, Optional
import numpy as np
import torch
from torch import Tensor
from Classification.Utils.tools import Tokenizer


"""
source:https://github.com/baudm/parseq
"""
class parseq(nn.Module):
    """
    Paper url:https://arxiv.org/pdf/2207.06966.pdf
    The PARSeq is a context-aware STR(Scene Text Recognition) model. The context-aware model is proposed to recognize image
    with occluded or incomplete characters. However, in the actual license recognition process, the covered images should
    not be solved by context reasoning, and should be rejected directly, so the current practical industrial application of this model will be limited.
    Besides, since the model involves NLP-related knowledge, it will be reproduced in the future.
    """
    def __init__(self, img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: Sequence[int], enc_mlp_ratio: int, enc_depth: Sequence[int],
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, max_label_length: int,
                 character:str, device="cpu"):
        super(parseq, self).__init__()
        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        self._device = torch.device(device)
        self.tokenizer = Tokenizer(character)
        self.bos_id = self.tokenizer.bos_id
        self.pad_id = self.tokenizer.pad_id
        self.eos_id = self.tokenizer.eos_id
        self.encoder = PARSeq_Encoder(img_size, patch_size, embed_dim=embed_dim, depths=enc_depth,
                                      num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio)
        decoder_layer = DecoderLayer(embed_dim*2, dec_num_heads, embed_dim*2 * dec_mlp_ratio, dropout)
        self.decoder = PARSeq_Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim*2))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim*2, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim*2)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim*2))
        self.dropout = nn.Dropout(p=dropout)

    def encode(self, img: torch.Tensor):
        """
        The Encoder is Swin Transformer
        :param img:
        :return:
        """
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                      tgt_query_mask=query_mask[i:j, :j])
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = ((tgt_in == self.eos_id).cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)

        return logits

class parseqV2(nn.Module):
    """
    Paper url:https://arxiv.org/pdf/2207.06966.pdf
    The PARSeq is a context-aware STR(Scene Text Recognition) model. The context-aware model is proposed to recognize image
    with occluded or incomplete characters. However, in the actual license recognition process, the covered images should
    not be solved by context reasoning, and should be rejected directly, so the current practical industrial application of this model will be limited.
    Besides, since the model involves NLP-related knowledge, it will be reproduced in the future.
    """
    def __init__(self, img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: Sequence[int], enc_mlp_ratio: int, enc_depth: Sequence[int],
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, max_label_length: int,
                 character:str, device="cpu"):
        super(parseqV2, self).__init__()
        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        self._device = torch.device(device)
        self.tokenizer = Tokenizer(character)
        self.bos_id = self.tokenizer.bos_id
        self.pad_id = self.tokenizer.pad_id
        self.eos_id = self.tokenizer.eos_id
        self.num_layer = len(dec_depth)
        self.encoder = PARSeq_EncoderV2(img_size, patch_size,
                                        embed_dim=embed_dim,
                                        depths=enc_depth,
                                        num_heads=enc_num_heads,
                                        mlp_ratio=enc_mlp_ratio)

        decoder_layer = DecoderLayer(embed_dim*2, dec_num_heads, embed_dim*2 * dec_mlp_ratio, dropout)
        self.decoder = PARSeq_Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim*2))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim*2, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim*2)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim*2))
        self.dropout = nn.Dropout(p=dropout)

    def encode(self, img: torch.Tensor):
        """
        The Encoder is Swin Transformer
        :param img:
        :return:
        """
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                      tgt_query_mask=query_mask[i:j, :j])
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = ((tgt_in == self.eos_id).cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)

        return logits
if __name__ == "__main__":
    model = parseq(img_size=(32, 256), patch_size=[2, 4],
                   embed_dim=96, enc_depth=(2, 6), enc_num_heads=(2, 6), enc_mlp_ratio=4,
                   dec_depth=1, dec_mlp_ratio=4, dec_num_heads=6,perm_num=6, perm_forward=True, perm_mirrored=True,
                   decode_ar=False,refine_iters=1, dropout=0.1, max_label_length=40, character="椎间盘买股djljjksjfsdfsfsfsfds")
    x = torch.randn((1, 3, 32, 256))
    print(model(x).shape)