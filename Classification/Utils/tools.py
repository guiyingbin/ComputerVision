
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
"""
Source: https://github.com/chenjun2hao/SRN.pytorch
"""
class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return batch_text, torch.IntTensor(length)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

"""
source:https://github.com/baudm/parseq
"""
class BaseTokenizer(ABC):

    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs

"""
source:https://github.com/baudm/parseq
"""
class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                 for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids

if __name__ == "__main__":
    file_path = r"E:\Dataset\OCR_Dataset\char_std_5990.txt"
    with open(file_path, "r") as f:
        content = f.readlines()
    character = "".join([c.strip() for c in content])
    converter = AttnLabelConverter(character)

    text = ["中国第二个省份", "ABDCDF1334342"]
    encode_text = converter.encode(text, batch_max_length=64)
    print(encode_text)
    # ncode_text = torch.stack(encode_text, dim=0)
    # print(encode_text.shape)
    # e