from typing import Any, Dict, List

import sentencepiece as sp
from torchtext.data.utils import get_tokenizer
import torch
import pickle as pk
from transformers import T5Tokenizer
class SentencePieceBPETokenizer:
    r"""
    A tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`_
    with BPE sub-routine. It encodes caption strings into list of tokens.

    Args:
        model_path: Path to the ``.model`` file trained by SentencePiece.
    """
    SP_SPACE = u"▁"

    def __init__(self, model_path: str,sos_id:int,eos_id:int):
        self.model_path = model_path

        # Load pretrained tokenizer model.
        self.tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base")

    def encode(self, text: str) -> List[int]:
        r"""Convert a text string to a list of integer token ids."""
        tok = self.tokenizer(text.strip().lower(), return_tensors='pt',padding='max_length',
            max_length=64,pad_to_max_length=True,
            return_attention_mask=False, truncation=True,
            add_special_tokens=True) # mã hóa caption vd "day là chuổi" => ["day","là","chuỗi"]
        return tok['input_ids'].squeeze()

    def decode(self, token_ids: List[int]) -> str:
        r"""Convert a sequence of token IDs to a text string."""
        return self.tokenizer.decode(token_ids)
