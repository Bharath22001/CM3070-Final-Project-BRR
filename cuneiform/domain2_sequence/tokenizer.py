"""
Maps cuneiform sign ids to model token ids and back. My input is already a
sequence of integer sign ids, so a plain dict lookup does the job and I don't
need a HuggingFace tokenizer.

Vocabulary (288 tokens, from build_vocab.py):
    0 <pad>   padding, ignored by the loss
    1 <s>     start of sequence
    2 </s>    end of sequence
    3 <unk>   a real sign too rare for its own token
    4 <mask>  a position to predict
    5 <brk>   a sign the annotators marked as broken (train_label 0)
    6..287    sign tokens, most frequent first

<brk> is input context only and is never predicted, since nobody knows what
the broken sign was. <mask> marks a position to predict: synthetic during
training, and missed or low-confidence detections at inference.
"""

from __future__ import annotations

import json

from cuneiform import config

VOCAB_JSON = config.D2_DIR / "vocab.json"


class SignTokenizer:
    def __init__(self, vocab_path=VOCAB_JSON):
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"{vocab_path} not found. Run build_vocab.py first."
            )
        with open(vocab_path, "r", encoding="utf-8") as f:
            v = json.load(f)

        self.specials: dict[str, int] = v["specials"]
        self.sign_to_id: dict[int, int] = {int(k): i for k, i in v["sign_to_id"].items()}
        self.id_to_sign: dict[int, int] = {int(k): s for k, s in v["id_to_sign"].items()}
        self.vocab_size: int = v["vocab_size"]
        self.vision_emit_classes: set[int] = set(v["vision_emit_classes"])

        self.pad_id = self.specials["<pad>"]
        self.bos_id = self.specials["<s>"]
        self.eos_id = self.specials["</s>"]
        self.unk_id = self.specials["<unk>"]
        self.mask_id = self.specials["<mask>"]
        self.brk_id = self.specials["<brk>"]

        # Positions that can never be prediction targets: specials and <brk>.
        self.non_target_ids = {
            self.pad_id, self.bos_id, self.eos_id, self.mask_id, self.brk_id,
        }

    def encode_sign(self, sign: int) -> int:
        """One train_label to one token id."""
        if sign == config.BROKEN_LABEL:
            return self.brk_id
        return self.sign_to_id.get(sign, self.unk_id)

    def encode_line(self, tokens: list[int]) -> list[int]:
        return [self.encode_sign(s) for s in tokens]

    def decode_id(self, token_id: int) -> str:
        """Token id to a readable label, for diagnostics."""
        for name, i in self.specials.items():
            if i == token_id:
                return name
        sign = self.id_to_sign.get(token_id)
        return f"sign_{sign}" if sign is not None else f"<invalid:{token_id}>"

    def decode(self, ids: list[int]) -> list[str]:
        return [self.decode_id(i) for i in ids]

    def is_predictable(self, token_id: int) -> bool:
        """True if this position can be a masking target."""
        return token_id not in self.non_target_ids

    def __repr__(self) -> str:
        return (f"SignTokenizer(vocab_size={self.vocab_size}, "
                f"signs={len(self.sign_to_id)}, "
                f"vision_classes={len(self.vision_emit_classes)})")