"""
Builds my Domain 2 masked language model from pre-trained distilroberta-base
(6 layers, 768 hidden, 12 heads).

My vocabulary is 288 sign tokens instead of 50,265 English wordpieces, so I
replace the word embeddings and keep the pre-trained transformer blocks, which
carry the general machinery for passing information between positions.

resize_token_embeddings() only truncates the embedding matrix, so each sign
would inherit a random English word's vector (my first loss was 16.71 against
ln(288) = 5.66). So after resizing I re-initialise the word embeddings, zero the
pad row and the LM head bias, and re-tie the head. Position embeddings stay
pre-trained.
"""

from __future__ import annotations

import torch
from transformers import RobertaForMaskedLM

from cuneiform.domain2_sequence.dataset import MAX_LEN
from cuneiform.domain2_sequence.tokenizer import SignTokenizer

BASE_CHECKPOINT = "distilroberta-base"


def _reinit_vocabulary_weights(model, tokenizer: SignTokenizer) -> None:
    """Re-initialise everything vocabulary-specific; the transformer body and positions stay pre-trained."""
    std = model.config.initializer_range   # 0.02 for RoBERTa

    emb = model.get_input_embeddings()
    with torch.no_grad():
        emb.weight.normal_(mean=0.0, std=std)
        emb.weight[tokenizer.pad_id].zero_()

    # The head bias still holds English token frequencies, so I zero it.
    head_bias = getattr(getattr(model, "lm_head", None), "bias", None)
    if head_bias is not None:
        with torch.no_grad():
            head_bias.zero_()

    # Re-tie so the output projection follows the new embeddings.
    model.tie_weights()


def build_model(tokenizer: SignTokenizer, pretrained: bool = True):
    """
    Return a RobertaForMaskedLM sized to my sign vocabulary.

    pretrained=True uses distilroberta-base weights with a fresh vocabulary.
    pretrained=False is fully random and only for the ablation, never deployed.
    """
    if pretrained:
        model = RobertaForMaskedLM.from_pretrained(BASE_CHECKPOINT)
        n_before = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(tokenizer.vocab_size)
        _reinit_vocabulary_weights(model, tokenizer)
        print(f"  Loaded {BASE_CHECKPOINT} | vocabulary {n_before:,} -> "
              f"{tokenizer.vocab_size} (embeddings reinitialised, body kept)")
    else:
        base = RobertaForMaskedLM.from_pretrained(BASE_CHECKPOINT)
        config = base.config
        del base
        config.vocab_size = tokenizer.vocab_size
        model = RobertaForMaskedLM(config)
        _reinit_vocabulary_weights(model, tokenizer)
        print(f"  Built {BASE_CHECKPOINT} architecture from RANDOM INIT "
              f"(ablation arm, vocab {tokenizer.vocab_size})")

    # My special-token ids differ from RoBERTa's, and RoBERTa derives position
    # ids from padding_idx, so I fix pad in both the config and the embedding.
    model.config.vocab_size = tokenizer.vocab_size
    model.config.pad_token_id = tokenizer.pad_id
    model.config.bos_token_id = tokenizer.bos_id
    model.config.eos_token_id = tokenizer.eos_id
    model.config.mask_token_id = tokenizer.mask_id

    model.get_input_embeddings().padding_idx = tokenizer.pad_id
    model.roberta.embeddings.padding_idx = tokenizer.pad_id

    return model


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    return {"total": total, "embedding": emb, "transformer": total - emb}


def describe(model, label: str = "") -> None:
    p = count_parameters(model)
    c = model.config
    print(f"\n  {label}")
    print(f"    layers x hidden x heads : {c.num_hidden_layers} x "
          f"{c.hidden_size} x {c.num_attention_heads}")
    print(f"    vocab size              : {c.vocab_size}")
    print(f"    max positions           : {c.max_position_embeddings} "
          f"(we use {MAX_LEN})")
    print(f"    embedding params        : {p['embedding']:>12,}")
    print(f"    transformer params      : {p['transformer']:>12,}")
    print(f"    TOTAL params            : {p['total']:>12,}")