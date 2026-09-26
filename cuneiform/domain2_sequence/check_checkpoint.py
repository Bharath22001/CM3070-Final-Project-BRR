"""
Checks that a saved Domain 2 checkpoint loads and behaves sensibly. A save that
was cut off (my disk once filled mid-save) can still load, so I check that the
weights are finite, the shapes match the vocabulary, and the validation loss
matches the last logged value.

Run from the project root:
    python -m cuneiform.domain2_sequence.check_checkpoint
"""

from __future__ import annotations

import json
import math

import torch
from torch.utils.data import DataLoader
from transformers import RobertaForMaskedLM

from cuneiform import config
from cuneiform.domain2_sequence.dataset import CuneiformMLMDataset
from cuneiform.domain2_sequence.tokenizer import SignTokenizer

RUN_NAME = "domain2_distilroberta_pretrained"
CKPT_DIR = config.D2_DIR / "checkpoints" / RUN_NAME
HISTORY_JSON = config.D2_DIR / f"history_{RUN_NAME}.json"


def main() -> None:
    tok = SignTokenizer()

    print("Checkpoint")
    print("-" * 68)
    print(f"  {CKPT_DIR}")
    if not CKPT_DIR.exists():
        print("  MISSING. Nothing was saved -- training must restart.")
        return

    for f in sorted(CKPT_DIR.iterdir()):
        print(f"    {f.name:<32} {f.stat().st_size / 1e6:>8.1f} MB")

    print("\nLoading")
    print("-" * 68)
    try:
        model = RobertaForMaskedLM.from_pretrained(CKPT_DIR)
    except Exception as e:
        print(f"  FAILED TO LOAD: {type(e).__name__}: {e}")
        print("  The save was truncated. Training must restart.")
        return
    print("  Loaded without error.")

    n_bad = sum(int((~torch.isfinite(p)).sum()) for p in model.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters        : {total:,}")
    print(f"  Non-finite values : {n_bad}   <-- MUST be 0")
    print(f"  Vocab size        : {model.config.vocab_size} "
          f"(tokenizer says {tok.vocab_size})")

    if n_bad or model.config.vocab_size != tok.vocab_size:
        print("  Checkpoint is not usable.")
        return

    if HISTORY_JSON.exists():
        with open(HISTORY_JSON, "r", encoding="utf-8") as f:
            history = json.load(f)
        last = history[-1]
        best = min(history, key=lambda h: h["val_loss"])
        print(f"\n  Epochs logged     : {len(history)}")
        print(f"  Last logged epoch : {last['epoch']} (val {last['val_loss']:.4f})")
        print(f"  Best logged epoch : {best['epoch']} (val {best['val_loss']:.4f})")

    print("\nRe-measuring validation loss")
    print("-" * 68)
    val_ds = CuneiformMLMDataset("val", tok, deterministic=True)
    loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model.eval()
    total_loss, n_batches = 0.0, 0
    n_top1 = n_top5 = n_targets = 0
    with torch.no_grad():
        for ids, attn, labels in loader:
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            total_loss += out.loss.item()
            n_batches += 1
            mask = labels != -100
            if mask.any():
                top5 = out.logits.topk(5, dim=-1).indices
                gold = labels.unsqueeze(-1)
                n_top1 += int(((top5[..., :1] == gold) & mask.unsqueeze(-1)).sum())
                n_top5 += int(((top5 == gold).any(dim=-1) & mask).sum())
                n_targets += int(mask.sum())

    val_loss = total_loss / max(1, n_batches)
    print(f"  val loss   : {val_loss:.4f}")
    print(f"  perplexity : {math.exp(min(20.0, val_loss)):.1f}")
    print(f"  top1       : {100 * n_top1 / max(1, n_targets):.1f}%")
    print(f"  top5       : {100 * n_top5 / max(1, n_targets):.1f}%")

    print("\n" + "=" * 68)
    print("CHECKPOINT USABLE" if val_loss < 5.0 else
          "CHECKPOINT SUSPECT -- loss too high, likely truncated")
    print("=" * 68)


if __name__ == "__main__":
    main()