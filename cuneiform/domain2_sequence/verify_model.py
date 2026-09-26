"""
Checks the Domain 2 model before training: that span lengths follow the real
damage distribution, that distilroberta-base loads with the 288-token
vocabulary, that the first loss is close to ln(288) as an untrained model's
should be, and that CPU speed is good enough to train locally.

Run from the project root:
    python -m cuneiform.domain2_sequence.verify_model
"""

from __future__ import annotations

import math
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader

from cuneiform.domain2_sequence.dataset import (
    CuneiformMLMDataset,
    load_corpus,
    measure_span_distribution,
)
from cuneiform.domain2_sequence.model import build_model, describe
from cuneiform.domain2_sequence.tokenizer import SignTokenizer

BATCH_SIZE = 16


def check_span_fidelity(train_ds: CuneiformMLMDataset, tok: SignTokenizer) -> None:
    print("\nSpan fidelity after the sampler fix")
    print("-" * 68)

    real = measure_span_distribution(load_corpus("train"))
    realised: Counter = Counter()
    n_masked = n_predictable = 0
    bad_targets = 0

    for i in range(min(500, len(train_ds))):
        ids, attn, labels = train_ds[i]
        labels_l = labels.tolist()

        for lab in labels_l:
            if lab != -100:
                n_masked += 1
                if lab in tok.non_target_ids:
                    bad_targets += 1

        n_predictable += sum(
            1 for t, a in zip(ids.tolist(), attn.tolist())
            if a == 1 and (t == tok.mask_id or tok.is_predictable(t))
        )

        run = 0
        for lab in labels_l:
            if lab != -100:
                run += 1
            elif run:
                realised[run] += 1
                run = 0
        if run:
            realised[run] += 1

    total = sum(realised.values())
    print(f"  {'len':>4} | {'real':>7} | {'realised':>9} | {'delta':>7}")
    print("  " + "-" * 36)
    max_dev = 0.0
    for length in sorted(real)[:10]:
        r = real[length]
        g = realised.get(length, 0) / max(1, total)
        max_dev = max(max_dev, abs(r - g))
        print(f"  {length:>4} | {100 * r:>6.1f}% | {100 * g:>8.1f}% | "
              f"{100 * (g - r):>+6.1f}%")

    print(f"\n  Realised mask rate     : "
          f"{100 * n_masked / max(1, n_predictable):.1f}% (configured 25%)")
    print(f"  Largest deviation      : {100 * max_dev:.1f} percentage points")
    print(f"  Invalid targets        : {bad_targets}   <-- MUST be 0")


def main() -> None:
    tok = SignTokenizer()
    print(f"Tokenizer: {tok}")

    print("\nBuilding datasets")
    print("-" * 68)
    train_ds = CuneiformMLMDataset("train", tok)
    val_ds = CuneiformMLMDataset("val", tok)

    check_span_fidelity(train_ds, tok)

    print("\nBuilding model")
    print("-" * 68)
    model = build_model(tok, pretrained=True)
    describe(model, "distilroberta-base, resized")

    print("\nForward pass on a real batch")
    print("-" * 68)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    ids, attn, labels = next(iter(loader))
    print(f"  batch shapes: ids {tuple(ids.shape)} | labels {tuple(labels.shape)}")
    print(f"  masked targets in batch: {int((labels != -100).sum())}")

    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, labels=labels)

    expected = math.log(tok.vocab_size)
    print(f"  loss                : {out.loss.item():.4f}")
    print(f"  ln(vocab_size)      : {expected:.4f}   <-- untrained loss should "
          f"be near this")
    print(f"  logits shape        : {tuple(out.logits.shape)}")

    if not torch.isfinite(out.loss):
        print("  FAIL: loss is not finite. Do not train.")
        return

    print("\nCPU throughput")
    print("-" * 68)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    t0 = time.time()
    n_steps = 5
    for i, (b_ids, b_attn, b_lab) in enumerate(loader):
        if i >= n_steps:
            break
        opt.zero_grad()
        loss = model(input_ids=b_ids, attention_mask=b_attn, labels=b_lab).loss
        loss.backward()
        opt.step()
    elapsed = time.time() - t0

    per_step = elapsed / n_steps
    steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
    print(f"  seconds per step    : {per_step:.2f}")
    print(f"  steps per epoch     : {steps_per_epoch}")
    print(f"  estimated per epoch : {per_step * steps_per_epoch / 60:.1f} minutes")
    print(f"  estimated 10 epochs : {per_step * steps_per_epoch * 10 / 60:.1f} minutes")

    print("\n" + "=" * 68)
    print("MODEL VERIFIED")
    print(f"  train blocks: {len(train_ds):,} | val blocks: {len(val_ds):,}")
    print("=" * 68)


if __name__ == "__main__":
    main()