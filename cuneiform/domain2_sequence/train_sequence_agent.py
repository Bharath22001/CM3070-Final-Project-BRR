"""
Fine-tunes my Domain 2 masked language model on the cuneiform corpus.

Every epoch I log val loss (for early stopping), perplexity (only meaningful
next to the baselines: unigram 105.8, bigram 84.2), top-1 and top-5 accuracy
on masked positions, and accuracy by span length, since long runs are the
real task.

The best checkpoint is saved whenever val loss improves, so Ctrl+C is safe.
With RESUME = True it continues from that checkpoint; the optimizer state isn't
saved, so expect a small loss bump on the first resumed epoch.

Run from the project root:
    python -m cuneiform.domain2_sequence.train_sequence_agent
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import RobertaForMaskedLM, get_linear_schedule_with_warmup

from cuneiform import config
from cuneiform.domain2_sequence.dataset import CuneiformMLMDataset
from cuneiform.domain2_sequence.model import build_model, describe
from cuneiform.domain2_sequence.tokenizer import SignTokenizer

# Configuration
PRETRAINED = True          # False = random-init ablation (not deployable)
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 25
PATIENCE = 4
WARMUP_FRACTION = 0.10
SEED = 1729

# Continue from the saved checkpoint. False starts fresh and overwrites the
# checkpoint folder and history file for this RUN_NAME.
RESUME = True

RUN_NAME = "domain2_distilroberta_pretrained" if PRETRAINED else "domain2_scratch"
CKPT_DIR = config.D2_DIR / "checkpoints" / RUN_NAME
HISTORY_JSON = config.D2_DIR / f"history_{RUN_NAME}.json"


def evaluate(model, loader, tok) -> dict:
    """Validation loss plus the diagnostic breakdowns."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    n_correct = n_top5 = n_targets = 0
    by_span_correct: dict[int, int] = defaultdict(int)
    by_span_total: dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for ids, attn, labels in loader:
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            total_loss += out.loss.item()
            n_batches += 1

            mask = labels != -100
            if not mask.any():
                continue

            logits = out.logits
            top5 = logits.topk(5, dim=-1).indices
            gold = labels.unsqueeze(-1)

            correct1 = (top5[..., :1] == gold) & mask.unsqueeze(-1)
            correct5 = (top5 == gold).any(dim=-1) & mask

            n_correct += int(correct1.sum())
            n_top5 += int(correct5.sum())
            n_targets += int(mask.sum())

            # Credit each masked position to the length of the run it sits in.
            for row in range(labels.shape[0]):
                row_mask = mask[row].tolist()
                row_ok = correct5[row].tolist()
                i = 0
                while i < len(row_mask):
                    if not row_mask[i]:
                        i += 1
                        continue
                    j = i
                    while j < len(row_mask) and row_mask[j]:
                        j += 1
                    span_len = j - i
                    bucket = span_len if span_len <= 4 else (8 if span_len <= 8 else 99)
                    for k in range(i, j):
                        by_span_total[bucket] += 1
                        if row_ok[k]:
                            by_span_correct[bucket] += 1
                    i = j

    val_loss = total_loss / max(1, n_batches)
    return {
        "val_loss": val_loss,
        "perplexity": math.exp(min(20.0, val_loss)),
        "top1": n_correct / max(1, n_targets),
        "top5": n_top5 / max(1, n_targets),
        "n_targets": n_targets,
        "by_span": {
            str(k): by_span_correct[k] / max(1, by_span_total[k])
            for k in sorted(by_span_total)
        },
    }


def load_or_build_model(tok: SignTokenizer):
    """
    Return (model, history, start_epoch, best_loss_seed). On a resume I read the
    history too, so early stopping counts from the best loss of the whole run.
    """
    if RESUME and CKPT_DIR.exists():
        model = RobertaForMaskedLM.from_pretrained(CKPT_DIR)
        print(f"  Resumed from {CKPT_DIR}")

        if HISTORY_JSON.exists():
            with open(HISTORY_JSON, "r", encoding="utf-8") as f:
                history = json.load(f)
            start_epoch = history[-1]["epoch"] + 1
            best_loss_seed = min(h["val_loss"] for h in history)
            print(f"  Continuing from epoch {start_epoch} "
                  f"(best so far {best_loss_seed:.4f})")
            print("  NOTE: optimizer state was not saved, so AdamW moments "
                  "restart cold.")
            print("  A small loss bump in the first resumed epoch is expected.")
            return model, history, start_epoch, best_loss_seed

        print("  No history file found; treating this as epoch 1.")
        return model, [], 1, float("inf")

    model = build_model(tok, pretrained=PRETRAINED)
    return model, [], 1, float("inf")


def main() -> None:
    torch.manual_seed(SEED)
    config.ensure_dirs()
    config.check_environment()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tok = SignTokenizer()
    print(f"Tokenizer: {tok}\n")

    print("Datasets")
    print("-" * 68)
    train_ds = CuneiformMLMDataset("train", tok, seed=SEED)
    val_ds = CuneiformMLMDataset("val", tok, seed=SEED, deterministic=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("\nModel")
    print("-" * 68)
    model, history, start_epoch, best_loss_seed = load_or_build_model(tok)
    describe(model, RUN_NAME)

    if start_epoch > MAX_EPOCHS:
        print(f"\nAlready at epoch {start_epoch - 1} of MAX_EPOCHS={MAX_EPOCHS}. "
              f"Raise MAX_EPOCHS to continue.")
        return

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # The schedule only covers the remaining epochs, so the learning-rate decay
    # restarts on a resume.
    remaining_epochs = MAX_EPOCHS - start_epoch + 1
    total_steps = len(train_loader) * remaining_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(total_steps * WARMUP_FRACTION),
        num_training_steps=total_steps,
    )

    print(f"\nTraining: epochs {start_epoch}..{MAX_EPOCHS}, patience {PATIENCE}, "
          f"{len(train_loader)} steps/epoch")
    print("Interrupting with Ctrl+C is safe -- the best checkpoint is already saved.")
    print("=" * 68)

    best_loss = best_loss_seed
    epochs_without_improvement = 0

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        t0 = time.time()

        for step, (ids, attn, labels) in enumerate(train_loader, start=1):
            optimiser.zero_grad()
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += loss.item()

            if step % 50 == 0:
                print(f"  epoch {epoch:>2} step {step:>4}/{len(train_loader)} "
                      f"| train loss {running / step:.4f}", flush=True)

        train_loss = running / max(1, len(train_loader))
        metrics = evaluate(model, val_loader, tok)
        mins = (time.time() - t0) / 60

        spans = "  ".join(
            f"{'5-8' if k == '8' else '9+' if k == '99' else k}:{100 * v:.0f}%"
            for k, v in metrics["by_span"].items()
        )
        print(f"\nEPOCH {epoch:>2} ({mins:.1f} min) "
              f"train {train_loss:.4f} | val {metrics['val_loss']:.4f} | "
              f"ppl {metrics['perplexity']:.1f} | "
              f"top1 {100 * metrics['top1']:.1f}% | top5 {100 * metrics['top5']:.1f}%")
        print(f"           top5 by span length -> {spans}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "minutes": round(mins, 2),
            **{k: v for k, v in metrics.items()},
        })
        with open(HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            epochs_without_improvement = 0
            try:
                model.save_pretrained(CKPT_DIR)
                print(f"           saved new best -> {CKPT_DIR}")
            except OSError as e:
                # A failed save can leave a truncated file, so I say so loudly.
                print(f"           SAVE FAILED: {e}")
                print("           Free disk space. The previous checkpoint may "
                      "now be truncated -- run check_checkpoint.py.")
        else:
            epochs_without_improvement += 1
            print(f"           no improvement ({epochs_without_improvement}/{PATIENCE})")
            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print("\n" + "=" * 68)
    print("TRAINING COMPLETE")
    print(f"  best val loss : {best_loss:.4f}  (perplexity "
          f"{math.exp(min(20.0, best_loss)):.1f} over {tok.vocab_size} tokens)")
    print(f"  baselines     : unigram 105.8 | bigram 84.2 | uniform 288.0")
    print(f"  checkpoint    : {CKPT_DIR}")
    print(f"  history       : {HISTORY_JSON}")
    print("=" * 68)


if __name__ == "__main__":
    main()