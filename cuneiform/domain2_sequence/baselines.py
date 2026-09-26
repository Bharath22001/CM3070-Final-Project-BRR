"""
Computes simple baselines for my Domain 2 model, so its perplexity means
something. The sign distribution is very skewed (24 signs cover half the
tokens), so a model that only learned which signs are common could still look
decent.

Unigram uses training frequencies with no context. Bigram uses the sign to
the left, backing off to unigram for unseen pairs. Both are scored on exactly
the masked positions the MLM was scored on. If the MLM is well below bigram it
is really using context on both sides; near unigram it is just a frequency table.

Run from the project root:
    python -m cuneiform.domain2_sequence.baselines
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cuneiform import config
from cuneiform.domain2_sequence.dataset import CuneiformMLMDataset, load_corpus
from cuneiform.domain2_sequence.tokenizer import SignTokenizer

SMOOTHING = 0.1          # add-k smoothing keeps log-probabilities finite
BACKOFF_WEIGHT = 0.4     # weight on the unigram term when a bigram is unseen


def build_ngram_tables(tok: SignTokenizer):
    """Count unigrams and bigrams over the train split only."""
    records = load_corpus("train")

    unigram = Counter()
    bigram = defaultdict(Counter)

    for rec in records:
        ids = tok.encode_line(rec["tokens"])
        # <brk> has no ground truth, so it stays out of the statistics.
        clean = [i for i in ids if i != tok.brk_id]
        for i, tid in enumerate(clean):
            unigram[tid] += 1
            if i > 0:
                bigram[clean[i - 1]][tid] += 1

    return unigram, bigram


def evaluate_baselines(tok: SignTokenizer, unigram, bigram, loader) -> dict:
    """
    Score both baselines on the same masked positions the MLM sees. The bigram
    only looks left, so inside a long masked span it falls back to unigram,
    which is exactly what a bidirectional model should do better at.
    """
    V = tok.vocab_size
    uni_total = sum(unigram.values())

    uni_logp_sum = big_logp_sum = 0.0
    uni_correct = big_correct = 0
    n = 0

    uni_ranked = [s for s, _ in unigram.most_common()]
    uni_top1 = uni_ranked[0] if uni_ranked else tok.unk_id

    for ids, attn, labels in loader:
        ids_l = ids.tolist()
        labels_l = labels.tolist()

        for row in range(len(labels_l)):
            for pos, gold in enumerate(labels_l[row]):
                if gold == -100:
                    continue
                n += 1

                p_uni = ((unigram.get(gold, 0) + SMOOTHING) /
                         (uni_total + SMOOTHING * V))
                uni_logp_sum += math.log(p_uni)
                if gold == uni_top1:
                    uni_correct += 1

                prev = ids_l[row][pos - 1] if pos > 0 else tok.bos_id
                ctx = bigram.get(prev)

                if ctx and prev != tok.mask_id:
                    ctx_total = sum(ctx.values())
                    p_big = ((ctx.get(gold, 0) + SMOOTHING) /
                             (ctx_total + SMOOTHING * V))
                    p_big = (1 - BACKOFF_WEIGHT) * p_big + BACKOFF_WEIGHT * p_uni
                    best = max(ctx.items(), key=lambda kv: kv[1])[0]
                else:
                    p_big = p_uni
                    best = uni_top1

                big_logp_sum += math.log(p_big)
                if gold == best:
                    big_correct += 1

    return {
        "n": n,
        "unigram_loss": -uni_logp_sum / max(1, n),
        "unigram_top1": uni_correct / max(1, n),
        "bigram_loss": -big_logp_sum / max(1, n),
        "bigram_top1": big_correct / max(1, n),
    }


BASELINES_JSON = "baselines.json"


def write_artifact(tok: SignTokenizer, r: dict, mlm: dict | None) -> Path:
    """
    Save the baseline panel to JSON so the report table can be rebuilt from it.
    Keys are sorted so two runs give identical bytes.
    """
    uniform_loss = math.log(tok.vocab_size)
    payload = {
        "population": (
            f"{r['n']} masked positions in the val split, the same positions "
            "the MLM was evaluated on. N-gram tables counted over the train "
            "split only. <brk> excluded from both, since it is damage with no "
            "ground truth and is never an MLM prediction target."
        ),
        "settings": {
            "smoothing": SMOOTHING,
            "backoff_weight": BACKOFF_WEIGHT,
            "vocab_size": tok.vocab_size,
        },
        "n_scored": r["n"],
        "rows": [
            {"model": "uniform", "what": "no knowledge at all",
             "loss": uniform_loss, "perplexity": math.exp(uniform_loss),
             "top1": 1.0 / tok.vocab_size},
            {"model": "unigram", "what": "sign frequency, zero context",
             "loss": r["unigram_loss"],
             "perplexity": math.exp(r["unigram_loss"]),
             "top1": r["unigram_top1"]},
            {"model": "bigram", "what": "previous sign only, backing off",
             "loss": r["bigram_loss"],
             "perplexity": math.exp(r["bigram_loss"]),
             "top1": r["bigram_top1"]},
        ],
    }

    if mlm is not None:
        payload["rows"].append({
            "model": "MLM (deployed)",
            "what": f"distilroberta-base, bidirectional, epoch {mlm['epoch']}",
            "loss": mlm["val_loss"],
            "perplexity": mlm["perplexity"],
            "top1": mlm["top1"],
        })
        payload["mlm_source"] = (
            "artifacts/domain2/history_domain2_distilroberta_pretrained.json, "
            f"best epoch by val_loss (epoch {mlm['epoch']} of 25)"
        )
        payload["margins"] = {
            "vs_unigram": {
                "perplexity_reduction":
                    1 - mlm["perplexity"] / math.exp(r["unigram_loss"]),
                "top1_ratio": mlm["top1"] / max(1e-9, r["unigram_top1"]),
            },
            "vs_bigram": {
                "perplexity_reduction":
                    1 - mlm["perplexity"] / math.exp(r["bigram_loss"]),
                "top1_ratio": mlm["top1"] / max(1e-9, r["bigram_top1"]),
            },
        }

    path = Path(config.D2_DIR) / BASELINES_JSON
    with open(path, "w", encoding="utf-8", newline="") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def main() -> None:
    config.check_environment()
    tok = SignTokenizer()

    print("Building n-gram tables from the train split")
    print("-" * 68)
    unigram, bigram = build_ngram_tables(tok)
    print(f"  unigram types      : {len(unigram)}")
    print(f"  bigram contexts    : {len(bigram)}")
    print(f"  tokens counted     : {sum(unigram.values()):,}")

    print("\nScoring on the validation masked positions")
    print("-" * 68)
    val_ds = CuneiformMLMDataset("val", tok, deterministic=True)
    loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    r = evaluate_baselines(tok, unigram, bigram, loader)

    uniform_loss = math.log(tok.vocab_size)

    print(f"\n  Masked positions scored: {r['n']:,}\n")
    print(f"  {'model':<24} {'loss':>8} {'perplexity':>12} {'top1':>8}")
    print("  " + "-" * 56)
    print(f"  {'uniform (no knowledge)':<24} {uniform_loss:>8.4f} "
          f"{math.exp(uniform_loss):>12.1f} {100 / tok.vocab_size:>7.1f}%")
    print(f"  {'unigram (frequency)':<24} {r['unigram_loss']:>8.4f} "
          f"{math.exp(r['unigram_loss']):>12.1f} "
          f"{100 * r['unigram_top1']:>7.1f}%")
    print(f"  {'bigram (local context)':<24} {r['bigram_loss']:>8.4f} "
          f"{math.exp(r['bigram_loss']):>12.1f} "
          f"{100 * r['bigram_top1']:>7.1f}%")
    # Read the MLM's numbers from its training history so they never go stale.
    history_path = config.D2_DIR / "history_domain2_distilroberta_pretrained.json"
    best = None
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        best = min(history, key=lambda h: h["val_loss"])
        label = f"MLM (epoch {best['epoch']})"
        print(f"  {label:<24} {best['val_loss']:>8.4f} "
              f"{best['perplexity']:>12.1f} {100 * best['top1']:>7.1f}%")

        print("\n  Margin over baselines:")
        print(f"    vs unigram : {100 * (1 - best['perplexity'] / math.exp(r['unigram_loss'])):.0f}% "
              f"lower perplexity, {best['top1'] / max(1e-9, r['unigram_top1']):.1f}x top1")
        print(f"    vs bigram  : {100 * (1 - best['perplexity'] / math.exp(r['bigram_loss'])):.0f}% "
              f"lower perplexity, {best['top1'] / max(1e-9, r['bigram_top1']):.1f}x top1")
    else:
        print(f"  {'MLM':<24} {'(no history file found)':>32}")

    print("\n" + "=" * 68)
    print("INTERPRETATION")
    print("  The MLM number is only meaningful as a MARGIN over these baselines.")
    print("  Compare it to bigram: that is the bar for 'more than local")
    print("  co-occurrence'. Beating unigram alone is not evidence of grammar.")
    print("=" * 68)

    path = write_artifact(tok, r, best)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()