"""
Checks that my Domain 2 checkpoint behaves at inference the way it did in
training, and calibrates the posterior floor below which a restoration is shown
as [x] instead of a name.

A: val with the training protocol, which should land near 22.2% top-1; if not,
   the checkpoint or loading is wrong.
B: val, leave-one-out (one mask, clean context), which should be at least A.
C: test, leave-one-out, the regime fusion works in.
D: calibration: accuracy against the LM's own posterior on the test split. The
   floor is where the cumulative accuracy below it drops to the unigram
   baseline. I print the full curve (and pick nothing if it is flat), measure
   the posterior raw and renormalised over signs, and run it both clean and
   damaged; the live floor comes from the damaged run.

Arms A to C use the same random draws as before; --blocks only affects arm D.

Run from the project root:
    python -m cuneiform.orchestrator.check_lm_inference
    python -m cuneiform.orchestrator.check_lm_inference --only-calibration
    python -m cuneiform.orchestrator.check_lm_inference --blocks 60 --per-block 25
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader

from cuneiform.domain2_sequence.dataset import (
    CuneiformMLMDataset, build_blocks, load_corpus,
)
from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.orchestrator.fusion import load_sequence_model, support_ids

N_BLOCKS = 40          # blocks sampled per split for the leave-one-out arms
MAX_POS_PER_BLOCK = 20  # positions scored per block
SEED = 1729

# Arm D defaults, kept separate so changing them never moves arm C.
CAL_BLOCKS = 120
CAL_PER_BLOCK = 40

# Mask rate for the damaged condition: 35 of 163 positions on P335940 (21.5%).
CAL_MASK_RATE = 0.215

N_BUCKETS = 10         # equal-count buckets, so none is empty


# Arms A, B, C

def training_protocol(model, tok, split: str) -> dict:
    """The exact evaluation used in training (same dataset class, masking and seed). This is the control."""
    ds = CuneiformMLMDataset(split, tok, seed=SEED, deterministic=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    n_correct = n_top5 = n_targets = 0
    with torch.no_grad():
        for ids, attn, labels in loader:
            logits = model(input_ids=ids, attention_mask=attn).logits
            mask = labels != -100
            if not mask.any():
                continue
            top5 = logits.topk(5, dim=-1).indices
            gold = labels.unsqueeze(-1)
            n_correct += int(((top5[..., :1] == gold) & mask.unsqueeze(-1)).sum())
            n_top5 += int(((top5 == gold).any(dim=-1) & mask).sum())
            n_targets += int(mask.sum())

    return {
        "top1": n_correct / max(1, n_targets),
        "top5": n_top5 / max(1, n_targets),
        "n": n_targets,
    }


def leave_one_out_protocol(model, tok, split: str, n_blocks: int = N_BLOCKS) -> dict:
    """
    Mask one position at a time with the rest of the block intact, as
    fusion.lm_posterior does. Reported over all 288 tokens (comparable with arm
    A) and over the 282 sign tokens (what fusion uses).
    """
    records = load_corpus(split)
    blocks = build_blocks(records, tok)

    rng = random.Random(SEED)
    rng.shuffle(blocks)
    blocks = blocks[:n_blocks]

    sup = support_ids(tok)
    not_sup = torch.ones(tok.vocab_size, dtype=torch.bool)
    not_sup[sup] = False

    n_free = n_restricted = n_targets = n_brk = 0

    with torch.no_grad():
        for body in blocks:
            ids = torch.tensor([tok.bos_id] + body + [tok.eos_id], dtype=torch.long)
            n_brk += int((ids == tok.brk_id).sum())

            targets = [i for i, t in enumerate(ids.tolist())
                       if tok.is_predictable(t) and t != tok.unk_id]
            if not targets:
                continue
            rng.shuffle(targets)
            targets = sorted(targets[:MAX_POS_PER_BLOCK])

            attn = (ids != tok.pad_id).long()
            for start in range(0, len(targets), 32):
                chunk = targets[start:start + 32]
                batch = ids.unsqueeze(0).repeat(len(chunk), 1).clone()
                for r, p in enumerate(chunk):
                    batch[r, p] = tok.mask_id

                logits = model(
                    input_ids=batch,
                    attention_mask=attn.unsqueeze(0).repeat(len(chunk), 1),
                ).logits

                for r, p in enumerate(chunk):
                    row = logits[r, p]
                    truth = int(ids[p])
                    if int(row.argmax()) == truth:
                        n_free += 1
                    masked_row = row.clone()
                    masked_row[not_sup] = float("-inf")
                    if int(masked_row.argmax()) == truth:
                        n_restricted += 1
                    n_targets += 1

    total_tokens = sum(len(b) for b in blocks)
    return {
        "top1_free": n_free / max(1, n_targets),
        "top1_restricted": n_restricted / max(1, n_targets),
        "n": n_targets,
        "brk_share": n_brk / max(1, total_tokens),
    }


# Arm D: calibrating the floor

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion, so every accuracy prints with its n and interval."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def unigram_baseline(blocks, tok) -> tuple[float, str]:
    """Accuracy of always answering the commonest sign, computed on this sample."""
    counts = Counter()
    for body in blocks:
        for t in body:
            if tok.is_predictable(t) and t != tok.unk_id:
                counts[t] += 1
    if not counts:
        return (0.0, "?")
    top_id, top_n = counts.most_common(1)[0]
    total = sum(counts.values())
    name = str(tok.id_to_sign.get(top_id, top_id))
    return (top_n / total, name)


def calibration_records(model, tok, split: str, n_blocks: int, per_block: int,
                        mask_rate: float, seed: int = SEED) -> tuple[list, list]:
    """
    Score positions on `split` and record (p_raw, p_norm, correct) for each.
    p_raw is the LM's top posterior over the 282 signs; p_norm renormalises it
    over the signs only. mask_rate 0 is the clean condition; above 0, that share
    of positions is masked together and targets come from the masked set, which
    needs only one forward pass per block. Returns (records, blocks).
    """
    records = load_corpus(split)
    blocks = build_blocks(records, tok)

    rng = random.Random(seed)
    rng.shuffle(blocks)
    blocks = blocks[:n_blocks]

    sup = support_ids(tok)
    not_sup = torch.ones(tok.vocab_size, dtype=torch.bool)
    not_sup[sup] = False

    out: list[tuple[float, float, bool]] = []

    with torch.no_grad():
        for body in blocks:
            ids = torch.tensor([tok.bos_id] + body + [tok.eos_id],
                               dtype=torch.long)
            attn = (ids != tok.pad_id).long()

            predictable = [i for i, t in enumerate(ids.tolist())
                           if tok.is_predictable(t) and t != tok.unk_id]
            if not predictable:
                continue

            if mask_rate <= 0.0:
                # Clean: one isolated mask per row
                targets = list(predictable)
                rng.shuffle(targets)
                targets = sorted(targets[:per_block])

                for start in range(0, len(targets), 32):
                    chunk = targets[start:start + 32]
                    batch = ids.unsqueeze(0).repeat(len(chunk), 1).clone()
                    for r, p in enumerate(chunk):
                        batch[r, p] = tok.mask_id
                    logits = model(
                        input_ids=batch,
                        attention_mask=attn.unsqueeze(0).repeat(len(chunk), 1),
                    ).logits
                    for r, p in enumerate(chunk):
                        out.append(_score_row(logits[r, p], int(ids[p]),
                                              sup, not_sup))
            else:
                # Damaged: mask a set together and score within it
                k = max(1, int(round(mask_rate * len(predictable))))
                damaged = sorted(rng.sample(predictable, min(k, len(predictable))))
                masked = ids.clone()
                masked[damaged] = tok.mask_id

                targets = list(damaged)
                rng.shuffle(targets)
                targets = sorted(targets[:per_block])

                logits = model(input_ids=masked.unsqueeze(0),
                               attention_mask=attn.unsqueeze(0)).logits[0]
                for p in targets:
                    out.append(_score_row(logits[p], int(ids[p]), sup, not_sup))

    return out, blocks


def _score_row(row: torch.Tensor, truth: int, sup: torch.Tensor,
               not_sup: torch.Tensor) -> tuple[float, float, bool]:
    """One logit row -> (raw posterior, renormalised posterior, correct?)."""
    probs = torch.softmax(row.float(), dim=-1)
    sup_mass = float(probs[sup].sum())
    p_raw = float(probs[sup].max())
    p_norm = p_raw / sup_mass if sup_mass > 0 else 0.0

    masked_row = row.clone()
    masked_row[not_sup] = float("-inf")
    correct = int(masked_row.argmax()) == truth
    return (p_raw, p_norm, correct)


def bucket_table(records: list, key: int, baseline: float,
                 n_buckets: int = N_BUCKETS) -> dict:
    """
    Equal-count buckets over the posterior, with per-bucket and cumulative-below
    accuracy. I choose the floor on the cumulative column, since the question
    is whether everything under the line is worth showing as a name.
    """
    vals = sorted(records, key=lambda r: r[key])
    n = len(vals)
    rows = []
    if n == 0:
        return {"rows": rows, "verdict": "no positions scored", "floor": None}

    size = max(1, n // n_buckets)
    edges = [i * size for i in range(n_buckets)] + [n]

    cum_k = cum_n = 0
    for b in range(n_buckets):
        lo_i, hi_i = edges[b], edges[b + 1]
        if lo_i >= hi_i:
            continue
        chunk = vals[lo_i:hi_i]
        k = sum(1 for r in chunk if r[2])
        m = len(chunk)
        cum_k += k
        cum_n += m
        rows.append({
            "lo": chunk[0][key],
            "hi": chunk[-1][key],
            "n": m,
            "acc": k / m,
            "ci": wilson(k, m),
            "cum_n": cum_n,
            "cum_acc": cum_k / cum_n,
            "cum_ci": wilson(cum_k, cum_n),
        })

    # If the bottom and top buckets' intervals overlap, the curve can't pick a floor.
    flat = rows[0]["ci"][1] >= rows[-1]["ci"][0]

    floor = None
    if not flat:
        for r in rows:
            if r["cum_acc"] <= baseline:
                floor = r["hi"]
    if flat:
        verdict = ("FLAT -- the bottom and top buckets' 95% intervals overlap. "
                   "This curve cannot select a floor. Report the inability.")
    elif floor is None:
        verdict = ("DISCRIMINATES but never falls to the unigram baseline. "
                   "No floor satisfies the criterion; every restoration beats "
                   "frequency, so none should be demoted to [x].")
    else:
        verdict = f"FLOOR = {floor:.4f} by the unigram criterion."
    return {"rows": rows, "verdict": verdict, "floor": floor, "flat": flat}


def print_table(title: str, table: dict, baseline: float) -> None:
    print(f"\n  {title}")
    print("  " + "-" * 74)
    print(f"  {'posterior range':<20} {'n':>5} {'acc':>7} "
          f"{'95% CI':>16}   {'acc below hi':>14} {'n below':>8}")
    print("  " + "-" * 74)
    for r in table["rows"]:
        print(f"  {r['lo']:.4f} - {r['hi']:.4f}   {r['n']:>5} "
              f"{100 * r['acc']:>6.1f}% "
              f"[{100 * r['ci'][0]:>5.1f},{100 * r['ci'][1]:>5.1f}]   "
              f"{100 * r['cum_acc']:>13.1f}% {r['cum_n']:>8}")
    print("  " + "-" * 74)
    print(f"  unigram baseline {100 * baseline:.1f}%")
    print(f"  VERDICT: {table['verdict']}")


def calibration_arm(model, tok, split: str, n_blocks: int, per_block: int) -> None:
    print("\n" + "=" * 78)
    print("D. CALIBRATION FOR FIX 3 -- accuracy vs the LM's own posterior")
    print("=" * 78)
    print(f"  split {split} | {n_blocks} blocks | <= {per_block} positions "
          f"per block | {N_BUCKETS} equal-count buckets")

    for label, rate in (("CLEAN   (one isolated mask -- arm C's regime)", 0.0),
                        (f"DAMAGED ({100 * CAL_MASK_RATE:.0f}% masked together "
                         f"-- the LIVE regime)", CAL_MASK_RATE)):
        print("\n" + "-" * 78)
        print(f"  CONDITION: {label}")
        print("-" * 78)
        recs, blocks = calibration_records(model, tok, split, n_blocks,
                                           per_block, rate)
        base, top_name = unigram_baseline(blocks, tok)
        overall_k = sum(1 for r in recs if r[2])
        lo, hi = wilson(overall_k, len(recs))
        print(f"  positions scored : {len(recs):,}")
        print(f"  overall top-1    : {100 * overall_k / max(1, len(recs)):.1f}% "
              f"[{100 * lo:.1f}, {100 * hi:.1f}]")
        print(f"  unigram baseline : {100 * base:.1f}%  "
              f"(commonest sign {top_name}; the recorded figure is 6.2%)")

        print_table("BY RAW POSTERIOR (max P_LM over the 282-sign support)",
                    bucket_table(recs, 0, base), base)
        print_table("BY RENORMALISED POSTERIOR (confidence AMONG signs)",
                    bucket_table(recs, 1, base), base)

    print("\n" + "=" * 78)
    print("READING ARM D")
    print("=" * 78)
    print("  The DAMAGED condition is the one FIX 3 must use. CLEAN is printed")
    print("  beside it so the difference between the two regimes is visible")
    print("  rather than assumed -- calibrating on CLEAN and applying the")
    print("  floor live would repeat the A.4 mistake.")
    print("  If the two posterior definitions choose the same floor, say so in")
    print("  one sentence and use the renormalised one. If they disagree, the")
    print("  disagreement is the finding and neither should be quoted alone.")
    print("  A FLAT verdict is an acceptable outcome: it means [x] stays dead")
    print("  in the live path and the reason is recorded.")
    print("=" * 78)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=CAL_BLOCKS,
                    help="blocks for arm D ONLY. Arms B and C are fixed at "
                         "N_BLOCKS so their recorded figures stay comparable.")
    ap.add_argument("--per-block", type=int, default=CAL_PER_BLOCK,
                    help="positions scored per block in arm D")
    ap.add_argument("--only-calibration", action="store_true",
                    help="skip arms A, B and C on a re-run")
    ap.add_argument("--split", default="test",
                    help="split for arm D. test is the only one with truth "
                         "in the regime fusion runs in; do not calibrate on val.")
    args = ap.parse_args()

    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    torch.manual_seed(SEED)

    tok = SignTokenizer()
    model = load_sequence_model()
    print(f"\n  {tok}\n")

    if not args.only_calibration:
        print("=" * 68)
        print("A. VAL, TRAINING PROTOCOL (control -- expect top1 near 22.2%)")
        print("=" * 68)
        a = training_protocol(model, tok, "val")
        print(f"  top1 {100 * a['top1']:.1f}%  top5 {100 * a['top5']:.1f}%  "
              f"({a['n']:,} masked positions)")

        print("\n" + "=" * 68)
        print("B. VAL, LEAVE-ONE-OUT")
        print("=" * 68)
        b = leave_one_out_protocol(model, tok, "val")
        print(f"  top1 {100 * b['top1_free']:.1f}% (full vocab)  "
              f"{100 * b['top1_restricted']:.1f}% (signs only)  "
              f"({b['n']:,} positions, context {100 * b['brk_share']:.1f}% <brk>)")

        print("\n" + "=" * 68)
        print("C. TEST, LEAVE-ONE-OUT (the regime fusion runs in)")
        print("=" * 68)
        c = leave_one_out_protocol(model, tok, "test")
        print(f"  top1 {100 * c['top1_free']:.1f}% (full vocab)  "
              f"{100 * c['top1_restricted']:.1f}% (signs only)  "
              f"({c['n']:,} positions, context {100 * c['brk_share']:.1f}% <brk>)")

        print("\n" + "=" * 68)
        print("READING A, B AND C")
        print("=" * 68)
        print(f"  A near 22.2%          : checkpoint and loading path are sound")
        print(f"  B >= A                : leave-one-out is implemented correctly")
        print(f"  C << B                : the test split is harder, not a bug")
        print(f"  <brk> share val {100 * b['brk_share']:.1f}% vs "
              f"test {100 * c['brk_share']:.1f}% is the likely reason if C is low")
        print("=" * 68)

    calibration_arm(model, tok, args.split, args.blocks, args.per_block)


if __name__ == "__main__":
    main()