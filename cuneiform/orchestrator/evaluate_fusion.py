"""
Measures whether combining vision and language beats either alone on the 32
held-out test tablets, and sweeps the fusion weight lambda. All vision input
here is simulated (see simulate_vision.py) and must be labelled that way.
evaluate_live.py does the same on real detections and imports this file's
metric functions.

Arms:
  1. Vision only: missed signs produce no token.
  2. Threshold -> mask: detections below T become <mask>, the LM fills them and
     the vision scores are thrown away.
  3. Fusion: log P_vision + lambda * log P_LM at every position, so vision
     always keeps a vote.
  4. Language ceiling: the LM on clean ground truth with matched random damage,
     a ceiling rather than a fair competitor.

For fusion the LM sees vision's labels as context, with only missed and unknown
positions masked; masking everything would leave it nothing to condition on.
WER is edit distance over signs, CER over the space-joined sign names. <brk>
and <unk> positions aren't scored, and every arm is scored on the same positions.

Run from the project root:
    python -m cuneiform.orchestrator.evaluate_fusion
"""

from __future__ import annotations

import json
import random

import torch

from cuneiform import config
from cuneiform.domain2_sequence.dataset import build_blocks, load_corpus
from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.orchestrator.fusion import (
    Detection, argmax_labels, fuse, lm_posterior, load_sequence_model,
    vision_posterior,
)
from cuneiform.orchestrator.simulate_vision import VisionSimulator

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

THRESHOLD = 0.50
MAX_BLOCKS = 60
SEED = 1729
RESULTS_JSON = config.ARTIFACTS_DIR / "fusion_results.json"


# Metrics

def levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_sign_names() -> dict[int, str]:
    """train_label to sign name, falling back to MZL#### so CER covers every sign."""
    if not config.D3_SIGN_LOOKUP.exists():
        raise FileNotFoundError(
            f"{config.D3_SIGN_LOOKUP} not found. Run the Domain 3 signlist "
            f"pipeline first."
        )
    with open(config.D3_SIGN_LOOKUP, "r", encoding="utf-8") as f:
        data = json.load(f)

    lookup = data["lookup"]        # the file is {lookup: ..., alternates: ...}
    names: dict[int, str] = {}
    for label, rec in lookup.items():
        name = rec.get("sign") or f"MZL{rec.get('mzl', 0)}"
        names[int(label)] = name
    return names


def score(pred: list[int], truth: list[int], names: dict[int, str]) -> dict:
    wer = levenshtein(pred, truth) / max(1, len(truth))
    p_txt = " ".join(names.get(s, f"?{s}") for s in pred)
    t_txt = " ".join(names.get(s, f"?{s}") for s in truth)
    cer = levenshtein(list(p_txt), list(t_txt)) / max(1, len(t_txt))
    return {"wer": wer, "cer": cer}


# Building the LM input for each arm

def vision_context(ids: torch.Tensor, dets: dict[int, Detection],
                   tok: SignTokenizer, threshold: float) -> torch.Tensor:
    """
    The block as the LM receives it: named detections at or above `threshold`
    become tokens and everything else becomes <mask>. threshold 0.0 is fusion
    (the confidence goes into the vision posterior instead); threshold T is
    the threshold design.
    """
    out = ids.clone()
    for pos, det in dets.items():
        if det.is_informative and det.confidence >= threshold:
            token = tok.sign_to_id.get(det.train_label)
            out[pos] = token if token is not None else tok.mask_id
        else:
            out[pos] = tok.mask_id
    return out


def synthetic_damage(ids: torch.Tensor, positions: list[int], rate: float,
                     tok: SignTokenizer, rng: random.Random) -> torch.Tensor:
    out = ids.clone()
    hidden = rng.sample(positions, k=int(len(positions) * rate))
    for p in hidden:
        out[p] = tok.mask_id
    return out


def context_density(inp: torch.Tensor, positions: list[int],
                    tok: SignTokenizer) -> float:
    """Share of scored positions holding a real token rather than <mask>."""
    return sum(1 for p in positions if inp[p] != tok.mask_id) / max(1, len(positions))


# Main evaluation

def main() -> None:
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    tok = SignTokenizer()
    names = load_sign_names()
    model = load_sequence_model()
    sim = VisionSimulator(tok, verbose=True)

    records = load_corpus("test")
    blocks = build_blocks(records, tok)
    if MAX_BLOCKS:
        blocks = blocks[:MAX_BLOCKS]

    tablets = {r["tablet"] for r in records}
    print(f"\n  test split: {len(tablets)} tablets | {len(blocks)} blocks scored")
    print(f"  threshold arm at T={THRESHOLD} | lambda sweep {LAMBDAS}")
    print("\n  running (leave-one-out LM posteriors are the slow part)...")

    uniform = torch.full((1, tok.vocab_size), 1.0 / tok.vocab_size,
                         dtype=torch.float64)

    agg: dict[str, list[dict]] = {}
    n_scored = n_brk = n_unk = 0
    n_overturned = n_broken = 0
    dens_thr: list[float] = []
    dens_fus: list[float] = []

    def add(arm: str, pred: list[int], truth: list[int]) -> None:
        agg.setdefault(arm, []).append(score(pred, truth, names))

    for block_id, body in enumerate(blocks):
        ids = torch.tensor([tok.bos_id] + body + [tok.eos_id], dtype=torch.long)

        n_brk += int((ids == tok.brk_id).sum())
        n_unk += int((ids == tok.unk_id).sum())

        dets = sim.simulate(ids.tolist(), block_id)
        if not dets:
            continue
        positions = sorted(dets)
        truth_map = {p: tok.id_to_sign[ids[p].item()] for p in positions}
        truth = [truth_map[p] for p in positions]
        n_scored += len(positions)

        # Arm 1: vision only. Missed positions emit nothing.
        add("1_vision_only",
            [dets[p].train_label for p in positions if dets[p].is_informative],
            truth)

        # Arm 2: threshold -> mask
        thr_in = vision_context(ids, dets, tok, THRESHOLD)
        dens_thr.append(context_density(thr_in, positions, tok))
        thr_masked = [p for p in positions if thr_in[p] == tok.mask_id]
        pred2 = {p: dets[p].train_label for p in positions
                 if thr_in[p] != tok.mask_id}
        if thr_masked:
            p_lm_thr = lm_posterior(model, thr_in, thr_masked, tok)
            filled = argmax_labels(fuse(uniform.expand_as(p_lm_thr),
                                        p_lm_thr, 1.0, tok), tok)
            pred2.update(dict(zip(thr_masked, filled)))
        add("2_threshold_mask", [pred2[p] for p in positions], truth)

        # Arm 3: fusion. Vision labels are the context; leave-one-out inside
        # lm_posterior stops the model reading the answer.
        fus_in = vision_context(ids, dets, tok, 0.0)
        dens_fus.append(context_density(fus_in, positions, tok))
        p_lm = lm_posterior(model, fus_in, positions, tok)
        p_v = vision_posterior([dets[p] for p in positions], tok)

        for lam in LAMBDAS:
            preds = argmax_labels(fuse(p_v, p_lm, lam, tok), tok)
            add(f"3_fusion_lam{lam}", preds, truth)

            if lam == 1.0:
                for p, got in zip(positions, preds):
                    if not dets[p].is_informative:
                        continue
                    was_wrong = dets[p].train_label != truth_map[p]
                    now_right = got == truth_map[p]
                    if was_wrong and now_right:
                        n_overturned += 1
                    elif not was_wrong and not now_right:
                        n_broken += 1

        # Arm 4: language ceiling, no vision, matched damage rate
        dmg_in = synthetic_damage(ids, positions, 0.80, tok, rng)
        dmg_pos = [p for p in positions if dmg_in[p] == tok.mask_id]
        pred4 = {p: truth_map[p] for p in positions if dmg_in[p] != tok.mask_id}
        if dmg_pos:
            p_lm4 = lm_posterior(model, dmg_in, dmg_pos, tok)
            filled = argmax_labels(fuse(uniform.expand_as(p_lm4),
                                        p_lm4, 1.0, tok), tok)
            pred4.update(dict(zip(dmg_pos, filled)))
        add("4_language_ceiling", [pred4[p] for p in positions], truth)

        if (block_id + 1) % 10 == 0:
            print(f"    {block_id + 1}/{len(blocks)} blocks", flush=True)

    mean = lambda arm, k: sum(d[k] for d in agg[arm]) / max(1, len(agg[arm]))
    avg = lambda xs: sum(xs) / max(1, len(xs))

    print("\n" + "=" * 68)
    print("RESULTS  (SIMULATED vision input -- see module docstring)")
    print("=" * 68)
    print(f"  {n_scored:,} scored positions | excluded: "
          f"{n_brk:,} <brk>, {n_unk:,} <unk>")
    print(f"  LM context density -- threshold arm {100 * avg(dens_thr):.1f}% | "
          f"fusion arm {100 * avg(dens_fus):.1f}% of scored positions")
    print("  (near 0% for the fusion arm means the LM is predicting from "
          "nothing -- a bug, not a finding)")
    print()
    print(f"  {'arm':<30} {'CER':>8} {'WER':>8}")
    print("  " + "-" * 48)
    for arm in ("1_vision_only", "2_threshold_mask"):
        print(f"  {arm:<30} {mean(arm, 'cer'):>8.4f} {mean(arm, 'wer'):>8.4f}")
    best = min(LAMBDAS, key=lambda l: mean(f"3_fusion_lam{l}", "wer"))
    arm3 = f"3_fusion_lam{best}"
    print(f"  {'3_fusion (best lambda=' + str(best) + ')':<30} "
          f"{mean(arm3, 'cer'):>8.4f} {mean(arm3, 'wer'):>8.4f}")
    print(f"  {'4_language_ceiling *':<30} "
          f"{mean('4_language_ceiling', 'cer'):>8.4f} "
          f"{mean('4_language_ceiling', 'wer'):>8.4f}")
    print("  * sees CORRECT ground-truth context; arms 2-3 see partly wrong "
          "context.")

    print("\n  LAMBDA SWEEP")
    print(f"  {'lambda':>8} {'CER':>10} {'WER':>10}")
    for lam in LAMBDAS:
        a = f"3_fusion_lam{lam}"
        note = "  <- vision + frequency tie-break" if lam == 0.0 else ""
        print(f"  {lam:>8} {mean(a, 'cer'):>10.4f} {mean(a, 'wer'):>10.4f}{note}")

    print(f"\n  At lambda=1.0, on positions vision named:")
    print(f"    wrong detections OVERTURNED by the LM : {n_overturned:,}")
    print(f"    correct detections BROKEN by the LM   : {n_broken:,}")
    print("    (a threshold design can do neither)")

    w2, w3 = mean("2_threshold_mask", "wer"), mean(arm3, "wer")
    print(f"\n  fusion vs threshold: WER {w2:.4f} -> {w3:.4f} "
          f"({100 * (w2 - w3) / max(1e-9, w2):+.1f}%)")

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "simulated_vision": True,
            "n_blocks": len(blocks), "n_scored": n_scored,
            "threshold": THRESHOLD, "best_lambda": best,
            "ctx_density_threshold": avg(dens_thr),
            "ctx_density_fusion": avg(dens_fus),
            "overturned": n_overturned, "broken": n_broken,
            "arms": {a: {"cer": mean(a, "cer"), "wer": mean(a, "wer")}
                     for a in agg},
            "per_block": agg,
        }, f, indent=2)
    print(f"\n  written: {RESULTS_JSON}")
    print("=" * 68)


if __name__ == "__main__":
    main()