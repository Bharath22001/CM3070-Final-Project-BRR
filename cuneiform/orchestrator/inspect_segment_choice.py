"""
Tries to choose the segment clustering cell size, and tests whether keeping
only annotated views fixes duplicate surfaces.

The score is sequence length against the corpus (slots divided by real corpus
tokens), which merging surfaces inflates and over-splitting deflates. It is a
length measure, not a correctness measure. View selection uses the annotation,
so it is an evaluation-only device. I later found this script's choice rested
on a median that one tablet could move, so its conclusion doesn't stand;
inspect_recovery_sweep.py replaced it.

Run from the project root:
    python -m cuneiform.orchestrator.inspect_segment_choice
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from cuneiform import config
from cuneiform.orchestrator.live_adapter import (
    build_slots, grid_components, load_detections,
)

CELLS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
RULE = "=" * 78


def head(t: str) -> None:
    print("\n" + RULE)
    print(t)
    print(RULE)


def verdict(ok, good: str, bad: str) -> str:
    return f"  => {'PASS. ' + good if bool(ok) else 'FAIL. ' + bad}"


def corpus_tokens(tablet: str) -> int:
    """Real (non-<brk>) corpus tokens for one tablet."""
    n = 0
    with config.D2_CORPUS_JSONL.open("r", encoding="utf-8") as fh:
        for raw in fh:
            rec = json.loads(raw)
            if rec["tablet"] == tablet:
                n += sum(1 for t in rec["tokens"] if t != config.BROKEN_LABEL)
    return n


def cluster(dets, cell_factor, min_size):
    """Returns index arrays in reading order, before any anchoring filter."""
    cx = np.array([d["cx"] for d in dets], float)
    cy = np.array([d["cy"] for d in dets], float)
    w = np.array([d["x2"] - d["x1"] for d in dets], float)
    cell = max(float(np.median(w)) * cell_factor, 1.0)
    labels = grid_components(cx, cy, cell)
    groups = [np.flatnonzero(labels == l) for l in np.unique(labels)]
    groups = [g for g in groups if len(g) >= min_size]
    groups.sort(key=lambda ix: (float(cy[ix].min()), float(cx[ix].min())))
    return groups


def anchored(dets, groups, g_ann) -> list:
    """Keep only clusters holding at least one annotated box centre."""
    if not len(g_ann):
        return groups
    ax = g_ann["cx"].to_numpy(float)
    ay = g_ann["cy"].to_numpy(float)
    out = []
    for ix in groups:
        x1 = min(dets[i]["x1"] for i in ix)
        y1 = min(dets[i]["y1"] for i in ix)
        x2 = max(dets[i]["x2"] for i in ix)
        y2 = max(dets[i]["y2"] for i in ix)
        if ((ax >= x1) & (ax <= x2) & (ay >= y1) & (ay <= y2)).any():
            out.append(ix)
    return out


def ratio(dets, groups, tokens) -> float:
    if not groups or tokens == 0:
        return float("nan")
    return len(build_slots(dets, groups)) / tokens


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-size", type=int, default=5)
    args = ap.parse_args()

    from cuneiform.domain1_vision.group_lines import load_boxes

    bb = load_boxes()
    tablets = sorted(p.stem for p in config.D1_LIVE_DETECTIONS.glob("*.json")
                     if not p.stem.startswith("_"))

    head("A. SETUP")
    print(f"  tablets: {len(tablets)}   conf: {args.conf}   "
          f"min cluster: {args.min_size}")
    print("  metric: slots / real corpus tokens. 1.00 = a position exists for")
    print("  every sign. LENGTH ONLY -- says nothing about correctness.")

    data = {}
    for t in tablets:
        dets = load_detections(t, args.conf)
        toks = corpus_tokens(t)
        g_ann = bb[bb["tablet_CDLI"] == t]
        row = {"dets": len(dets), "tokens": toks,
               "n_ann_seg": int(g_ann["segm_idx"].nunique()) if len(g_ann) else 0}
        for cf in CELLS:
            groups = cluster(dets, cf, args.min_size) if dets else []
            anc = anchored(dets, groups, g_ann)
            row[cf] = {
                "n_all": len(groups), "n_anc": len(anc),
                "r_all": ratio(dets, groups, toks),
                "r_anc": ratio(dets, anc, toks),
            }
        data[t] = row

    def med(cf, key):
        v = [data[t][cf][key] for t in tablets
             if not np.isnan(data[t][cf][key])] if "r_" in key else \
            [data[t][cf][key] for t in tablets]
        return float(np.median(v)) if v else float("nan")

    def score(cf, key):
        """Median absolute distance from 1.0; lower is better."""
        v = [abs(data[t][cf][key] - 1.0) for t in tablets
             if not np.isnan(data[t][cf][key])]
        return float(np.median(v)) if v else float("inf")

    head("B. CELL SWEEP -- all clusters vs anchored clusters only")
    print(f"  {'cell':>5} {'med clusters':>13} {'med anchored':>13} "
          f"{'med ratio ALL':>14} {'med ratio ANC':>14} {'|err| ANC':>10}")
    for cf in CELLS:
        print(f"  {cf:>5.1f} {med(cf,'n_all'):>13.1f} {med(cf,'n_anc'):>13.1f} "
              f"{med(cf,'r_all'):>14.2f} {med(cf,'r_anc'):>14.2f} "
              f"{score(cf,'r_anc'):>10.2f}")

    best = min(CELLS, key=lambda cf: score(cf, "r_anc"))
    print(f"\n  best cell by median |ratio - 1| on anchored clusters: {best}")

    head(f"C. PER-TABLET AT cell = {best}")
    print(f"  {'tablet':<12} {'det':>5} {'toks':>6} {'ann':>4} "
          f"{'cl':>3} {'anc':>4} {'r_ALL':>7} {'r_ANC':>7}")
    for t in tablets:
        d, c = data[t], data[t][best]
        print(f"  {t:<12} {d['dets']:>5} {d['tokens']:>6} {d['n_ann_seg']:>4} "
              f"{c['n_all']:>3} {c['n_anc']:>4} "
              f"{c['r_all']:>7.2f} {c['r_anc']:>7.2f}")

    head("D. DOES VIEW SELECTION EARN ITS KEEP?")
    gain = [abs(data[t][best]["r_all"] - 1) - abs(data[t][best]["r_anc"] - 1)
            for t in tablets if not np.isnan(data[t][best]["r_anc"])]
    helped = sum(1 for g in gain if g > 0.05)
    hurt = sum(1 for g in gain if g < -0.05)
    print(f"  tablets improved by anchoring : {helped}")
    print(f"  tablets made worse            : {hurt}")
    print(f"  median |err| ALL / ANCHORED   : "
          f"{score(best,'r_all'):.2f} / {score(best,'r_anc'):.2f}")
    print(verdict(
        score(best, "r_anc") < score(best, "r_all") and helped > hurt,
        "View selection reduces length error. Use it in evaluation, DECLARED as an "
        "evaluation-time device that consumes ground truth.",
        "View selection does not help. The duplicate-view problem needs a "
        "different rule; do not ship this one."))

    head("E. THE RESIDUAL")
    bad = [t for t in tablets
           if not np.isnan(data[t][best]["r_anc"])
           and abs(data[t][best]["r_anc"] - 1) > 0.5]
    print(f"  tablets still beyond +/-0.5 after anchoring: {len(bad)}")
    for t in bad:
        print(f"    {t:<12} r_anc {data[t][best]['r_anc']:.2f}")
    print("\n  These are the ones to name as a limitation. A tablet whose")
    print("  ratio stays far from 1 after view selection has a problem the")
    print("  geometry cannot fix.")
    print("\n" + RULE)


if __name__ == "__main__":
    main()