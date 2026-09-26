"""
Turns an unordered set of detections into an ordered reading sequence, with a
<mask> wherever a sign should be but nothing was detected. Without this, a
missed sign would simply vanish and Domain 2 could never repair it.

I recover the reading lines, sort left to right, and where two neighbouring
detections are far enough apart to hold signs I insert that many masks. Pitch
is a ratio of the segment's median box width, since photo scales vary about
tenfold. The count is only approximate (exact about a quarter of the time for
a 4-sign gap), but a wrong count only misaligns; it never invents a sign.

Line edges aren't extrapolated by default, since a line's true length is
unknown; extend_edges=True turns it on.

Run the self-test from the project root:
    python -m cuneiform.orchestrator.insert_gaps
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cuneiform.domain1_vision.group_lines import (
    group_segment, load_boxes,
)

# Sign pitch as a multiple of the segment's median box width, used when a
# segment is too sparse to measure. calibrate_pitch_ratio() measures about 0.93
# on the current annotations; I keep 0.92 as the shipped value.
DEFAULT_PITCH_RATIO = 0.92

# A gap must be wider than this many pitches before any mask is inserted.
MIN_GAP_PITCHES = 1.5

# Cap on masks per gap, since counting gets less accurate as gaps get longer.
MAX_INSERT = 8


@dataclass
class Slot:
    """One position in the reconstructed reading sequence."""
    kind: str                    # "detected" | "inserted"
    line: int
    x: float
    label: int | None = None
    confidence: float = 0.0


def calibrate_pitch_ratio(bb: pd.DataFrame, verbose: bool = True) -> float:
    """
    Measure sign pitch as a multiple of median box width, per segment, and
    report its spread to show whether one constant is good enough.
    """
    ratios = []
    for _, g in bb.groupby(["split", "tablet_CDLI", "segm_idx"]):
        if len(g) < 10:
            continue
        line_of = group_segment(g)
        cx = g["cx"].to_numpy(float)
        gaps = []
        for lid in np.unique(line_of):
            m = line_of == lid
            if m.sum() < 3:
                continue
            gaps.extend(np.diff(np.sort(cx[m])).tolist())
        if len(gaps) < 5:
            continue
        w = float(np.median(g["w"]))
        if w > 0:
            ratios.append(float(np.median(gaps)) / w)

    r = np.array(ratios)
    if verbose:
        print(f"  pitch / box-width ratio across {len(r)} segments:")
        for q in (10, 25, 50, 75, 90):
            print(f"    p{q:<3} {np.percentile(r, q):>6.2f}")
        print(f"    median {np.median(r):.2f}  "
              f"(a tight spread means one constant is a fair model)")
    return float(np.median(r))


def insert_masks(cx, cy, w, h, labels, confs, pitch_ratio: float,
                 extend_edges: bool = False) -> list[Slot]:
    """
    Build the reading sequence for one segment from its detections. cx, cy, w, h
    are in tablet coordinates. Lines go top to bottom by mean y, and boxes left
    to right within each line.
    """
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)
    w = np.asarray(w, float)
    # Convert to arrays so labels[a] is positional even if a pandas Series is passed.
    labels = np.asarray(labels)
    confs = np.asarray(confs, float)

    if len(cx) == 0:
        return []
    if len(cx) < 3:
        order = np.argsort(cx)
        return [Slot("detected", 0, float(cx[i]), labels[i], confs[i])
                for i in order]

    g = pd.DataFrame({"cx": cx, "cy": cy, "w": w, "h": np.asarray(h, float)})
    line_of = group_segment(g)

    pitch = max(float(np.median(w)) * pitch_ratio, 1.0)

    # Lines in reading order: top to bottom by mean y.
    line_ids = sorted(np.unique(line_of),
                      key=lambda l: float(np.mean(cy[line_of == l])))

    # For edge extrapolation, the segment's typical line extent.
    if extend_edges:
        starts = [float(np.min(cx[line_of == l])) for l in line_ids]
        ends = [float(np.max(cx[line_of == l])) for l in line_ids]
        seg_start, seg_end = float(np.median(starts)), float(np.median(ends))

    slots: list[Slot] = []
    for out_line, lid in enumerate(line_ids):
        m = np.flatnonzero(line_of == lid)
        m = m[np.argsort(cx[m])]

        if extend_edges:
            lead = (cx[m[0]] - seg_start) / pitch
            for j in range(min(int(round(lead)), MAX_INSERT)):
                slots.append(Slot("inserted", out_line,
                                  seg_start + j * pitch))

        for a, b in zip(m, m[1:]):
            slots.append(Slot("detected", out_line, float(cx[a]),
                              labels[a], confs[a]))
            gap = cx[b] - cx[a]
            if gap > MIN_GAP_PITCHES * pitch:
                # round(gap/pitch) counts spans; one of them is the step to the
                # next detection, so one fewer sign is missing.
                n = min(int(round(gap / pitch)) - 1, MAX_INSERT)
                for j in range(max(0, n)):
                    slots.append(Slot("inserted", out_line,
                                      float(cx[a] + (j + 1) * pitch)))
        slots.append(Slot("detected", out_line, float(cx[m[-1]]),
                          labels[m[-1]], confs[m[-1]]))

        if extend_edges:
            trail = (seg_end - cx[m[-1]]) / pitch
            for j in range(min(int(round(trail)), MAX_INSERT)):
                slots.append(Slot("inserted", out_line,
                                  float(cx[m[-1]] + (j + 1) * pitch)))

    return slots


# Self-test: does insertion recover the missing positions?

def _evaluate(bb: pd.DataFrame, pitch_ratio: float, recall: float,
              extend_edges: bool, seed: int = 1729) -> dict:
    """
    Drop boxes to simulate the detector's recall, then check how well the
    inserter restores the sequence length, i.e. whether every missing sign
    became a <mask> instead of disappearing.
    """
    rng = np.random.default_rng(seed)
    true_len = kept_len = out_len = 0
    per_seg_err, per_seg_base = [], []

    for _, g in bb.groupby(["split", "tablet_CDLI", "segm_idx"]):
        if len(g) < 10:
            continue
        n = len(g)
        keep = rng.random(n) < recall
        if keep.sum() < 2:
            continue

        sub = g[keep]
        slots = insert_masks(
            sub["cx"], sub["cy"], sub["w"], sub["h"],
            sub["train_label"].to_numpy(), np.full(int(keep.sum()), 0.9),
            pitch_ratio, extend_edges=extend_edges)

        true_len += n
        kept_len += int(keep.sum())
        out_len += len(slots)
        per_seg_err.append(len(slots) - n)
        per_seg_base.append(int(keep.sum()) - n)

    err = np.array(per_seg_err)
    base = np.array(per_seg_base)
    return {
        "true": true_len, "kept": kept_len, "out": out_len,
        "recovery": out_len / max(1, true_len),
        "baseline_recovery": kept_len / max(1, true_len),
        "median_abs_err": float(np.median(np.abs(err))),
        "median_abs_base": float(np.median(np.abs(base))),
        "n_segments": len(err),
    }


def main() -> None:
    print("=" * 70)
    print("GAP INSERTION SELF-TEST")
    print("=" * 70)

    bb = load_boxes()
    print(f"  {len(bb):,} boxes | {bb['tablet_CDLI'].nunique()} tablets\n")

    ratio = calibrate_pitch_ratio(bb)
    print(f"\n  using pitch ratio {ratio:.2f} x median box width")

    print("\n" + "-" * 70)
    print("  SEQUENCE LENGTH RECOVERY at the detector's measured 24% recall")
    print("-" * 70)
    print("  'recovery' = reconstructed length / true length.")
    print("  1.00 is perfect; the baseline is what happens with NO insertion,")
    print("  i.e. the silent deletion the pipeline does today.\n")

    for edges in (False, True):
        r = _evaluate(bb, ratio, recall=0.243, extend_edges=edges)
        tag = "interior gaps + edge extrapolation" if edges else "interior gaps only"
        print(f"  {tag}")
        print(f"    segments               : {r['n_segments']}")
        print(f"    true positions         : {r['true']:,}")
        print(f"    detected (baseline)    : {r['kept']:,}  "
              f"recovery {r['baseline_recovery']:.2f}")
        print(f"    after insertion        : {r['out']:,}  "
              f"recovery {r['recovery']:.2f}")
        print(f"    median |length error| per segment: "
              f"{r['median_abs_err']:.0f}  (baseline {r['median_abs_base']:.0f})")
        print()

    print("-" * 70)
    print("  SENSITIVITY TO RECALL (interior gaps only)")
    print("-" * 70)
    print(f"  {'recall':>8} {'baseline':>10} {'inserted':>10}")
    for rec in (0.243, 0.40, 0.60, 0.80):
        r = _evaluate(bb, ratio, recall=rec, extend_edges=False)
        print(f"  {rec:>8.2f} {r['baseline_recovery']:>10.2f} "
              f"{r['recovery']:>10.2f}")
    print("\n  (Recovery should approach 1.00 as recall rises: with more")
    print("   anchors the gaps are shorter and the count is more often right.)")
    print("=" * 70)


if __name__ == "__main__":
    main()