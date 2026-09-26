"""
My first attempt at grouping annotated boxes into reading lines from geometry
alone, then measuring the sign pitch (centre-to-centre spacing) along each line.

It finds each segment's line slope with a projection sweep, bands the boxes
along that slope, and checks the result against the line anchors. This version
failed its own check (it produced an impossible 5 px pitch), so I replaced it
with group_lines.py and kept this one as a record. It writes nothing.

Run from the project root:
    python -m cuneiform.domain1_vision.inspect_box_pitch
"""

from __future__ import annotations

import ast
from collections import defaultdict

import numpy as np
import pandas as pd

from cuneiform import config

LINE_TRAIN = config.ANNOTATIONS_DIR / "line_annotations_train.csv"

# Candidate slopes, wide enough to cover every slope I measured on real lines.
SLOPES = np.linspace(-0.18, 0.08, 53)


def rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_boxes() -> pd.DataFrame:
    """Load bbox annotations, honouring the [x1,y1,x2,y2] convention."""
    frames = []
    for path in (config.TRAIN_BBOX_CSV, config.TEST_BBOX_CSV):
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
        df["split"] = "train" if "train" in path.name else "test"
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No bbox annotation CSVs found.")

    bb = pd.concat(frames, ignore_index=True)
    bb["bbox"] = bb["bbox"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v)

    arr = np.asarray(bb["bbox"].tolist(), dtype=float)
    if not ((arr[:, 2] > arr[:, 0]).all() and (arr[:, 3] > arr[:, 1]).all()):
        raise ValueError("bbox column is not [x1,y1,x2,y2]. Stop and check.")

    bb["cx"] = (arr[:, 0] + arr[:, 2]) / 2
    bb["cy"] = (arr[:, 1] + arr[:, 3]) / 2
    bb["w"] = arr[:, 2] - arr[:, 0]
    bb["h"] = arr[:, 3] - arr[:, 1]

    print(f"  {len(bb):,} boxes | {bb['tablet_CDLI'].nunique()} tablets")
    print(f"  median box {np.median(bb['w']):.0f} x {np.median(bb['h']):.0f} px")
    return bb


def best_slope(cx: np.ndarray, cy: np.ndarray, band: float) -> float:
    """
    Pick the slope whose projected y histogram is most peaked, scored by the
    sum of squared bin counts. A wrong slope smears each line across bins.
    """
    best, best_score = 0.0, -1.0
    for s in SLOPES:
        proj = cy - s * cx
        bins = np.floor((proj - proj.min()) / band).astype(int)
        counts = np.bincount(bins)
        score = float((counts.astype(float) ** 2).sum())
        if score > best_score:
            best, best_score = float(s), score
    return best


def group_lines(g: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Assign a line id to every box in one segment. The band height scales with
    the segment's median box height, and I split wherever the projected
    coordinate jumps by more than one band.
    """
    cx = g["cx"].to_numpy(float)
    cy = g["cy"].to_numpy(float)
    band = float(np.median(g["h"])) * 1.2

    slope = best_slope(cx, cy, band)
    proj = cy - slope * cx

    order = np.argsort(proj)
    line_of = np.empty(len(g), dtype=int)
    current = 0
    line_of[order[0]] = 0
    for prev, idx in zip(order, order[1:]):
        if proj[idx] - proj[prev] > band:
            current += 1
        line_of[idx] = current
    return line_of, slope


def measure(bb: pd.DataFrame) -> dict:
    """Group every segment and collect pitch statistics."""
    rule("1. GROUPING BOXES INTO LINES, AND SIGN PITCH")

    pitches, per_line_cv, line_sizes, slopes = [], [], [], []
    overlaps = 0
    n_pairs = 0
    grouped: dict[tuple, dict] = {}

    for (split, tablet, segm), g in bb.groupby(
            ["split", "tablet_CDLI", "segm_idx"]):
        if len(g) < 3:
            continue
        line_of, slope = group_lines(g)
        slopes.append(slope)
        grouped[(split, tablet, segm)] = {
            "line_of": line_of, "slope": slope,
            "cx": g["cx"].to_numpy(float), "cy": g["cy"].to_numpy(float),
            "w": g["w"].to_numpy(float),
        }

        cx = g["cx"].to_numpy(float)
        w = g["w"].to_numpy(float)
        for lid in np.unique(line_of):
            m = line_of == lid
            line_sizes.append(int(m.sum()))
            if m.sum() < 3:
                continue
            xs = np.sort(cx[m])
            d = np.diff(xs)
            pitches.extend(d.tolist())
            n_pairs += len(d)
            # Two boxes almost on top of each other suggest merged lines.
            overlaps += int((d < 0.5 * np.mean(w[m])).sum())
            if len(d) >= 2 and d.mean() > 0:
                per_line_cv.append(d.std() / d.mean())

    p = np.array(pitches)
    cv = np.array(per_line_cv)

    print(f"  segments grouped        : {len(grouped):,}")
    print(f"  lines recovered         : {len(line_sizes):,}")
    print(f"  median boxes per line   : {np.median(line_sizes):.0f}  "
          f"(p10 {np.percentile(line_sizes, 10):.0f}, "
          f"p90 {np.percentile(line_sizes, 90):.0f})")
    print(f"  fitted slope: median {np.median(slopes):+.3f} | "
          f"p5 {np.percentile(slopes, 5):+.3f} | "
          f"p95 {np.percentile(slopes, 95):+.3f}")
    print(f"    [inspect_line_geometry measured -0.027 median from anchors]")

    print(f"\n  SIGN PITCH (centre-to-centre, px), {len(p):,} neighbour pairs:")
    for q in (5, 25, 50, 75, 95):
        print(f"    p{q:<3} {np.percentile(p, q):>7.1f}")
    print(f"    mean {p.mean():>7.1f}  std {p.std():>7.1f}")
    print(f"  suspiciously tight pairs (< half a sign width): "
          f"{overlaps:,} ({100 * overlaps / max(1, n_pairs):.1f}%)")
    print("    [a high figure means lines were over-merged -- grouping is wrong]")

    print(f"\n  PITCH COEFFICIENT OF VARIATION, within a line:")
    for q in (25, 50, 75, 90):
        print(f"    p{q:<3} {np.percentile(cv, q):>7.3f}")
    print(f"    mean {cv.mean():>7.3f}")

    return {"grouped": grouped, "pitch": p, "cv": cv}


def validate_against_anchors(bb: pd.DataFrame, grouped: dict) -> None:
    """
    Check the grouping against the line anchors. Anchors are sparse, so I
    compare line counts per tablet rather than ids (segm_idx clashes between
    the train and test files).
    """
    rule("2. VALIDATING THE GROUPING AGAINST THE ANCHOR FILE")

    if not LINE_TRAIN.exists():
        print(f"  {LINE_TRAIN} missing; skipping.")
        return

    anchors = pd.read_csv(LINE_TRAIN)

    recovered = defaultdict(int)
    for (split, tablet, segm), d in grouped.items():
        recovered[tablet] += len(np.unique(d["line_of"]))

    anchor_lines = anchors.groupby("tablet_CDLI").apply(
        lambda g: g.groupby("segm_idx")["line_idx"].nunique().sum(),
        include_groups=False).to_dict()

    shared = sorted(set(recovered) & set(anchor_lines))
    print(f"  tablets comparable: {len(shared)}")
    if not shared:
        print("  ** No overlap between the two files.")
        return

    print(f"\n  {'tablet':<14} {'anchor lines':>13} {'recovered':>10} {'ratio':>7}")
    ratios = []
    for t in shared:
        a, r = anchor_lines[t], recovered[t]
        ratio = r / max(1, a)
        ratios.append(ratio)
        print(f"  {t:<14} {a:>13} {r:>10} {ratio:>7.2f}")

    med = float(np.median(ratios))
    print(f"\n  median ratio (recovered / anchor lines): {med:.2f}")
    if 0.8 <= med <= 1.25:
        print("  => Line counts agree. The geometric grouping is consistent")
        print("     with the independent human line annotation.")
    elif med > 1.25:
        print("  => Recovering MORE lines than annotated: the grouper is")
        print("     SPLITTING lines. Pitch would then be measured within")
        print("     fragments and understate the true spacing.")
    else:
        print("  => Recovering FEWER lines than annotated: the grouper is")
        print("     MERGING lines. Pitch would then include jumps between")
        print("     lines and overstate the true spacing.")


def gap_simulation(pitch: np.ndarray, cv: np.ndarray) -> None:
    """
    Estimate how often round(gap / median_pitch) gives the right number of
    missing signs, by building synthetic gaps from k real consecutive pitches
    and checking whether the estimate recovers k.
    """
    rule("3. WOULD GAP COUNTING ACTUALLY WORK?")

    med = float(np.median(pitch))
    rng = np.random.default_rng(1729)

    print(f"  median pitch used as the unit: {med:.1f} px\n")
    print(f"  {'true k':>7} {'exact':>8} {'within 1':>10}")
    for k in (1, 2, 3, 4, 6, 8):
        draws = rng.choice(pitch, size=(4000, k), replace=True)
        est = np.round(draws.sum(axis=1) / med).astype(int)
        exact = float((est == k).mean())
        near = float((np.abs(est - k) <= 1).mean())
        print(f"  {k:>7} {100 * exact:>7.1f}% {100 * near:>9.1f}%")

    print(f"\n  median within-line pitch CV: {np.median(cv):.3f}")
    if np.median(cv) < 0.5:
        print("  => Pitch is regular enough to COUNT. Insert")
        print("     round(gap / median_pitch) masks, and report the accuracy")
        print("     table above as the justification.")
    else:
        print("  => Pitch is too irregular to count reliably. Insert ONE")
        print("     <mask> per gap, and state the CV as the measured reason.")
    print("\n  Either way the count is an ESTIMATE. Domain 2 sees <mask>, which")
    print("  is an honest 'a sign was here and it is unknown' -- inserting a")
    print("  wrong NUMBER of masks costs alignment, not a wrong sign identity.")


def main() -> None:
    rule("0. LOADING BOX ANNOTATIONS")
    bb = load_boxes()
    res = measure(bb)
    validate_against_anchors(bb, res["grouped"])
    gap_simulation(res["pitch"], res["cv"])

    rule("WHAT THIS DECIDES")
    print("  section 1 -> the median pitch is the unit the inserter counts in.")
    print("  section 2 -> whether the line grouping can be trusted at all.")
    print("               If it splits or merges, sections 1 and 3 are void.")
    print("  section 3 -> count masks, or insert exactly one per gap.")
    print("=" * 70)


if __name__ == "__main__":
    main()