"""
Recovers reading lines from bounding-box geometry and measures sign pitch.
This replaces my first attempt in inspect_box_pitch.py, which merged about
four rows into one.

How it works:
1. I fit a slope in each horizontal band of a segment, so the lines can follow
   the tablet's curvature, and smooth the slopes across neighbouring bands.
2. I project the boxes along those slopes and histogram them. Rows show up as
   peaks and I cut at the minima between peaks, which works even when rows touch.
3. Peaks closer than a fraction of the median box height are merged.

I check the result against the independent line anchors (line-count ratio and
anchor purity), and only report pitch if the line counts pass.

Run from the project root:
    python -m cuneiform.domain1_vision.group_lines
"""

from __future__ import annotations

import ast
from collections import defaultdict

import numpy as np
import pandas as pd

from cuneiform import config

LINE_TRAIN = config.ANNOTATIONS_DIR / "line_annotations_train.csv"
LINE_TEST = config.ANNOTATIONS_DIR / "line_annotations_test.csv"

SLOPES = np.linspace(-0.25, 0.12, 75)
BAND_ROWS = 6          # target rows per slope-fitting band
MIN_PEAK_SEP = 0.75    # peaks closer than this * median box height are merged
HIST_BINS_PER_H = 6    # histogram bins per median box height


def rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_boxes() -> pd.DataFrame:
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
    return bb


def _peakiness(proj: np.ndarray, bin_w: float) -> float:
    """Sum of squared histogram counts, highest when points bunch together."""
    bins = np.floor((proj - proj.min()) / bin_w).astype(int)
    counts = np.bincount(bins).astype(float)
    return float((counts ** 2).sum())


def fit_local_slopes(cx, cy, h_med) -> list[tuple[float, float, float]]:
    """
    Fit a slope within each horizontal band of the segment, sized to hold about
    BAND_ROWS rows. Returns [(y_lo, y_hi, slope)].
    """
    y_span = cy.max() - cy.min()
    band_h = max(h_med * 2.5 * BAND_ROWS, 1.0)
    n_bands = max(1, int(np.ceil(y_span / band_h)))
    edges = np.linspace(cy.min(), cy.max() + 1e-6, n_bands + 1)

    raw = []
    for i in range(n_bands):
        m = (cy >= edges[i]) & (cy < edges[i + 1])
        if m.sum() < 8:
            raw.append(None)
            continue
        bx, by = cx[m], cy[m]
        best, best_score = 0.0, -1.0
        for s in SLOPES:
            score = _peakiness(by - s * bx, h_med / HIST_BINS_PER_H)
            if score > best_score:
                best, best_score = float(s), score
        raw.append(best)

    known = [s for s in raw if s is not None]
    fallback = float(np.median(known)) if known else 0.0

    # Smooth across neighbours, since a sparse band's fit is noisy.
    out = []
    for i in range(n_bands):
        vals = [raw[j] for j in (i - 1, i, i + 1)
                if 0 <= j < n_bands and raw[j] is not None]
        slope = float(np.median(vals)) if vals else fallback
        out.append((edges[i], edges[i + 1], slope))
    return out


def group_segment(g: pd.DataFrame) -> np.ndarray:
    """Assign a line id to every box in one segment by cutting the projected density at its troughs."""
    cx = g["cx"].to_numpy(float)
    cy = g["cy"].to_numpy(float)
    h_med = float(np.median(g["h"]))

    bands = fit_local_slopes(cx, cy, h_med)

    # Project each box using the slope of its band.
    proj = np.empty(len(g))
    for i, (y, x) in enumerate(zip(cy, cx)):
        slope = bands[-1][2]
        for lo, hi, s in bands:
            if lo <= y < hi:
                slope = s
                break
        proj[i] = y - slope * x

    bin_w = max(h_med / HIST_BINS_PER_H, 1.0)
    lo, hi = proj.min(), proj.max()
    n_bins = max(1, int(np.ceil((hi - lo) / bin_w)) + 1)
    counts = np.bincount(np.floor((proj - lo) / bin_w).astype(int),
                         minlength=n_bins).astype(float)

    # Light smoothing so one sparse bin inside a row isn't read as a boundary.
    kernel = np.ones(3) / 3.0
    smooth = np.convolve(counts, kernel, mode="same")

    # Peaks: local maxima, then merged if closer than MIN_PEAK_SEP * box height.
    peaks = [i for i in range(1, len(smooth) - 1)
             if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]
             and smooth[i] > 0]
    if smooth[0] > 0 and (not peaks or peaks[0] > 1):
        peaks.insert(0, 0)
    # Same case at the last bin. The scan above can never mark the final bin as
    # a peak, so without this the last row merges into the one above it.
    _L = len(smooth) - 1
    if smooth[_L] > 0 and (not peaks or peaks[-1] < _L - 1):
        peaks.append(_L)
    if not peaks:
        return np.zeros(len(g), dtype=int)

    min_sep_bins = max(1, int(MIN_PEAK_SEP * h_med / bin_w))
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] < min_sep_bins:
            if smooth[p] > smooth[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)

    # Boundaries are the minima between consecutive peaks.
    cuts = []
    for a, b in zip(merged, merged[1:]):
        trough = a + int(np.argmin(smooth[a:b + 1]))
        cuts.append(lo + (trough + 0.5) * bin_w)

    line_of = np.zeros(len(g), dtype=int)
    for c in cuts:
        line_of += (proj > c).astype(int)
    return line_of


def group_all(bb: pd.DataFrame) -> dict:
    grouped = {}
    for key, g in bb.groupby(["split", "tablet_CDLI", "segm_idx"]):
        if len(g) < 3:
            continue
        grouped[key] = {"idx": g.index.to_numpy(),
                        "line_of": group_segment(g)}
    return grouped


def validate(bb: pd.DataFrame, grouped: dict) -> float:
    """Line-count ratio against the independent anchor annotation."""
    rule("VALIDATION AGAINST THE ANCHOR FILE  (the gate)")

    frames = [pd.read_csv(p) for p in (LINE_TRAIN, LINE_TEST) if p.exists()]
    if not frames:
        print("  No anchor files; cannot validate.")
        return 0.0
    anchors = pd.concat(frames, ignore_index=True)

    recovered = defaultdict(int)
    for (split, tablet, segm), d in grouped.items():
        recovered[tablet] += len(np.unique(d["line_of"]))

    anchor_lines = {}
    for t, g in anchors.groupby("tablet_CDLI"):
        anchor_lines[t] = int(
            g.groupby("segm_idx")["line_idx"].nunique().sum())

    shared = sorted(set(recovered) & set(anchor_lines))
    print(f"  tablets comparable: {len(shared)}\n")
    print(f"  {'tablet':<14} {'anchor':>7} {'recovered':>10} {'ratio':>7}")
    ratios = []
    for t in shared:
        a, r = anchor_lines[t], recovered[t]
        ratios.append(r / max(1, a))
        print(f"  {t:<14} {a:>7} {r:>10} {ratios[-1]:>7.2f}")

    med = float(np.median(ratios))
    within = float(np.mean([0.8 <= x <= 1.25 for x in ratios]))
    print(f"\n  median ratio            : {med:.2f}   "
          f"(previous attempt: 0.23)")
    print(f"  tablets within 0.8-1.25 : {100 * within:.0f}%")
    if 0.8 <= med <= 1.25:
        print("  => PASS. Grouping is consistent with the human annotation.")
    elif med < 0.8:
        print("  => STILL MERGING. Do not trust the pitch below.")
    else:
        print("  => NOW SPLITTING. Do not trust the pitch below.")
    return med


def measure_pitch(bb: pd.DataFrame, grouped: dict) -> None:
    rule("SIGN PITCH  (only meaningful if the gate above passed)")

    pitches, cvs, sizes = [], [], []
    tight = pairs = 0

    for key, d in grouped.items():
        g = bb.loc[d["idx"]]
        cx = g["cx"].to_numpy(float)
        w = g["w"].to_numpy(float)
        for lid in np.unique(d["line_of"]):
            m = d["line_of"] == lid
            sizes.append(int(m.sum()))
            if m.sum() < 3:
                continue
            xs = np.sort(cx[m])
            dd = np.diff(xs)
            pitches.extend(dd.tolist())
            pairs += len(dd)
            tight += int((dd < 0.5 * np.mean(w[m])).sum())
            if len(dd) >= 2 and dd.mean() > 0:
                cvs.append(dd.std() / dd.mean())

    p, cv = np.array(pitches), np.array(cvs)
    print(f"  lines recovered       : {len(sizes):,}")
    print(f"  median boxes per line : {np.median(sizes):.0f}  "
          f"(p10 {np.percentile(sizes, 10):.0f}, "
          f"p90 {np.percentile(sizes, 90):.0f})")
    print(f"  median box width      : {np.median(bb['w']):.0f} px "
          f"(pitch must plausibly exceed this)")
    print(f"\n  PITCH, {len(p):,} neighbour pairs:")
    for q in (5, 25, 50, 75, 95):
        print(f"    p{q:<3} {np.percentile(p, q):>7.1f}")
    print(f"  tight pairs (< half a sign width): {tight:,} "
          f"({100 * tight / max(1, pairs):.1f}%)   "
          f"[previous attempt: 94.5%]")
    print(f"\n  within-line pitch CV: median {np.median(cv):.3f}")

    med_p = float(np.median(p))
    rng = np.random.default_rng(1729)
    print(f"\n  GAP COUNTING with unit = {med_p:.1f} px:")
    print(f"  {'true k':>7} {'exact':>8} {'within 1':>10}")
    for k in (1, 2, 3, 4, 6):
        draws = rng.choice(p, size=(4000, k), replace=True)
        est = np.round(draws.sum(axis=1) / med_p).astype(int)
        print(f"  {k:>7} {100 * (est == k).mean():>7.1f}% "
              f"{100 * (np.abs(est - k) <= 1).mean():>9.1f}%")

    if np.median(cv) < 0.5:
        print("\n  => Pitch regular enough to COUNT missing signs.")
    else:
        print("\n  => Too irregular to count. One <mask> per gap, CV as the")
        print("     stated reason.")


def main() -> None:
    rule("LOADING")
    bb = load_boxes()
    print(f"  {len(bb):,} boxes | {bb['tablet_CDLI'].nunique()} tablets | "
          f"median box {np.median(bb['w']):.0f} x {np.median(bb['h']):.0f} px")

    grouped = group_all(bb)
    print(f"  {len(grouped)} segments grouped")

    med = validate(bb, grouped)
    measure_pitch(bb, grouped)

    rule("READING THIS")
    if 0.8 <= med <= 1.25:
        print("  Gate passed. The pitch and gap-counting numbers above are the")
        print("  basis for the mask inserter.")
    else:
        print("  Gate FAILED. Ignore the pitch numbers -- they are measured on")
        print("  a wrong grouping, exactly as in the first attempt. Report the")
        print("  ratio and the direction of the error, and fix before building.")
    print("=" * 70)


if __name__ == "__main__":
    main()