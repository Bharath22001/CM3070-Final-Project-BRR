"""
Checks whether a simple y-projection can split live detections into segments.

On the annotated boxes it can, but the detector also reads faces and fragments
the annotation never covered, sometimes side by side in the same y band. So I
cluster the detection centres with connected components on a coarse grid (the
cell size scales with the median box width) and count cluster pairs that
overlap in y but not in x. Any such pair means y-projection is not enough.
Section E sweeps the cell size to show the answer doesn't depend on it.

Run from the project root:
    python -m cuneiform.domain1_vision.inspect_live_segments
    python -m cuneiform.domain1_vision.inspect_live_segments --conf 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque

import numpy as np

from cuneiform import config

RULE = "=" * 74


def head(title: str) -> None:
    print("\n" + RULE)
    print(title)
    print(RULE)


def verdict(ok, good: str, bad: str) -> str:
    # bool() rather than `ok is True`, since numpy.bool_(True) is not True.
    return f"  => {'PASS. ' + good if bool(ok) else 'FAIL. ' + bad}"


# Loading

def live_tablets() -> list[str]:
    """Tablet ids with a saved detection file (_summary.json is not one)."""
    out = []
    for p in sorted(config.D1_LIVE_DETECTIONS.glob("*.json")):
        if p.stem.startswith("_"):
            continue
        out.append(p.stem)
    return out


def load_live(tablet: str) -> dict:
    with (config.D1_LIVE_DETECTIONS / f"{tablet}.json").open(
            "r", encoding="utf-8") as fh:
        return json.load(fh)


# Grid connected components

def grid_components(cx: np.ndarray, cy: np.ndarray, cell: float) -> np.ndarray:
    """
    Label each point by its 8-connected component on a grid of the given cell
    size, so points join if a chain of occupied cells links them.
    """
    if len(cx) == 0:
        return np.zeros(0, dtype=int)

    gx = np.floor(cx / cell).astype(int)
    gy = np.floor(cy / cell).astype(int)

    occupied: dict[tuple[int, int], list[int]] = {}
    for i in range(len(cx)):
        occupied.setdefault((int(gx[i]), int(gy[i])), []).append(i)

    labels = np.full(len(cx), -1, dtype=int)
    seen: set[tuple[int, int]] = set()
    comp = 0

    for key in occupied:
        if key in seen:
            continue
        queue = deque([key])
        seen.add(key)
        while queue:
            c = queue.popleft()
            for i in occupied[c]:
                labels[i] = comp
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in occupied and n not in seen:
                        seen.add(n)
                        queue.append(n)
        comp += 1

    return labels


# Per-tablet analysis

def analyse(tablet: str, bb, conf: float, cell_factor: float,
            min_size: int) -> dict:
    rec = load_live(tablet)
    dets = [d for d in rec["detections"] if d["conf"] >= conf]

    g = bb[bb["tablet_CDLI"] == tablet]
    n_segments_annotated = int(g["segm_idx"].nunique()) if len(g) else 0

    if not dets:
        return {"tablet": tablet, "n_det": 0, "n_clusters": 0,
                "n_small": 0, "n_det_small": 0,
                "n_seg_ann": n_segments_annotated, "pairs_y_disjoint": 0,
                "pairs_x_sep": 0, "pairs_tangled": 0, "n_unann": 0,
                "n_det_unann": 0, "clusters": []}

    cx = np.array([d["cx"] for d in dets], dtype=float)
    cy = np.array([d["cy"] for d in dets], dtype=float)
    w = np.array([d["x2"] - d["x1"] for d in dets], dtype=float)

    cell = max(float(np.median(w)) * cell_factor, 1.0)
    labels = grid_components(cx, cy, cell)

    # Cluster extents come from the member boxes, not their centres.
    clusters = []
    for lab in np.unique(labels):
        m = labels == lab
        members = [dets[i] for i in np.flatnonzero(m)]
        clusters.append({
            "n": int(m.sum()),
            "x1": min(d["x1"] for d in members),
            "y1": min(d["y1"] for d in members),
            "x2": max(d["x2"] for d in members),
            "y2": max(d["y2"] for d in members),
        })

    small = [c for c in clusters if c["n"] < min_size]
    kept = [c for c in clusters if c["n"] >= min_size]
    kept.sort(key=lambda c: (c["y1"], c["x1"]))

    # Pairwise relations among the kept clusters.
    y_disjoint = x_sep = tangled = 0
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            a, b = kept[i], kept[j]
            y_over = not (a["y2"] <= b["y1"] or b["y2"] <= a["y1"])
            x_over = not (a["x2"] <= b["x1"] or b["x2"] <= a["x1"])
            if not y_over:
                y_disjoint += 1
            elif not x_over:
                x_sep += 1        # a y-projection would merge these
            else:
                tangled += 1      # neither projection separates them

    # Clusters with no annotated box: surfaces the detector read but nobody annotated.
    n_unann = n_det_unann = 0
    if len(g):
        agx = g["cx"].to_numpy(float)
        agy = g["cy"].to_numpy(float)
        for c in kept:
            inside = ((agx >= c["x1"]) & (agx <= c["x2"]) &
                      (agy >= c["y1"]) & (agy <= c["y2"])).sum()
            if inside == 0:
                n_unann += 1
                n_det_unann += c["n"]

    return {
        "tablet": tablet, "n_det": len(dets), "n_clusters": len(kept),
        "n_small": len(small), "n_det_small": sum(c["n"] for c in small),
        "n_seg_ann": n_segments_annotated,
        "pairs_y_disjoint": y_disjoint, "pairs_x_sep": x_sep,
        "pairs_tangled": tangled,
        "n_unann": n_unann, "n_det_unann": n_det_unann,
        "clusters": kept,
    }


# Main

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection confidence threshold for the main table")
    ap.add_argument("--cell", type=float, default=2.0,
                    help="grid cell size as a multiple of median box width")
    ap.add_argument("--min-size", type=int, default=5,
                    help="clusters smaller than this are dropped as debris")
    ap.add_argument("--tablets", nargs="*", default=None)
    args = ap.parse_args()

    from cuneiform.domain1_vision.group_lines import load_boxes

    head("A. INVENTORY")
    tablets = args.tablets if args.tablets else live_tablets()
    if not tablets:
        print(f"  No detection files in {config.D1_LIVE_DETECTIONS}")
        print("  Run run_yolo_inference.py first.")
        return

    bb = load_boxes()
    print(f"  detection files : {len(tablets)}")
    print(f"  conf threshold  : {args.conf}")
    print(f"  grid cell       : {args.cell} x median box width")
    print(f"  min cluster size: {args.min_size} detections")

    rows = [analyse(t, bb, args.conf, args.cell, args.min_size)
            for t in tablets]

    # B
    head("B. CLUSTERS vs ANNOTATED SEGMENTS")
    print("  'ann seg' is the segment count implied by the CompVis annotation.")
    print("  'clusters' is what the LIVE detections actually form.\n")
    print(f"  {'tablet':<12} {'det':>6} {'ann seg':>8} {'clusters':>9} "
          f"{'dropped':>8} {'y-disj':>7} {'x-SEP':>6} {'tangl':>6}")
    for r in rows:
        print(f"  {r['tablet']:<12} {r['n_det']:>6} {r['n_seg_ann']:>8} "
              f"{r['n_clusters']:>9} {r['n_det_small']:>8} "
              f"{r['pairs_y_disjoint']:>7} {r['pairs_x_sep']:>6} "
              f"{r['pairs_tangled']:>6}")

    tot_x = sum(r["pairs_x_sep"] for r in rows)
    tot_t = sum(r["pairs_tangled"] for r in rows)
    n_more = sum(1 for r in rows if r["n_clusters"] > max(1, r["n_seg_ann"]))
    print(f"\n  tablets with MORE clusters than annotated segments: "
          f"{n_more} of {len(rows)}")

    # C: the gate
    head("C. THE GATE -- is a y-projection sufficient on LIVE detections?")
    print(f"  pairs separable by y alone      : "
          f"{sum(r['pairs_y_disjoint'] for r in rows)}")
    print(f"  pairs that OVERLAP in y, x-sep  : {tot_x}   <- y-projection MERGES")
    print(f"  pairs tangled in both axes      : {tot_t}")
    bad = [r["tablet"] for r in rows if r["pairs_x_sep"] or r["pairs_tangled"]]
    print(f"  tablets affected                : {len(bad)} of {len(rows)}")
    if bad:
        print(f"    {', '.join(bad)}")
    print(verdict(
        tot_x == 0 and tot_t == 0,
        "A y-projection recovers segments on the live population. Keep the "
        "y-projection.",
        "A y-projection would merge surfaces that are side by side. The "
        "segment step must be 2D clustering with a minimum-size filter."))

    # Show the evidence for the worst tablets.
    worst = sorted(rows, key=lambda r: -(r["pairs_x_sep"] + r["pairs_tangled"]))
    for r in worst[:3]:
        if not (r["pairs_x_sep"] or r["pairs_tangled"]):
            break
        print(f"\n  {r['tablet']} cluster extents (x1,y1,x2,y2) and size:")
        for c in r["clusters"]:
            print(f"    n={c['n']:>5}  "
                  f"({c['x1']:>7.0f},{c['y1']:>7.0f})-"
                  f"({c['x2']:>7.0f},{c['y2']:>7.0f})")

    # D
    head("D. SURFACES THE ANNOTATION NEVER COVERED")
    print("  A cluster holding ZERO annotated box centres is a surface the")
    print("  detector read and CompVis did not mark. This is the overlay")
    print("  finding as a number.\n")
    print(f"  {'tablet':<12} {'clusters':>9} {'unannotated':>12} "
          f"{'det in those':>13} {'share':>7}")
    for r in rows:
        share = r["n_det_unann"] / max(1, r["n_det"])
        print(f"  {r['tablet']:<12} {r['n_clusters']:>9} {r['n_unann']:>12} "
              f"{r['n_det_unann']:>13} {100 * share:>6.1f}%")
    td, tu = sum(r["n_det"] for r in rows), sum(r["n_det_unann"] for r in rows)
    print(f"\n  pooled: {tu:,} of {td:,} detections ({100 * tu / max(1, td):.1f}%) "
          f"sit on unannotated surfaces")

    # E
    head("E. PARAMETER SENSITIVITY -- is the cell size load-bearing?")
    print("  If the cluster count swings with the cell size, single-link is the")
    print("  wrong method here and this whole section is void, exactly as in")
    print("  inspect_box_pitch.py. Stability is the licence to use it.\n")
    print(f"  {'cell':>6} {'conf':>6} {'median clusters':>16} "
          f"{'total x-sep':>12} {'total tangled':>14}")
    for cf in (1.0, 1.5, 2.0, 3.0, 4.0):
        for cv in (0.15, 0.25, 0.50):
            rs = [analyse(t, bb, cv, cf, args.min_size) for t in tablets]
            med = float(np.median([r["n_clusters"] for r in rs]))
            print(f"  {cf:>6.1f} {cv:>6.2f} {med:>16.1f} "
                  f"{sum(r['pairs_x_sep'] for r in rs):>12} "
                  f"{sum(r['pairs_tangled'] for r in rs):>14}")

    print("\n" + RULE)


if __name__ == "__main__":
    main()