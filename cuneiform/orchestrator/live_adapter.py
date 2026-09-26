"""
Turns saved Domain 1 detections into segments, lines, a slot sequence and
finally fusion.Detection objects: the plumbing between Domains 1 and 2.

Segments are found by 2D clustering rather than a y-projection, because some
plates hold surfaces side by side (P334932 has two separate objects), and a
y-projection would merge them and fill the gap between them with fake masks.
Segments are read top to bottom, then left to right, which is a convention:
geometry alone can't tell obverse from reverse.

The defaults here (cell 3.0, conf 0.25, min size 5) are for running this file
on its own. The production pipeline uses cell 2.0, conf 0.15 and min size 3.

Run from the project root:
    python -m cuneiform.orchestrator.live_adapter --tablet P335940
    python -m cuneiform.orchestrator.live_adapter --tablet P334932 --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque

import numpy as np

from cuneiform import config
from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.orchestrator.fusion import Detection
from cuneiform.orchestrator.insert_gaps import (
    DEFAULT_PITCH_RATIO, Slot, insert_masks,
)

# Segment clustering defaults for standalone runs.
SEGMENT_CELL_FACTOR = 3.0
SEGMENT_MIN_SIZE = 5

# Confidence threshold for standalone runs only; the pipeline uses 0.15.
DEFAULT_CONF = 0.25


# Loading

def load_detections(tablet: str, conf: float) -> list[dict]:
    """Read one saved detection file and apply the confidence threshold."""
    path = config.D1_LIVE_DETECTIONS / f"{tablet}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run run_yolo_inference.py first."
        )
    with path.open("r", encoding="utf-8") as fh:
        rec = json.load(fh)
    return [d for d in rec["detections"] if d["conf"] >= conf]


def load_sign_names() -> dict[int, str]:
    """
    train_label to display name, only for the readable dump. Unnamed signs
    fall back to MZL{n}.
    """
    try:
        with config.D3_SIGN_LOOKUP.open("r", encoding="utf-8") as fh:
            lookup = json.load(fh)["lookup"]
    except Exception:
        return {}
    out = {}
    for k, v in lookup.items():
        name = v.get("sign")
        if name is None and v.get("mzl") is not None:
            name = f"MZL{v['mzl']}"
        out[int(k)] = name or f"?{k}"
    return out


# Segmentation

def grid_components(cx: np.ndarray, cy: np.ndarray, cell: float) -> np.ndarray:
    """
    8-connected components of detection centres on a coarse grid. A separate
    copy from the diagnostic script, so production never depends on it.
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


def segment(dets: list[dict], cell_factor: float = SEGMENT_CELL_FACTOR,
            min_size: int = SEGMENT_MIN_SIZE) -> list[np.ndarray]:
    """
    Group detections into physical surfaces and return index arrays in reading
    order. Clusters smaller than min_size are dropped as debris, and the caller
    reports how many detections that cost.
    """
    if not dets:
        return []

    cx = np.array([d["cx"] for d in dets], float)
    cy = np.array([d["cy"] for d in dets], float)
    w = np.array([d["x2"] - d["x1"] for d in dets], float)

    cell = max(float(np.median(w)) * cell_factor, 1.0)
    labels = grid_components(cx, cy, cell)

    groups = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) >= min_size:
            groups.append(idx)

    # Reading order: top to bottom, then left to right (a convention).
    groups.sort(key=lambda ix: (float(cy[ix].min()), float(cx[ix].min())))
    return groups


# Slots

def build_slots(dets: list[dict], groups: list[np.ndarray],
                pitch_ratio: float = DEFAULT_PITCH_RATIO,
                extend_edges: bool = False) -> list[tuple[int, Slot]]:
    """
    Run the line grouper and gap inserter on each segment. Returns
    [(segment_index, Slot)]; the segment index travels alongside because every
    segment numbers its lines from 0.
    """
    out: list[tuple[int, Slot]] = []
    for seg_i, idx in enumerate(groups):
        sub = [dets[i] for i in idx]
        slots = insert_masks(
            np.array([d["cx"] for d in sub], float),
            np.array([d["cy"] for d in sub], float),
            np.array([d["x2"] - d["x1"] for d in sub], float),
            np.array([d["y2"] - d["y1"] for d in sub], float),
            np.array([d["train_label"] for d in sub]),
            np.array([d["conf"] for d in sub], float),
            pitch_ratio, extend_edges=extend_edges,
        )
        out.extend((seg_i, s) for s in slots)
    return out


def to_detections(slots: list[tuple[int, Slot]],
                  tok: SignTokenizer) -> list[Detection]:
    """
    Slot list to fusion.Detection list, applying the emit filter. A box whose
    class is outside the 82 emit classes becomes "unknown", which I keep
    distinct from "missed" (no box at all).
    """
    out = []
    for _, s in slots:
        if s.kind == "inserted" or s.label is None:
            out.append(Detection("missed"))
        elif int(s.label) in tok.vision_emit_classes:
            out.append(Detection("detected", train_label=int(s.label),
                                 confidence=float(s.confidence)))
        else:
            out.append(Detection("unknown"))
    return out


# Inspection dump

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tablet", required=True)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--cell", type=float, default=SEGMENT_CELL_FACTOR)
    ap.add_argument("--min-size", type=int, default=SEGMENT_MIN_SIZE)
    ap.add_argument("--extend-edges", action="store_true",
                    help="off by default; report both figures")
    args = ap.parse_args()

    tok = SignTokenizer()
    names = load_sign_names()
    dets = load_detections(args.tablet, args.conf)
    groups = segment(dets, args.cell, args.min_size)
    kept = sum(len(g) for g in groups)

    print("=" * 74)
    print(f"LIVE ADAPTER -- {args.tablet}")
    print("=" * 74)
    print(f"  detections >= {args.conf}   : {len(dets)}")
    print(f"  segments (cell {args.cell}, min {args.min_size}) : {len(groups)}")
    print(f"  detections kept          : {kept}  "
          f"(dropped as debris: {len(dets) - kept})")
    print(f"  edge extrapolation       : {'ON' if args.extend_edges else 'off'}")

    slots = build_slots(dets, groups, extend_edges=args.extend_edges)
    detections = to_detections(slots, tok)

    n_det = sum(1 for d in detections if d.kind == "detected")
    n_unk = sum(1 for d in detections if d.kind == "unknown")
    n_mis = sum(1 for d in detections if d.kind == "missed")
    n = max(1, len(detections))

    print(f"\n  slots produced           : {len(slots)}")
    print(f"    detected (usable name) : {n_det:>5}  {100*n_det/n:>5.1f}%")
    print(f"    unknown  (box, no name): {n_unk:>5}  {100*n_unk/n:>5.1f}%")
    print(f"    missed   (inserted)    : {n_mis:>5}  {100*n_mis/n:>5.1f}%")
    print(f"    reaching Domain 2 as <mask>: "
          f"{100*(n_unk+n_mis)/n:.1f}%   (simulated figure was 80.4%)")

    # Length comparison only; the sequences aren't aligned. On these 30 tablets
    # only 54.1% of real tokens are both boxed and nameable, so exact agreement
    # isn't the target.
    try:
        real = 0
        with config.D2_CORPUS_JSONL.open("r", encoding="utf-8") as fh:
            for raw in fh:
                rec = json.loads(raw)
                if rec["tablet"] == args.tablet:
                    real += sum(1 for t in rec["tokens"]
                                if t != config.BROKEN_LABEL)
        if real:
            print(f"\n  real corpus tokens       : {real}")
            print(f"  slots / corpus tokens    : {len(slots)/real:.2f}   "
                  f"(length only; sequences are not aligned)")
    except Exception as exc:
        print(f"\n  corpus comparison skipped: {exc}")

    print("\n" + "-" * 74)
    print("  THE SEQUENCE -- read this before scaling up")
    print("-" * 74)
    cur = (None, None)
    line_out: list[str] = []
    for (seg_i, s), d in zip(slots, detections):
        if (seg_i, s.line) != cur:
            if line_out:
                print("    " + " ".join(line_out))
            line_out = []
            cur = (seg_i, s.line)
            print(f"\n  segment {seg_i}  line {s.line}")
        if d.kind == "detected":
            line_out.append(names.get(int(s.label), f"#{s.label}"))
        elif d.kind == "unknown":
            line_out.append("[x]")
        else:
            line_out.append("___")
    if line_out:
        print("    " + " ".join(line_out))
    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()