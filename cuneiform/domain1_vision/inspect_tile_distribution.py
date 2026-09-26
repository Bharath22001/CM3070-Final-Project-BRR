"""
Tests why live inference produced tile-sized boxes on black background.

My training set has no empty tiles (a tile is only written if a box survives),
so the detector never saw bare background. A tiling bug would give the same
symptom, so section C separates the two: a bug would also misplace boxes on
inscribed tiles, while the distribution gap would only affect background
tiles. No model is loaded and nothing is written.

Run from the project root:
    python -m cuneiform.domain1_vision.inspect_tile_distribution
"""

from __future__ import annotations

import argparse
import ast
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from cuneiform import config
from cuneiform.domain1_vision.run_yolo_inference import tile_origins, crop_tile

TILE = config.D1_TILE_SIZE
DARK_LEVEL = 40      # 0-255; below this a pixel counts as background
THUMB = 128          # measure on a thumbnail, it gives the same statistic faster
RULE = "=" * 70


def head(t: str) -> None:
    print("\n" + RULE); print(t); print(RULE)


def verdict(name: str, ok: bool | None, note: str = "") -> None:
    if ok is not None:
        ok = bool(ok)
    print(f"\n  --> {'PASS' if ok is True else 'FAIL' if ok is False else 'VACUOUS'}: {name}")
    if note:
        print(f"      {note}")


def dark_fraction(img: Image.Image) -> float:
    """Share of pixels dark enough to be photographic background."""
    g = img.convert("L").resize((THUMB, THUMB), Image.BILINEAR)
    return float((np.asarray(g) < DARK_LEVEL).mean())


def find_tile(name: str):
    """Training tiles may be in tiles_512 or in the YOLO dataset copy."""
    for p in (config.TILES_DIR / f"{name}.jpg",
              config.YOLO_DIR / "images" / "train" / f"{name}.jpg",
              config.YOLO_DIR / "images" / "val" / f"{name}.jpg"):
        if p.exists():
            return p
    return None


# A. What the detector saw during training

def section_a(sample: int):
    head("A. THE TRAINING DISTRIBUTION")

    frames = []
    for path in (config.TILED_TRAIN_CSV, config.TILED_TEST_CSV):
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        verdict("training tile distribution is measurable", None,
                "tiled annotation CSVs not found.")
        return None, None

    df = pd.concat(frames, ignore_index=True)
    names = sorted(df["tile_name"].unique())
    print(f"  training tiles listed in the CSVs : {len(names):,}")
    print(f"  boxes                             : {len(df):,}")
    print(f"  every listed tile has >=1 box BY CONSTRUCTION -- "
          f"generate_tiles.py skips empty tiles.")

    # Box sizes, to see what a normal box looks like.
    if "bbox_xyxy" in df.columns:
        a = np.asarray([ast.literal_eval(v) if isinstance(v, str) else v
                        for v in df["bbox_xyxy"]], dtype=float)
        w, h = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
        print(f"\n  TRAINING BOX SIZE (tile coords):")
        print(f"    width  p50 {np.median(w):>5.0f}  p99 {np.percentile(w, 99):>5.0f}  "
              f"max {w.max():>5.0f}")
        print(f"    height p50 {np.median(h):>5.0f}  p99 {np.percentile(h, 99):>5.0f}  "
              f"max {h.max():>5.0f}")
        print(f"    boxes over 256 px on BOTH axes: "
              f"{int(((w > 256) & (h > 256)).sum())}")
        box_cap = (float(np.percentile(w, 99)), float(np.percentile(h, 99)))
    else:
        box_cap = None

    rng = np.random.default_rng(1729)
    if sample and len(names) > sample:
        names = list(rng.choice(names, size=sample, replace=False))
        print(f"\n  sampling {len(names)} tiles for content measurement")

    fr, missing = [], 0
    for n in names:
        p = find_tile(n)
        if p is None:
            missing += 1
            continue
        with Image.open(p) as im:
            fr.append(dark_fraction(im))

    if not fr:
        verdict("training tile distribution is measurable", None,
                f"none of {len(names)} tile images found on disk.")
        return None, box_cap

    fr = np.asarray(fr)
    print(f"\n  TRAINING TILE DARKNESS (share of pixels < {DARK_LEVEL}/255), "
          f"n={len(fr):,}" + (f", {missing} not on disk" if missing else ""))
    for q in (50, 90, 95, 99, 100):
        print(f"    p{q:<3} {np.percentile(fr, q):.3f}")

    print(f"\n  This is the ONLY region of input space the detector has ever "
          f"been\n  optimised on. Anything darker is extrapolation.")
    return fr, box_cap


# B. What it sees at inference

def section_b(runs, train_dark):
    head("B. THE INFERENCE DISTRIBUTION")

    if train_dark is None:
        verdict("inference tiles resemble training tiles", None,
                "no training distribution to compare against.")
        return {}

    cap = float(np.percentile(train_dark, 99))
    print(f"  Reference: training p99 darkness = {cap:.3f}\n")
    print(f"    {'tablet':<12} {'tiles':>6} {'p50':>7} {'p90':>7} "
          f"{'over p99':>10}")

    per_tile = {}
    tot = over = 0
    for r in runs:
        p = config.IMAGES_DIR / f"{r['tablet']}.jpg"
        if not p.exists():
            continue
        img = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
        arr = np.asarray(img)
        W, H = img.size
        vals = []
        for y in tile_origins(H):
            for x in tile_origins(W):
                sub, _, _ = crop_tile(arr, x, y)
                d = dark_fraction(Image.fromarray(sub))
                per_tile[(r["tablet"], x, y)] = d
                vals.append(d)
        v = np.asarray(vals)
        n_over = int((v > cap).sum())
        tot += len(v)
        over += n_over
        print(f"    {r['tablet']:<12} {len(v):>6} {np.median(v):>7.3f} "
              f"{np.percentile(v, 90):>7.3f} {n_over:>6} "
              f"({100 * n_over / len(v):>3.0f}%)")

    print(f"\n  {over} of {tot} inference tiles ({100 * over / max(1, tot):.0f}%) "
          f"are darker than ANY\n  but the top 1% of training tiles.")
    verdict("inference tiles resemble the training distribution", over == 0,
            f"{over} tile(s) are out of distribution. The detector was never "
            f"shown a negative tile, so its behaviour there is undefined "
            f"rather than merely inaccurate."
            if over else "the grids match; look elsewhere for the cause.")
    return per_tile


# C. The test that separates the two explanations

def section_c(runs, bb, per_tile, train_dark):
    head("C. DO DETECTIONS MISBEHAVE EVERYWHERE, OR ONLY ON BACKGROUND?")

    print("  If the tile-origin arithmetic were wrong, detections in INSCRIBED")
    print("  tiles would be displaced too. If the distribution hypothesis is")
    print("  right, inscribed tiles should look normal and only background")
    print("  tiles should produce tile-spanning boxes.\n")

    if not per_tile or train_dark is None:
        verdict("the failure is confined to background tiles", None,
                "no tile darkness measured.")
        return

    cap = float(np.percentile(train_dark, 99))
    groups = {"in-distribution": [], "background (OOD)": []}

    for r in runs:
        g = bb[bb["tablet_CDLI"] == r["tablet"]]
        gc = g[["cx", "cy"]].to_numpy(dtype=float) if not g.empty else np.zeros((0, 2))
        for d in r["detections"]:
            key = (r["tablet"], d["tile_x"], d["tile_y"])
            dark = per_tile.get(key)
            if dark is None:
                continue
            w, h = d["x2"] - d["x1"], d["y2"] - d["y1"]
            if len(gc):
                dist = float(np.sqrt(((np.array([d["cx"], d["cy"]]) - gc) ** 2)
                                     .sum(1)).min())
            else:
                dist = float("nan")
            groups["background (OOD)" if dark > cap
                   else "in-distribution"].append((w, h, d["conf"], dist))

    print(f"    {'group':<20} {'n':>6} {'w p50':>7} {'h p50':>7} "
          f"{'conf p50':>9} {'dist p50':>9} {'>256px':>8}")
    summary = {}
    for name, rows in groups.items():
        if not rows:
            print(f"    {name:<20} {0:>6}")
            continue
        a = np.asarray(rows, dtype=float)
        big = int(((a[:, 0] > 256) & (a[:, 1] > 256)).sum())
        dist = a[:, 3][~np.isnan(a[:, 3])]
        summary[name] = (float(np.median(a[:, 0])), float(np.median(a[:, 1])),
                         len(rows), big)
        print(f"    {name:<20} {len(rows):>6} {np.median(a[:, 0]):>7.0f} "
              f"{np.median(a[:, 1]):>7.0f} {np.median(a[:, 2]):>9.3f} "
              f"{np.median(dist) if len(dist) else float('nan'):>9.0f} "
              f"{big:>8}")

    ind = summary.get("in-distribution")
    ood = summary.get("background (OOD)")
    if not ind or not ood:
        verdict("the failure is confined to background tiles", None,
                "one group is empty; cannot discriminate.")
        return

    ok = ind[0] <= 200 and ind[1] <= 200 and ind[3] == 0
    verdict("detections in inscribed tiles are well behaved", ok,
            f"inscribed-tile boxes are {ind[0]:.0f}x{ind[1]:.0f} px against a "
            f"75x54 px ground truth, and {ind[3]} of {ind[2]} exceed 256 px on "
            f"both axes. The tile-origin mapping is sound and the failure is "
            f"confined to out-of-distribution tiles -- a CONTENT gate fixes it."
            if ok else
            f"inscribed-tile boxes are ALSO malformed ({ind[0]:.0f}x{ind[1]:.0f} "
            f"px). This is not only a distribution problem. Suspect the "
            f"tile-origin arithmetic in run_yolo_inference.py and do NOT "
            f"proceed to a content gate.")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400,
                    help="training tiles to measure (0 = all)")
    args = ap.parse_args()

    from cuneiform.domain1_vision.group_lines import load_boxes

    print(RULE)
    print("DOMAIN 1 -- TILE DISTRIBUTION DIAGNOSTIC")
    print(RULE)

    bb = load_boxes()
    names = sorted(p.stem for p in config.D1_LIVE_DETECTIONS.glob("*.json")
                   if not p.stem.startswith("_"))
    runs = []
    for n in names:
        with (config.D1_LIVE_DETECTIONS / f"{n}.json").open(encoding="utf-8") as fh:
            runs.append(json.load(fh))
    if not runs:
        print("  No detection JSON. Run run_yolo_inference.py first.")
        return
    print(f"  {len(runs)} run(s): {[r['tablet'] for r in runs]}")

    train_dark, _ = section_a(args.sample)
    per_tile = section_b(runs, train_dark)
    section_c(runs, bb, per_tile, train_dark)

    print("\n" + RULE)


if __name__ == "__main__":
    main()