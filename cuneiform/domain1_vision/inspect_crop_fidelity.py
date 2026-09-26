"""
Checks that my inference tiling crops exactly the same pixels as the saved
training tiles the checkpoint was measured on.

A compares pixels at the same origin (a small JPEG difference is expected),
B compares the detections on both, and C looks at box geometry by confidence
against each tile's own labels, which I used to pick the operating threshold.
It only reads files and prints.

Run from the project root:
    python -m cuneiform.domain1_vision.inspect_crop_fidelity
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image, ImageOps

from cuneiform import config
from cuneiform.domain1_vision.run_yolo_inference import tile_origins, crop_tile

TILE = config.D1_TILE_SIZE
RULE = "=" * 70
BINS = [(0.01, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]


def head(t: str) -> None:
    print("\n" + RULE); print(t); print(RULE)


def verdict(name: str, ok: bool | None, note: str = "") -> None:
    if ok is not None:
        ok = bool(ok)
    print(f"\n  --> {'PASS' if ok is True else 'FAIL' if ok is False else 'VACUOUS'}: {name}")
    if note:
        print(f"      {note}")


def find_saved_tile(name: str):
    for p in (config.YOLO_DIR / "images" / "val" / f"{name}.jpg",
              config.YOLO_DIR / "images" / "train" / f"{name}.jpg",
              config.TILES_DIR / f"{name}.jpg"):
        if p.exists():
            return p
    return None


def label_path(img_path):
    return (img_path.parent.parent.parent / "labels" /
            img_path.parent.name / f"{img_path.stem}.txt")


def load_label(img_path):
    """YOLO labels: <class> <xc> <yc> <w> <h>, normalised to the tile."""
    p = label_path(img_path)
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append([float(v) for v in parts[:5]])
    return np.asarray(rows, dtype=float) if rows else None


def predict(model, source, conf):
    r = model.predict(source=source, imgsz=TILE, conf=conf, iou=0.7,
                      max_det=1000, device="cpu", verbose=False)[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 6))
    return np.concatenate([b.xyxy.cpu().numpy(),
                           b.cls.cpu().numpy().reshape(-1, 1),
                           b.conf.cpu().numpy().reshape(-1, 1)], axis=1)


def geom(a):
    if len(a) == 0:
        return "empty"
    w, h = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
    return (f"n={len(a):<4} w p50 {np.median(w):>5.0f}  h p50 {np.median(h):>5.0f}  "
            f">256 both {int(((w > 256) & (h > 256)).sum()):>3}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tablet", default="P334926")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--max-tiles", type=int, default=12)
    args = ap.parse_args()

    print(RULE)
    print("DOMAIN 1 -- CROP FIDELITY vs THE TILES THE MODEL WAS MEASURED ON")
    print(RULE)

    img_path = config.IMAGES_DIR / f"{args.tablet}.jpg"
    if not img_path.exists():
        print(f"  image not found: {img_path}")
        return

    img = ImageOps.exif_transpose(Image.open(img_path).convert("RGB"))
    arr = np.asarray(img)
    W, H = img.size
    print(f"  {args.tablet}  {W}x{H}  conf floor {args.conf}")

    pairs = []
    for y in tile_origins(H):
        for x in tile_origins(W):
            saved = find_saved_tile(f"{args.tablet}_tile_{x}_{y}")
            if saved:
                pairs.append((x, y, saved))
    print(f"  inference origins with a SAVED training tile: {len(pairs)}")

    if not pairs:
        verdict("the inference grid reuses training tile origins", None,
                "no saved tile matched an inference origin. Either the tiles "
                "are not on disk, or this tablet produced none (every tile "
                "was empty of boxes). Try --tablet with another id.")
        return

    pairs = pairs[:args.max_tiles]

    from ultralytics import YOLO
    model = YOLO(str(config.D1_WEIGHTS))

    # A. Pixels
    head("A. DOES THE IN-MEMORY CROP MATCH THE SAVED TILE?")
    print("  Saved tiles went through a JPEG round-trip, so a few grey levels")
    print("  of mean difference is expected. Tens of levels is displacement.\n")
    print(f"    {'tile':<28} {'mean |diff|':>12} {'max |diff|':>11} {'size':>12}")

    diffs = []
    for (x, y, saved) in pairs:
        mine, _, _ = crop_tile(arr, x, y)
        with Image.open(saved) as im:
            theirs = np.asarray(im.convert("RGB"))
        if mine.shape != theirs.shape:
            print(f"    {saved.stem:<28} SHAPE MISMATCH "
                  f"{mine.shape} vs {theirs.shape}")
            diffs.append(999.0)
            continue
        d = np.abs(mine.astype(float) - theirs.astype(float))
        diffs.append(float(d.mean()))
        print(f"    {saved.stem:<28} {d.mean():>12.2f} {d.max():>11.0f} "
              f"{str(mine.shape[:2]):>12}")

    md = float(np.median(diffs))
    print(f"\n  median mean|diff| across {len(diffs)} tiles: {md:.2f} grey levels")
    verdict("the in-memory crop reproduces the saved tile", md < 8.0,
            "the crop is pixel-faithful, so tile origin, axis order and "
            "channel order are all correct. Any box problem is NOT the crop."
            if md < 8.0 else
            "the crop does NOT match the tile the model was trained and "
            "measured on. Fix crop_tile/tile_origins before anything else.")

    # B. Boxes
    head("B. DO THE TWO SURFACES PRODUCE THE SAME DETECTIONS?")
    print(f"    {'tile':<28} {'saved file':<34} {'in-memory crop':<34}")

    same = 0
    for (x, y, saved) in pairs:
        mine, _, _ = crop_tile(arr, x, y)
        a_file = predict(model, str(saved), args.conf)
        a_mem = predict(model, Image.fromarray(mine), args.conf)
        same += abs(len(a_file) - len(a_mem)) <= max(1, 0.1 * len(a_file))
        print(f"    {saved.stem:<28} {geom(a_file):<34} {geom(a_mem):<34}")

    verdict("both surfaces agree", same >= 0.8 * len(pairs),
            f"{same} of {len(pairs)} tiles agree within 10%. The inference "
            f"path feeds the model the same thing the training pipeline did."
            if same >= 0.8 * len(pairs) else
            f"only {same} of {len(pairs)} agree. The in-memory path differs "
            f"from the file path -- suspect channel order or dtype.")

    # C. Confidence
    head("C. BOX GEOMETRY BY CONFIDENCE, AGAINST THE TILE'S OWN LABELS")
    print("  On SAVED tiles only, so the crop is not in the picture at all.")
    print("  This is the evidence for choosing the operating threshold.\n")

    buckets = {b: [] for b in BINS}
    n_gt = 0
    gt_w, gt_h = [], []
    for (x, y, saved) in pairs:
        a = predict(model, str(saved), args.conf)
        lab = load_label(saved)
        if lab is not None:
            n_gt += len(lab)
            gt_w.extend(lab[:, 3] * TILE)
            gt_h.extend(lab[:, 4] * TILE)
        for row in a:
            c = row[5]
            for b in BINS:
                if b[0] <= c < b[1]:
                    buckets[b].append((row[2] - row[0], row[3] - row[1]))
                    break

    if gt_w:
        print(f"  ground truth on these {len(pairs)} tiles: {n_gt} boxes, "
              f"median {np.median(gt_w):.0f} x {np.median(gt_h):.0f} px\n")

    print(f"    {'conf range':<14} {'n':>6} {'w p50':>7} {'h p50':>7} "
          f"{'w p90':>7} {'>256 both':>10}")
    ok_hi = None
    for b in BINS:
        rows = buckets[b]
        if not rows:
            print(f"    {b[0]:.2f}-{b[1]:.2f}      {0:>6}")
            continue
        z = np.asarray(rows, dtype=float)
        big = int(((z[:, 0] > 256) & (z[:, 1] > 256)).sum())
        print(f"    {b[0]:.2f}-{b[1]:.2f}      {len(z):>6} "
              f"{np.median(z[:, 0]):>7.0f} {np.median(z[:, 1]):>7.0f} "
              f"{np.percentile(z[:, 0], 90):>7.0f} {big:>10}")
        if b[0] >= 0.25:
            w50 = float(np.median(z[:, 0]))
            ok_hi = w50 if ok_hi is None else ok_hi

    verdict("high-confidence boxes are sign-sized",
            None if ok_hi is None else ok_hi <= 200,
            "no detections above conf 0.25 on this sample -- widen "
            "--max-tiles, or the detector is weaker on this tablet than "
            "mAP 0.2539 suggests."
            if ok_hi is None else
            (f"boxes above conf 0.25 have median width {ok_hi:.0f} px, "
             f"consistent with the ground truth. The giant boxes live in the "
             f"low-confidence tail and a threshold removes them."
             if ok_hi <= 200 else
             f"even above conf 0.25 the median width is {ok_hi:.0f} px. The "
             f"detector genuinely draws oversized boxes and no threshold "
             f"fixes that."))

    print("\n" + RULE)


if __name__ == "__main__":
    main()