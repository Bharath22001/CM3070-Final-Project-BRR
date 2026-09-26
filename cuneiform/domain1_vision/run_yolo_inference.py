"""
Runs my trained YOLOv8s detector over full tablet photographs and saves the
detections in tablet coordinates.

The tile grid matches generate_tiles.py but adds a flush final tile on each
axis, because the training grid stops short of the right and bottom edges and
about 4% of signs would otherwise never be seen. Images smaller than a tile are
edge-padded, not resized. The class id the model predicts is the train_label
directly, and I detect at a low confidence floor so the operating threshold can
be applied later. Cross-tile NMS runs before any thresholding, which gives the
same result as thresholding first.

Run from the project root:
    python -m cuneiform.domain1_vision.run_yolo_inference --limit 1
    python -m cuneiform.domain1_vision.run_yolo_inference --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from cuneiform import config

TILE = config.D1_TILE_SIZE
STEP = config.D1_TILE_STEP
RULE = "=" * 70


# Tablet selection

def runnable_tablets():
    """
    A tablet is runnable if it has bbox ground truth, an image on disk and is in
    Domain 2's test split. Returns (list, boxes_df).
    """
    from cuneiform.domain1_vision.group_lines import load_boxes

    bb = load_boxes()
    annotated = set(bb["tablet_CDLI"].unique())

    test_split = set()
    with config.D2_CORPUS_JSONL.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                rec = json.loads(raw)
                if rec.get("split") == "test":
                    test_split.add(rec["tablet"])

    have_img = {t for t in annotated if (config.IMAGES_DIR / f"{t}.jpg").exists()}
    return sorted(annotated & have_img & test_split), bb


# The covering tile grid

def tile_origins(dim: int, tile: int = TILE, step: int = STEP) -> list[int]:
    """
    Tile origins along one axis that cover the full extent, adding a flush
    final origin at dim - tile when the stride stops short.
    """
    if dim <= tile:
        return [0]
    origins = list(range(0, dim - tile + 1, step))
    flush = dim - tile
    if origins[-1] != flush:
        origins.append(flush)
    return origins


def crop_tile(arr: np.ndarray, x: int, y: int, tile: int = TILE):
    """
    Crop one tile, edge-padding the bottom and right if the image runs out.
    Returns (tile, valid_w, valid_h); detections centred in the padding are
    dropped later. I replicate edges because a black border would look like a
    strong edge and attract false detections.
    """
    H, W = arr.shape[:2]
    sub = arr[y:min(y + tile, H), x:min(x + tile, W)]
    valid_h, valid_w = sub.shape[0], sub.shape[1]
    pad_h, pad_w = tile - valid_h, tile - valid_w
    if pad_h or pad_w:
        sub = np.pad(sub, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return sub, valid_w, valid_h


# Cross-tile NMS

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """
    Class-agnostic greedy NMS, returning the indices to keep, highest score first.

    Two overlapping boxes with different classes are one sign with two guesses,
    so I keep only one; per-class NMS would inflate the sign count and throw off
    the gap counting. The cost is losing the runner-up class at each position.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]

        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)

        order = rest[iou <= iou_thr]

    return keep


# Inference over one tablet

def detect_tablet(model, tablet_id: str, conf: float, nms_iou: float,
                  batch: int = 8, verbose: bool = True,
                  img_path=None) -> dict:
    """
    Detect signs on one tablet photograph.

    `img_path` defaults to the corpus image for `tablet_id`. The dashboard's
    upload route passes its own path so uploads go through exactly this code.
    """
    img_path = (config.IMAGES_DIR / f"{tablet_id}.jpg"
                if img_path is None else Path(img_path))

    # Convert then EXIF-transpose, in the same order as generate_tiles.py, so
    # the boxes are in the same frame as the training annotations.
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    W, H = img.size
    arr = np.asarray(img)

    xs, ys = tile_origins(W), tile_origins(H)
    origins = [(x, y) for y in ys for x in xs]

    tiles, metas = [], []
    for (x, y) in origins:
        sub, vw, vh = crop_tile(arr, x, y)
        tiles.append(Image.fromarray(sub))
        metas.append((x, y, vw, vh))

    raw = []
    n_pad_dropped = 0
    t0 = time.time()

    for s in range(0, len(tiles), batch):
        chunk = tiles[s:s + batch]
        results = model.predict(
            source=chunk,
            imgsz=TILE,
            conf=conf,
            iou=0.7,             # Ultralytics' within-tile NMS
            max_det=1000,        # the default 300 could clip a dense tile at low conf
            device="cpu",
            verbose=False,
        )
        for r, (x, y, vw, vh) in zip(results, metas[s:s + batch]):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            cf = b.conf.cpu().numpy()
            for k in range(len(cls)):
                bx1, by1, bx2, by2 = xyxy[k]
                cx_t, cy_t = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                # Drop anything centred in the padding.
                if cx_t >= vw or cy_t >= vh:
                    n_pad_dropped += 1
                    continue
                raw.append((
                    float(bx1 + x), float(by1 + y),
                    float(min(bx2, vw) + x), float(min(by2, vh) + y),
                    int(cls[k]), float(cf[k]), x, y,
                ))

    secs = time.time() - t0

    dets = []
    if raw:
        boxes = np.asarray([r[:4] for r in raw], dtype=float)
        scores = np.asarray([r[5] for r in raw], dtype=float)
        for i in nms(boxes, scores, nms_iou):
            x1, y1, x2, y2, c, cf, tx, ty = raw[i]
            dets.append({
                "x1": round(x1, 2), "y1": round(y1, 2),
                "x2": round(x2, 2), "y2": round(y2, 2),
                "cx": round((x1 + x2) / 2.0, 2),
                "cy": round((y1 + y2) / 2.0, 2),
                "train_label": c,          # the class id is the train_label
                "conf": round(cf, 4),
                "tile_x": tx, "tile_y": ty,
            })
        dets.sort(key=lambda d: -d["conf"])

    if verbose:
        print(f"    {tablet_id:<12} {W:>5}x{H:<6} tiles {len(tiles):>3} "
              f"raw {len(raw):>5} -> nms {len(dets):>5}  "
              f"pad-dropped {n_pad_dropped:>4}  {secs:>6.1f}s")

    return {
        "tablet": tablet_id,
        "image": str(img_path),
        "image_size": [W, H],
        "tile_size": TILE,
        "tile_step": STEP,
        "n_tiles": len(tiles),
        "tile_origins": [[int(a), int(b)] for a, b in origins],
        "conf_floor": conf,
        "nms_iou": nms_iou,
        "within_tile_iou": 0.7,
        "max_det_per_tile": 1000,
        "device": "cpu",
        "checkpoint": str(config.D1_WEIGHTS),
        "checkpoint_nc": config.D1_CHECKPOINT_NC,
        "n_raw": len(raw),
        # Pre-NMS boxes, kept so the NMS IoU can be swept later.
        # Rows are [x1, y1, x2, y2, cls, conf].
        "raw_detections": [[round(r[0], 1), round(r[1], 1),
                            round(r[2], 1), round(r[3], 1),
                            int(r[4]), round(r[5], 4)] for r in raw],
        "n_pad_dropped": n_pad_dropped,
        "n_detections": len(dets),
        "seconds": round(secs, 2),
        "detections": dets,
    }


# Main

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tablets", nargs="*", default=None,
                    help="explicit tablet ids; default is the runnable set")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N runnable tablets")
    ap.add_argument("--conf", type=float, default=0.01,
                    help="confidence FLOOR, not the operating threshold")
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    config.ensure_dirs()
    config.check_environment()

    print(RULE)
    print("DOMAIN 1 -- LIVE INFERENCE")
    print(RULE)

    if not config.D1_WEIGHTS.exists():
        print(f"  Checkpoint not found: {config.D1_WEIGHTS}")
        print("  Copy the trained best.pt there first.")
        return

    runnable, bb = runnable_tablets()
    targets = args.tablets if args.tablets else runnable
    if args.limit is not None:
        targets = targets[:args.limit]

    print(f"  checkpoint : {config.D1_WEIGHTS.name} "
          f"({config.D1_WEIGHTS.stat().st_size:,} bytes)")
    print(f"  runnable   : {len(runnable)} tablet(s)")
    print(f"  running on : {len(targets)}")
    print(f"  conf floor : {args.conf}   cross-tile NMS IoU: {args.nms_iou}")
    print(f"  output     : {config.D1_LIVE_DETECTIONS}")

    from ultralytics import YOLO
    model = YOLO(str(config.D1_WEIGHTS))
    head_nc = int(getattr(model.model, "nc", -1))
    print(f"\n  detection head nc = {head_nc}")
    if head_nc != config.D1_CHECKPOINT_NC:
        print(f"  WARNING: expected nc = {config.D1_CHECKPOINT_NC}. "
              f"This is not the deployed checkpoint. Stop and check.")

    print(f"\n  {'tablet':<12} {'image':>12} {'tiles':>9} {'raw':>9} "
          f"{'nms':>9} {'pad':>16} {'time':>8}")

    summary, t_start = [], time.time()
    for t in targets:
        try:
            out = detect_tablet(model, t, args.conf, args.nms_iou, args.batch)
        except Exception as exc:
            print(f"    {t:<12} FAILED: {exc}")
            continue

        dest = config.D1_LIVE_DETECTIONS / f"{t}.json"
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)

        confs = np.asarray([d["conf"] for d in out["detections"]]) \
            if out["detections"] else np.zeros(0)
        summary.append({
            "tablet": t,
            "n_gt_boxes": int((bb["tablet_CDLI"] == t).sum()),
            "n_tiles": out["n_tiles"],
            "n_detections": out["n_detections"],
            "by_threshold": {f"{th:.2f}": int((confs >= th).sum())
                             for th in (0.05, 0.10, 0.15, 0.25, 0.35, 0.50)},
            "seconds": out["seconds"],
        })

    total = time.time() - t_start

    # Detection counts at a few thresholds (not recall or precision), to help
    # pick the operating threshold.
    print("\n" + RULE)
    print("DETECTION COUNT vs ANNOTATED BOX COUNT, BY THRESHOLD")
    print(RULE)
    print("  Counts only: no IoU matching, so this is not recall.\n")
    print(f"    {'tablet':<12} {'gt':>6} " +
          " ".join(f"{th:>7}" for th in ("0.05", "0.10", "0.15",
                                         "0.25", "0.35", "0.50")))
    tot_gt = 0
    tot = {k: 0 for k in ("0.05", "0.10", "0.15", "0.25", "0.35", "0.50")}
    for s in summary:
        tot_gt += s["n_gt_boxes"]
        for k in tot:
            tot[k] += s["by_threshold"][k]
        print(f"    {s['tablet']:<12} {s['n_gt_boxes']:>6} " +
              " ".join(f"{s['by_threshold'][k]:>7}" for k in tot))
    print(f"    {'TOTAL':<12} {tot_gt:>6} " +
          " ".join(f"{tot[k]:>7}" for k in tot))
    print("\n    ratio det/gt " + " " * 6 +
          " ".join(f"{tot[k] / max(1, tot_gt):>7.2f}" for k in tot))

    man = config.D1_LIVE_DETECTIONS / "_summary.json"
    with man.open("w", encoding="utf-8") as fh:
        json.dump({"n_tablets": len(summary),
                   "conf_floor": args.conf,
                   "nms_iou": args.nms_iou,
                   "total_seconds": round(total, 1),
                   "per_tablet": summary}, fh, indent=1)

    print(f"\n  {len(summary)} tablet(s) in {total / 60:.1f} min")
    print(f"  wrote {man}")
    print(RULE)


if __name__ == "__main__":
    main()