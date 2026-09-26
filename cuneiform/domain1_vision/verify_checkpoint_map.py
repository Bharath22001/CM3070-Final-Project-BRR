"""
Re-runs validation on my deployed checkpoint to see whether it reproduces its
recorded mAP50 of 0.2539, using the local yolo_dataset.

I write a temporary data yaml whose class count matches the checkpoint's head
instead of reusing the dataset's own yaml. No dataset file is changed.

Run from the project root:
    python -m cuneiform.domain1_vision.verify_checkpoint_map
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import yaml
from PIL import Image, ImageDraw

from cuneiform import config

TILE = config.D1_TILE_SIZE
RULE = "=" * 70
RECORDED_MAP50 = 0.2539
RECORDED_MAP5095 = 0.1538
RECORDED_RECALL = 0.243
RECORDED_PRECISION = 0.457


def head(t: str) -> None:
    print("\n" + RULE); print(t); print(RULE)


def verdict(name: str, ok: bool | None, note: str = "") -> None:
    if ok is not None:
        ok = bool(ok)
    print(f"\n  --> {'PASS' if ok is True else 'FAIL' if ok is False else 'VACUOUS'}: {name}")
    if note:
        print(f"      {note}")


def survey_labels(split: str):
    """Count label files and classes, and how many files a smaller nc would reject."""
    d = config.YOLO_DIR / "labels" / split
    if not d.exists():
        return None
    n_files = n_boxes = n_rejected = 0
    classes = Counter()
    wh = []
    for p in d.glob("*.txt"):
        n_files += 1
        bad = False
        for line in p.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            classes[c] += 1
            n_boxes += 1
            wh.append((float(parts[3]) * TILE, float(parts[4]) * TILE))
            if c >= config.D1_CHECKPOINT_NC:
                bad = True
        n_rejected += bad
    return {"files": n_files, "boxes": n_boxes, "rejected_files": n_rejected,
            "classes": classes, "wh": np.asarray(wh) if wh else None}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(RULE)
    print("DOMAIN 1 -- CAN WE REPRODUCE mAP50 = 0.2539?")
    print(RULE)

    # A. The local dataset
    head("A. WHAT IS IN THE LOCAL yolo_dataset?")

    for split in ("train", "val"):
        s = survey_labels(split)
        if s is None:
            print(f"  {split}: labels directory missing")
            continue
        imgs = len(list((config.YOLO_DIR / "images" / split).glob("*.jpg"))) \
            if (config.YOLO_DIR / "images" / split).exists() else 0
        hi = max(s["classes"]) if s["classes"] else -1
        print(f"  {split:<6} images {imgs:>6} | label files {s['files']:>6} | "
              f"boxes {s['boxes']:>7} | distinct classes {len(s['classes']):>4} | "
              f"max class {hi:>4}")
        print(f"         files rejected by nc={config.D1_CHECKPOINT_NC}: "
              f"{s['rejected_files']}")
        if s["wh"] is not None:
            print(f"         median box {np.median(s['wh'][:, 0]):.0f} x "
                  f"{np.median(s['wh'][:, 1]):.0f} px")

    # Expected: 730 train tiles / 7,450 boxes, 289 val / 4,422.
    s_val = survey_labels("val")
    s_tr = survey_labels("train")
    if s_val and s_tr:
        match = (abs(s_tr["files"] - 730) <= 5 and abs(s_val["files"] - 289) <= 5
                 and abs(s_val["boxes"] - 4422) <= 50)
        verdict("the local dataset matches the recorded one", match,
                f"train {s_tr['files']} tiles / {s_tr['boxes']} boxes, "
                f"val {s_val['files']} / {s_val['boxes']}, against the recorded "
                f"730/7,450 and 289/4,422."
                + ("" if match else
                   " These DO NOT match. The local dataset has been "
                   "regenerated since the Colab run, so any val() result below "
                   "is measured on different data than the model trained on."))

    # B. Validate with a correct yaml
    head(f"B. VALIDATING WITH A CORRECT nc={config.D1_CHECKPOINT_NC} YAML")

    yaml_path = config.D1_DIR / f"val_nc{config.D1_CHECKPOINT_NC}.yaml"
    spec = {
        "path": str(config.YOLO_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": config.D1_CHECKPOINT_NC,
        "names": {i: f"class_{i}" for i in range(config.D1_CHECKPOINT_NC)},
    }
    config.D1_DIR.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)
    print(f"  wrote {yaml_path}  (nc={config.D1_CHECKPOINT_NC}, "
          f"images unchanged)")
    print("  NOTE: YOLO's 'val' split IS the CompVis TEST split.\n")

    from ultralytics import YOLO
    model = YOLO(str(config.D1_WEIGHTS))
    print(f"  head nc = {int(getattr(model.model, 'nc', -1))}")
    print("  running val() on CPU -- this takes a couple of minutes...\n")

    m = model.val(data=str(yaml_path), imgsz=TILE, batch=8,
                  device="cpu", split="val", verbose=False, plots=False)

    got50 = float(m.box.map50)
    got5095 = float(m.box.map)
    got_p = float(m.box.mp)
    got_r = float(m.box.mr)

    print(f"    {'metric':<14} {'recorded':>10} {'measured':>10} {'delta':>10}")
    for name, rec, got in (("mAP50", RECORDED_MAP50, got50),
                           ("mAP50-95", RECORDED_MAP5095, got5095),
                           ("precision", RECORDED_PRECISION, got_p),
                           ("recall", RECORDED_RECALL, got_r)):
        print(f"    {name:<14} {rec:>10.4f} {got:>10.4f} {got - rec:>+10.4f}")

    close = abs(got50 - RECORDED_MAP50) <= 0.03
    verdict("the checkpoint reproduces its recorded mAP50", close,
            f"measured {got50:.4f} against a recorded {RECORDED_MAP50:.4f}. "
            f"The weights and the data are the recorded ones, so "
            f"the giant boxes at predict() time are a USAGE difference between "
            f"val() and predict(), not a broken checkpoint. Investigate that "
            f"next before building on the live detections."
            if close else
            f"measured {got50:.4f} against a recorded {RECORDED_MAP50:.4f}. "
            f"This checkpoint does NOT reproduce its recorded number on "
            f"this data. Either the weights or the local dataset is not what "
            f"I think it is. Every Domain 1 figure in the report "
            f"depends on resolving this.")

    # C. Look at one tile
    head("C. ONE VAL TILE, PREDICTIONS AGAINST ITS OWN LABELS")

    val_imgs = sorted((config.YOLO_DIR / "images" / "val").glob("*.jpg"))
    if not val_imgs:
        print("  no val images on disk.")
        return

    best, best_n = None, -1
    for p in val_imgs[:200]:
        lp = config.YOLO_DIR / "labels" / "val" / f"{p.stem}.txt"
        if lp.exists():
            n = len(lp.read_text().strip().splitlines())
            if n > best_n:
                best, best_n = p, n
    if best is None:
        print("  no val labels on disk.")
        return

    print(f"  tile {best.stem} with {best_n} ground-truth boxes\n")
    r = model.predict(source=str(best), imgsz=TILE, conf=0.01, iou=0.7,
                      max_det=1000, device="cpu", verbose=False)[0]

    img = Image.open(best).convert("RGB")
    dr = ImageDraw.Draw(img)
    for line in (config.YOLO_DIR / "labels" / "val" / f"{best.stem}.txt") \
            .read_text().strip().splitlines():
        q = [float(v) for v in line.split()[:5]]
        xc, yc, w, h = q[1] * TILE, q[2] * TILE, q[3] * TILE, q[4] * TILE
        dr.rectangle([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2],
                     outline=(0, 220, 0), width=2)

    print(f"    {'#':<3} {'conf':>6} {'cls':>5} {'x1':>6} {'y1':>6} "
          f"{'x2':>6} {'y2':>6} {'w':>5} {'h':>5}")
    if r.boxes is not None and len(r.boxes):
        xy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        cl = r.boxes.cls.cpu().numpy().astype(int)
        for i in np.argsort(-cf)[:10]:
            x1, y1, x2, y2 = xy[i]
            print(f"    {i:<3} {cf[i]:>6.3f} {cl[i]:>5} {x1:>6.0f} {y1:>6.0f} "
                  f"{x2:>6.0f} {y2:>6.0f} {x2 - x1:>5.0f} {y2 - y1:>5.0f}")
            dr.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=2)

    out = config.D1_DIR / f"{best.stem}_tile_overlay.jpg"
    img.save(out, quality=92)
    print(f"\n  overlay: {out}")
    print("  Green = this tile's own labels. Red = predictions. One 512px tile,")
    print("  no stitching, no crop of mine -- the simplest possible surface.")

    print("\n" + RULE)


if __name__ == "__main__":
    main()