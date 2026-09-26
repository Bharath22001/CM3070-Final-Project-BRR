"""
Audits my saved detections against the annotated ground truth. It is not an
accuracy measurement; it checks that the output looks like sign detection at all.

A: box sizes against the ~75x54 px median sign. B: how many classes appear and
how many survive the 82-class emit filter. C: how sensitive the box count is to
the cross-tile NMS IoU. D: whether each box lands near some annotated sign.
E: an overlay JPEG with ground truth and detections drawn together.
No model is loaded.

Run from the project root:
    python -m cuneiform.domain1_vision.inspect_detections
    python -m cuneiform.domain1_vision.inspect_detections --tablets P334926
"""

from __future__ import annotations

import argparse
import json
import sys
import ast

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from cuneiform import config

RULE = "=" * 70

# Median annotated sign size, measured on the training set.
GT_MEDIAN_W = 75.0
GT_MEDIAN_H = 54.0


def head(t: str) -> None:
    print("\n" + RULE)
    print(t)
    print(RULE)


def verdict(name: str, ok: bool | None, note: str = "") -> None:
    # Coerce to a real bool here: numpy comparisons return np.bool_, and
    # `np.bool_(True) is True` is False.
    if ok is not None:
        ok = bool(ok)
    tag = "PASS" if ok is True else ("FAIL" if ok is False else "VACUOUS")
    print(f"\n  --> {tag}: {name}")
    if note:
        print(f"      {note}")


def gt_corner_cols(bb):
    """
    Find the ground-truth corner columns, whatever load_boxes() calls them.
    Returns None if they can't be found, and callers report that as VACUOUS.
    """
    for cols in (("x1", "y1", "x2", "y2"),
                 ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
                 ("xmin", "ymin", "xmax", "ymax")):
        if set(cols).issubset(bb.columns):
            return cols
    return None

def attach_corners(bb):
    """
    Build x1/y1/x2/y2 from load_boxes() output, checking the corner convention
    and cross-checking against cx/cy instead of trusting it.
    """
    if {"x1", "y1", "x2", "y2"}.issubset(bb.columns):
        return bb
    if "bbox" not in bb.columns:
        print("  no 'bbox' column; falling back to centres only.")
        return bb

    vals = bb["bbox"].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else v)
    a = np.asarray(vals.tolist(), dtype=float)

    if not ((a[:, 2] > a[:, 0]).all() and (a[:, 3] > a[:, 1]).all()):
        print("  'bbox' failed the x2>x1 / y2>y1 check -- NOT used as corners.")
        return bb

    mw, mh = np.median(a[:, 2] - a[:, 0]), np.median(a[:, 3] - a[:, 1])
    print(f"  parsed 'bbox' as [x1,y1,x2,y2]: median {mw:.0f} x {mh:.0f} px")
    if not (20 <= mw <= 300 and 15 <= mh <= 300):
        print("  implausible sign size -- convention is being misread. NOT used.")
        return bb

    if {"cx", "cy"}.issubset(bb.columns):
        dx = np.abs((a[:, 0] + a[:, 2]) / 2 - bb["cx"].to_numpy(dtype=float))
        dy = np.abs((a[:, 1] + a[:, 3]) / 2 - bb["cy"].to_numpy(dtype=float))
        print(f"  cross-check vs cx/cy: max centre error "
              f"{dx.max():.2f} x {dy.max():.2f} px")
        if max(dx.max(), dy.max()) > 1.0:
            print("  centres DISAGREE with the parsed corners. NOT used.")
            return bb

    bb = bb.copy()
    bb[["x1", "y1", "x2", "y2"]] = a
    return bb

def load_run(tablet: str) -> dict | None:
    p = config.D1_LIVE_DETECTIONS / f"{tablet}.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def emit_classes() -> set[int] | None:
    p = config.D2_DIR / "vocab.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        v = json.load(fh)
    ec = v.get("vision_emit_classes")
    return set(int(c) for c in ec) if ec else None


# A. Geometry

def section_a(runs, bb, cols) -> None:
    head("A. ARE THE DETECTED BOXES SIGN-SIZED?")

    if cols:
        g = bb[list(cols)].to_numpy(dtype=float)
        gw, gh = g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]
        print(f"  ground truth (all tablets, n={len(g):,}): "
              f"median {np.median(gw):.0f} x {np.median(gh):.0f} px")
    else:
        print(f"  ground truth corner columns not found; using the "
              f"recorded median {GT_MEDIAN_W:.0f} x {GT_MEDIAN_H:.0f} px")
        print(f"  columns available: {sorted(bb.columns)}")

    print(f"\n    {'tablet':<12} {'n':>6} {'w p10':>7} {'w p50':>7} "
          f"{'w p90':>7} {'h p50':>7} {'max w':>7} {'max h':>7}")

    all_w, all_h = [], []
    for r in runs:
        d = r["detections"]
        if not d:
            print(f"    {r['tablet']:<12} {0:>6}  (no detections)")
            continue
        w = np.asarray([x["x2"] - x["x1"] for x in d])
        h = np.asarray([x["y2"] - x["y1"] for x in d])
        all_w.append(w)
        all_h.append(h)
        print(f"    {r['tablet']:<12} {len(d):>6} "
              f"{np.percentile(w, 10):>7.0f} {np.median(w):>7.0f} "
              f"{np.percentile(w, 90):>7.0f} {np.median(h):>7.0f} "
              f"{w.max():>7.0f} {h.max():>7.0f}")

    if not all_w:
        verdict("detected boxes are sign-sized", None, "no detections to judge.")
        return

    W = np.concatenate(all_w)
    H = np.concatenate(all_h)
    mw, mh = float(np.median(W)), float(np.median(H))
    ref_w = float(np.median(gw)) if cols else GT_MEDIAN_W
    ref_h = float(np.median(gh)) if cols else GT_MEDIAN_H

    # A box covering a quarter of a tile is the giant-box failure.
    huge = int(((W > 256) & (H > 256)).sum())
    print(f"\n  pooled detected median : {mw:.0f} x {mh:.0f} px")
    print(f"  ground-truth median    : {ref_w:.0f} x {ref_h:.0f} px")
    print(f"  ratio                  : {mw / ref_w:.2f} x {mh / ref_h:.2f}")
    print(f"  boxes larger than 256x256 px : {huge} of {len(W)} "
          f"({100 * huge / len(W):.1f}%)")

    ok = (0.5 <= mw / ref_w <= 2.0) and (0.5 <= mh / ref_h <= 2.0) and huge == 0
    verdict("detected boxes are sign-sized", ok,
            "box geometry matches the annotated signs; this is NOT the "
            "giant-box failure mode Chapter 5 described."
            if ok else
            "boxes are not sign-sized. Do NOT build the live adapter on this: "
            "the gap geometry assumes one box per sign.")


# B. Class behaviour

def section_b(runs, emit) -> None:
    head("B. IS THE CLASSIFIER DISCRIMINATING, OR DEFAULTING?")

    print("  Chapter 5's documented failure was a model emitting the same few")
    print("  frequent classes regardless of input. That would show here as a")
    print("  tiny distinct-class count and one class taking a large share.\n")

    cls = [d["train_label"] for r in runs for d in r["detections"]]
    if not cls:
        verdict("classifier is discriminating", None, "no detections.")
        return

    arr = np.asarray(cls)
    vals, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)

    print(f"  detections {len(arr):,} | distinct classes {len(vals)}")
    print(f"  commonest class share : {counts.max() / len(arr) * 100:.1f}%")
    print(f"\n  top 8 classes: " + ", ".join(
        f"{int(vals[i])}({counts[i]})" for i in order[:8]))

    if arr.max() >= config.D1_CHECKPOINT_NC:
        print(f"\n  WARNING: class {arr.max()} >= nc={config.D1_CHECKPOINT_NC}. "
              f"Impossible for this head. Stop and investigate.")

    if emit:
        inside = int(np.isin(arr, list(emit)).sum())
        print(f"\n  within the 82 emit classes : {inside:,} of {len(arr):,} "
              f"({100 * inside / len(arr):.1f}%)")
        print(f"  -> the remaining {len(arr) - inside:,} become "
              f"Detection(\"unknown\") in the live adapter.")
    else:
        print("\n  vocab.json not found; emit-filter share not measured.")

    ok = len(vals) >= 10 and counts.max() / len(arr) < 0.5
    verdict("classifier is discriminating between inputs", ok,
            f"{len(vals)} distinct classes, no single class dominant."
            if ok else
            "output is concentrated on very few classes -- the Chapter 5 "
            "failure signature. Investigate before building on it.")


# C. NMS sweep

def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest]); iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest]); iy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= thr]
    return keep


def section_c(runs, bb) -> None:
    head("C. HOW SENSITIVE IS THE SIGN COUNT TO THE NMS THRESHOLD?")

    if "raw_detections" not in (runs[0] or {}):
        verdict("NMS threshold is evidence-based", None,
                "raw_detections not in the JSON -- rerun inference after the "
                "edit that saves them. Sweep skipped.")
        return

    print("  The post-NMS count IS the sign count insert_masks() does its gap")
    print("  arithmetic against. A threshold that merges adjacent signs")
    print("  would silently shorten every line.\n")

    thrs = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    print(f"    {'tablet':<12} {'gt':>6} {'raw':>6} " +
          " ".join(f"{t:>6.2f}" for t in thrs))

    tot = {t: 0 for t in thrs}
    tot_gt = tot_raw = 0
    for r in runs:
        raw = r.get("raw_detections") or []
        if not raw:
            continue
        a = np.asarray(raw, dtype=float)
        boxes, scores = a[:, :4], a[:, 5]
        n_gt = int((bb["tablet_CDLI"] == r["tablet"]).sum())
        tot_gt += n_gt
        tot_raw += len(raw)
        row = {t: len(nms(boxes, scores, t)) for t in thrs}
        for t in thrs:
            tot[t] += row[t]
        print(f"    {r['tablet']:<12} {n_gt:>6} {len(raw):>6} " +
              " ".join(f"{row[t]:>6}" for t in thrs))

    print(f"    {'TOTAL':<12} {tot_gt:>6} {tot_raw:>6} " +
          " ".join(f"{tot[t]:>6}" for t in thrs))
    print(f"    {'det/gt':<12} {'':>6} {'':>6} " +
          " ".join(f"{tot[t] / max(1, tot_gt):>6.2f}" for t in thrs))

    # A flat curve means the IoU choice barely matters; a steep one means it does.
    lo, hi = tot[0.40], tot[0.60]
    spread = (hi - lo) / max(1, tot[0.50])
    print(f"\n  count change across IoU 0.40 -> 0.60: "
          f"{lo} -> {hi} ({100 * spread:+.1f}% of the 0.50 count)")
    verdict("the sign count is stable around the chosen IoU", spread < 0.25,
            f"a {100 * spread:.0f}% swing means the NMS threshold is doing real "
            f"work and must be chosen deliberately, not defaulted."
            if spread >= 0.25 else
            "the count is insensitive here, so 0.50 is a safe choice rather "
            "than a load-bearing one. Say so in the report.")


# D. Placement

def section_d(runs, bb) -> None:
    head("D. DO DETECTIONS LAND WHERE SIGNS ACTUALLY ARE?")

    print("  Centre-distance to the NEAREST annotated box, class ignored.")
    print("  This is a placement sanity check, NOT recall -- recall needs IoU")
    print("  matching and a class comparison, done in evaluate_domain1.\n")

    tol = GT_MEDIAN_W  # within one sign width of some annotated sign
    print(f"    {'tablet':<12} {'n':>6} {'p50 dist':>9} {'p90 dist':>9} "
          f"{'within ' + str(int(tol)) + 'px':>14}")

    hits = total = 0
    for r in runs:
        d = r["detections"]
        g = bb[bb["tablet_CDLI"] == r["tablet"]]
        if not d or g.empty:
            continue
        dc = np.asarray([[x["cx"], x["cy"]] for x in d])
        gc = g[["cx", "cy"]].to_numpy(dtype=float)
        dist = np.sqrt(((dc[:, None, :] - gc[None, :, :]) ** 2).sum(-1)).min(1)
        near = int((dist <= tol).sum())
        hits += near
        total += len(dist)
        print(f"    {r['tablet']:<12} {len(d):>6} {np.median(dist):>9.0f} "
              f"{np.percentile(dist, 90):>9.0f} "
              f"{near:>7} ({100 * near / len(dist):>4.0f}%)")

    if not total:
        verdict("detections land on inscribed regions", None, "nothing to check.")
        return

    frac = hits / total
    print(f"\n  pooled: {hits:,} of {total:,} detections ({100 * frac:.1f}%) "
          f"sit within {int(tol)} px of an annotated sign")
    verdict("detections land on inscribed regions", frac >= 0.70,
            f"only {100 * frac:.0f}% are near any annotated sign. Either the "
            f"detector fires on blank clay, or it finds real signs the sparse "
            f"annotation never marked (the annotation has only about 0.6 "
            f"boxes per corpus token), so BOTH are possible. Look at the overlay."
            if frac < 0.70 else
            "placement is consistent with the annotation.")


# E. Overlay

def section_e(runs, bb, cols, conf_draw: float) -> None:
    head("E. VISUAL OVERLAY")

    for r in runs:
        t = r["tablet"]
        p = config.IMAGES_DIR / f"{t}.jpg"
        if not p.exists():
            continue
        img = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
        dr = ImageDraw.Draw(img)

        g = bb[bb["tablet_CDLI"] == t]
        if cols:
            for _, row in g.iterrows():
                dr.rectangle([row[cols[0]], row[cols[1]],
                              row[cols[2]], row[cols[3]]],
                             outline=(0, 200, 0), width=3)
        else:
            for _, row in g.iterrows():
                cx, cy = float(row["cx"]), float(row["cy"])
                dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4],
                           outline=(0, 200, 0), width=3)

        n = 0
        for d in r["detections"]:
            if d["conf"] < conf_draw:
                continue
            dr.rectangle([d["x1"], d["y1"], d["x2"], d["y2"]],
                         outline=(255, 40, 40), width=3)
            n += 1

        img.thumbnail((1400, 1400))
        out = config.D1_LIVE_DETECTIONS / f"{t}_overlay.jpg"
        img.save(out, quality=88)
        print(f"    {t:<12} green = {len(g)} annotated, "
              f"red = {n} detections at conf >= {conf_draw}")
        print(f"      {out}")

    print("\n  OPEN THESE. Every numeric check above can pass on output that "
          "is visibly wrong.")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tablets", nargs="*", default=None)
    ap.add_argument("--draw-conf", type=float, default=0.15,
                    help="only draw detections at or above this, for legibility")
    args = ap.parse_args()

    from cuneiform.domain1_vision.group_lines import load_boxes

    print(RULE)
    print("DOMAIN 1 -- DETECTION AUDIT")
    print(RULE)

    bb = load_boxes()
    bb = attach_corners(bb)
    cols = gt_corner_cols(bb)

    if args.tablets:
        names = args.tablets
    else:
        names = sorted(p.stem for p in config.D1_LIVE_DETECTIONS.glob("*.json")
                       if not p.stem.startswith("_"))

    runs = [r for r in (load_run(t) for t in names) if r]
    if not runs:
        print("  No detection JSON found. Run run_yolo_inference.py first.")
        return

    print(f"  {len(runs)} tablet run(s): {[r['tablet'] for r in runs]}")
    print(f"  ground truth: {len(bb):,} boxes | corner columns: "
          f"{cols if cols else 'NOT FOUND (centres only)'}")

    section_a(runs, bb, cols)
    section_b(runs, emit_classes())
    section_c(runs, bb)
    section_d(runs, bb)
    section_e(runs, bb, cols, args.draw_conf)

    print("\n" + RULE)


if __name__ == "__main__":
    main()