"""
Measures what my detector actually does, on the things the pipeline uses.

mAP50 alone is misleading here: 18 val classes have no training boxes (which
caps mAP50 at 0.914), and the pipeline may only emit the 82 classes with 20 or
more training boxes. So I also separate localisation (was a box placed?) from
classification (was it named correctly?), since a placed but unnamed box can
still be repaired by Domain 2 while a missed sign cannot.

Everything is measured on the 289 saved val tiles (4,422 boxes). These are tile
figures, not tablet figures. The matcher and the 101-point AP reproduce
Ultralytics' own, and gate G1 checks that against model.val(). This script only
measures: it writes JSON under artifacts/domain1/eval/ and prints a panel.

Run from the project root:
    python -m cuneiform.domain1_vision.evaluate_domain1
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cuneiform import config
from cuneiform.domain2_sequence.build_vocab import count_vision_labels

# Ultralytics' own val() defaults, so G1 compares like with like.
VAL_CONF_FLOOR = 0.001
VAL_NMS_IOU = 0.7
VAL_MAX_DET = 300

# The deployed operating point (conf 0.15).
OPERATING_CONF = 0.15

IOU_MATCH = 0.5

# How close my mAP50 must be to Ultralytics' for G1 to pass.
G1_TOLERANCE = 0.005

# Expected number of val tiles; anything else means a different dataset (G3).
EXPECTED_VAL_TILES = 289
EXPECTED_VAL_BOXES = 4422
EXPECTED_EMIT_CLASSES = 82

# Buckets of per-tablet median sign height in source pixels. Nothing in the
# pipeline normalises scale, so the same sign can be 35 px on one tablet and
# 194 px on another; these buckets show whether that costs accuracy.
SCALE_BUCKETS = [
    (0.0, 45.0, "tiny <45px"),
    (45.0, 60.0, "small 45-60"),
    (60.0, 90.0, "mid 60-90"),
    (90.0, 130.0, "large 90-130"),
    (130.0, 1e9, "huge >130"),
]

CHECKPOINTS = {
    "deployed_colab": config.D1_WEIGHTS,
    "local_rocm": config.PROJECT_ROOT / "runs" / "domain1_local_v1" / "weights" / "best.pt",
}

EVAL_DIR = config.D1_DIR / "eval"


# Small utilities

def _trapz(y, x):
    """numpy renamed trapz to trapezoid; support both."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    IoU between two sets of xyxy boxes, shape (len(a), len(b)). Written out by
    hand so G1 tests my arithmetic against Ultralytics'.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_predictions(pred_cls, true_cls, iou, threshold, class_aware=True):
    """
    Ultralytics-compatible matching. Returns (correct_pred, gt_to_pred).

    Wrong-class pairs are masked, pairs above the IoU threshold are sorted by
    IoU (highest first), then I keep the first match for each gt and each pred.
    """
    n_pred = len(pred_cls)
    n_gt = len(true_cls)
    correct = np.zeros(n_pred, dtype=bool)
    gt_to_pred = np.full(n_gt, -1, dtype=int)
    if n_pred == 0 or n_gt == 0:
        return correct, gt_to_pred

    iou = iou.copy()
    if class_aware:
        iou = iou * (true_cls[:, None] == pred_cls[None, :])

    gt_idx, pred_idx = np.nonzero(iou >= threshold)
    if len(gt_idx) == 0:
        return correct, gt_to_pred

    matches = np.stack([gt_idx, pred_idx], axis=1)
    if matches.shape[0] > 1:
        order = iou[matches[:, 0], matches[:, 1]].argsort()[::-1]
        matches = matches[order]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    correct[matches[:, 1]] = True
    gt_to_pred[matches[:, 0]] = matches[:, 1]
    return correct, gt_to_pred


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    101-point interpolated AP over a monotonised precision envelope.

    I use Ultralytics' four sentinels (mrec = [0, *recall, recall[-1], 1],
    mpre = [1, *precision, 0, 0]) so precision drops straight to zero at the
    highest recall reached. The usual two-sentinel version credits area for
    recall the detector never reached and overstates mAP50.
    """
    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    mrec = np.concatenate(
        ([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0])
    )
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


def ap_per_class(tp, conf, pred_cls, target_cls) -> dict[int, float]:
    """
    AP at one IoU threshold for every class in the ground truth. Classes with
    ground truth but no prediction count as 0, as in Ultralytics.
    """
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    out: dict[int, float] = {}
    for c, n_gt in zip(*np.unique(target_cls, return_counts=True)):
        sel = pred_cls == c
        if sel.sum() == 0 or n_gt == 0:
            out[int(c)] = 0.0
            continue
        tpc = tp[sel].cumsum()
        fpc = (~tp[sel]).cumsum()
        recall = tpc / (n_gt + 1e-16)
        precision = tpc / (tpc + fpc + 1e-16)
        out[int(c)] = compute_ap(recall, precision)
    return out


def percentiles(values, ps=(1, 5, 25, 50, 75, 95, 99)) -> dict[str, float]:
    if not len(values):
        return {f"p{p}": float("nan") for p in ps}
    arr = np.sort(np.asarray(values, dtype=float))
    return {f"p{p}": float(arr[min(len(arr) - 1, int(p / 100 * (len(arr) - 1)))])
            for p in ps}


# Loading

def tablet_of(tile_stem: str) -> str:
    """
    'P334926_tile_0_384' -> 'P334926'. I split on the literal '_tile_', since
    some tablet ids are not P-numbers.
    """
    marker = "_tile_"
    idx = tile_stem.find(marker)
    if idx < 0:
        raise ValueError(
            f"Tile name {tile_stem!r} does not contain {marker!r}. The naming "
            f"convention assumed by scale bucketing (gate G5) does not hold."
        )
    return tile_stem[:idx]


def load_ground_truth(split: str = "val"):
    """
    Read the YOLO label files into per-tile xyxy pixel boxes. I read the .txt
    files because they are exactly what Ultralytics trained on.
    """
    img_dir = config.YOLO_DIR / "images" / split
    lbl_dir = config.YOLO_DIR / "labels" / split
    tiles = sorted(p for p in img_dir.glob("*.jpg"))

    gt = {}
    for img in tiles:
        lbl = lbl_dir / (img.stem + ".txt")
        cls_list, box_list = [], []
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                c = int(parts[0])
                xc, yc, w, h = (float(v) for v in parts[1:5])
                cls_list.append(c)
                box_list.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
        gt[img.stem] = {
            "path": img,
            "cls": np.asarray(cls_list, dtype=int),
            # Normalised for now; scaled to pixels once the tile size is known.
            "box_norm": np.asarray(box_list, dtype=np.float64).reshape(-1, 4),
        }
    return gt


def tablet_scale_map() -> dict[str, float]:
    """
    Per-tablet median ground-truth sign height in source pixels, loaded through
    generate_tiles.load_annotations so the box convention is checked.
    """
    from cuneiform.domain1_vision.generate_tiles import load_annotations

    df = load_annotations(str(config.TEST_BBOX_CSV))
    if df.empty:
        raise FileNotFoundError(f"{config.TEST_BBOX_CSV} produced no rows.")

    heights = defaultdict(list)
    for tablet, bbox in zip(df["tablet_CDLI"], df["bbox"]):
        box = bbox if isinstance(bbox, (list, tuple)) else ast.literal_eval(bbox)
        heights[str(tablet)].append(float(box[3]) - float(box[1]))
    return {t: stats.median(hs) for t, hs in heights.items()}


def bucket_for(median_height: float) -> str:
    for lo, hi, label in SCALE_BUCKETS:
        if lo <= median_height < hi:
            return label
    return SCALE_BUCKETS[-1][2]


# Gates

class Gates:
    """Records PASS/FAIL with the raw evidence beside each verdict."""

    def __init__(self):
        self.results: list[dict] = []

    def record(self, name: str, passed: bool, evidence: str, detail: str = "") -> bool:
        self.results.append({
            "gate": name, "passed": bool(passed),
            "evidence": evidence, "detail": detail,
        })
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}")
        for line in evidence.splitlines():
            print(f"         {line}")
        if detail:
            for line in detail.splitlines():
                print(f"         {line}")
        if not passed:
            print("         >>> IF THIS FAILED, IGNORE EVERY NUMBER BELOW. <<<")
        return passed

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.results)


# Inference

def run_inference(weights: Path, gt: dict, device: str):
    """
    Run every val tile through the model and post-process exactly as model.val() does.

    I use val()'s NMS call with multi_label=True, since predict() keeps only
    each box's best class and gives a different mAP. I also feed true 512 px
    tiles: val() pads them to 544 by default (rect=True), so the G1 reference
    run uses rect=False to match. That means my figures are at a true 512 input,
    slightly different from the historical 0.252. Inference runs on CPU so the
    results are deterministic.
    """
    import cv2
    import torch
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils import nms as ultra_nms

    # AutoBackend with fuse=True is exactly what val() runs.
    net = AutoBackend(str(weights), device=torch.device(device), dnn=False,
                      fp16=False, fuse=True, verbose=False)
    net.eval()

    preds = {}
    stems = sorted(gt.keys())
    with torch.inference_mode():
        for i, stem in enumerate(stems, 1):
            if i % 50 == 0 or i == len(stems):
                print(f"    {i}/{len(stems)} tiles", flush=True)
            im = cv2.imread(str(gt[stem]["path"]))
            if im is None:
                raise IOError(f"cv2 could not read {gt[stem]['path']}")
            h, w = im.shape[:2]
            x = torch.from_numpy(
                np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))
            ).to(device).float().div_(255.0).unsqueeze(0)

    # Check what the network is actually fed, not the source image size.
            if tuple(x.shape) != (1, 3, config.D1_TILE_SIZE, config.D1_TILE_SIZE):
                raise ValueError(
                    f"{stem}: network input is {tuple(x.shape)}, expected "
                    f"(1, 3, {config.D1_TILE_SIZE}, {config.D1_TILE_SIZE}). The "
                    f"run is not comparable to the rect=False reference. Stop."
                )

            out = net(x)
            det = ultra_nms.non_max_suppression(
                out,
                VAL_CONF_FLOOR,
                VAL_NMS_IOU,
                nc=0,
                multi_label=True,          # matches val()
                agnostic=False,
                max_det=VAL_MAX_DET,
            )[0]
            det = det.cpu().numpy().astype(np.float64)
            preds[stem] = {
                "box": det[:, :4].reshape(-1, 4),
                "conf": det[:, 4],
                "cls": det[:, 5].astype(int),
                "shape": (int(h), int(w)),
            }
    return preds


def reference_map50(weights: Path, device: str) -> float:
    """
    Ultralytics' own mAP50 on the val split, the reference for G1. rect=False
    is required so both sides run at a true 512.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(config.YOLO_DATA_YAML),
        imgsz=config.D1_TILE_SIZE,
        batch=1,
        rect=False,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
        save_json=False,
        project=str(EVAL_DIR / "_ultralytics_val"),
        name=weights.parent.parent.name or "ref",
        exist_ok=True,
    )
    return float(metrics.box.map50)


# Metric panel

def build_flat_arrays(gt, preds, class_aware: bool):
    """
    Flatten per-tile matches into the arrays ap_per_class needs, plus the per-gt
    and per-prediction records the other metrics use.
    """
    tp, conf, pred_cls, target_cls = [], [], [], []
    gt_records = []     # one per ground-truth box
    pred_records = []   # one per prediction

    for stem in sorted(gt.keys()):
        g, p = gt[stem], preds[stem]
        h, w = p["shape"]
        gbox = g["box_norm"] * np.array([w, h, w, h], dtype=np.float64)
        gcls = g["cls"]

        iou = box_iou(gbox, p["box"])
        correct, gt_to_pred = match_predictions(
            p["cls"], gcls, iou, IOU_MATCH, class_aware=class_aware
        )

        tp.append(correct)
        conf.append(p["conf"])
        pred_cls.append(p["cls"] if class_aware else np.zeros_like(p["cls"]))
        target_cls.append(gcls if class_aware else np.zeros_like(gcls))

        for j in range(len(gcls)):
            k = int(gt_to_pred[j])
            gt_records.append({
                "tile": stem,
                "gt_cls": int(gcls[j]),
                "matched": k >= 0,
                "pred_cls": int(p["cls"][k]) if k >= 0 else None,
                "conf": float(p["conf"][k]) if k >= 0 else None,
            })
        for k in range(len(p["cls"])):
            pred_records.append({
                "tile": stem,
                "pred_cls": int(p["cls"][k]),
                "conf": float(p["conf"][k]),
                "matched": bool(correct[k]),
                "w": float(p["box"][k, 2] - p["box"][k, 0]),
                "h": float(p["box"][k, 3] - p["box"][k, 1]),
            })

    return (
        np.concatenate(tp) if tp else np.zeros(0, bool),
        np.concatenate(conf) if conf else np.zeros(0),
        np.concatenate(pred_cls) if pred_cls else np.zeros(0, int),
        np.concatenate(target_cls) if target_cls else np.zeros(0, int),
        gt_records, pred_records,
    )


def pr_at_conf(gt_records, pred_records, threshold: float):
    """
    Recall and precision at one confidence threshold. Under class-agnostic
    matching, recall is purely a localisation figure.
    """
    kept = [p for p in pred_records if p["conf"] >= threshold]
    hit = sum(1 for g in gt_records
              if g["matched"] and g["conf"] is not None and g["conf"] >= threshold)
    n_gt = len(gt_records)
    n_pred = len(kept)
    tp = sum(1 for p in kept if p["matched"])
    return {
        "conf": threshold,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "recall": hit / n_gt if n_gt else float("nan"),
        "precision": tp / n_pred if n_pred else float("nan"),
        "spurious": n_pred - tp,
        "spurious_per_tile": (n_pred - tp) / EXPECTED_VAL_TILES,
    }


def evaluate_one(name: str, weights: Path, gt, device, gates: Gates,
                 train_counts: Counter, emit_classes: set[int],
                 tile_bucket: dict[str, str], run_reference: bool) -> dict:
    print(f"\n{'=' * 74}\nCHECKPOINT: {name}  ({weights})\n{'=' * 74}")

    # G2: checkpoint identity
    import torch

    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    head_nc = None
    model_obj = ckpt.get("model")
    if model_obj is not None:
        head_nc = getattr(model_obj, "nc", None)
        if head_nc is None:
            head_nc = getattr(model_obj, "yaml", {}).get("nc")
    train_args = ckpt.get("train_args", {}) or {}
    fitness = ckpt.get("best_fitness")
    gates.record(
        f"G2 checkpoint identity [{name}]",
        head_nc == config.D1_CHECKPOINT_NC,
        f"head nc={head_nc} (expected {config.D1_CHECKPOINT_NC}) | "
        f"run name={train_args.get('name')!r} | epochs={train_args.get('epochs')} | "
        f"best_fitness={float(fitness) if fitness is not None else None}",
        "I once read nc=354 off a stale checkpoint in a downloads folder and drew\n"
        "wrong conclusions from it, so the head is checked here.",
    )
    del ckpt, model_obj

    print(f"\n  Running inference on {len(gt)} tiles (device={device})...")
    preds = run_inference(weights, gt, device)

    # Class-aware and class-agnostic flattening
    tp_a, conf_a, pcls_a, tcls_a, gtrec_a, prec_a = build_flat_arrays(gt, preds, True)
    tp_n, conf_n, pcls_n, tcls_n, gtrec_n, prec_n = build_flat_arrays(gt, preds, False)

    ap_aware = ap_per_class(tp_a, conf_a, pcls_a, tcls_a)
    ap_agnostic = ap_per_class(tp_n, conf_n, pcls_n, tcls_n)

    val_classes = set(ap_aware)
    learnable = {c for c in val_classes if train_counts.get(c, 0) > 0}
    unlearnable = val_classes - learnable
    emit_in_val = val_classes & emit_classes

    def mean_ap(subset) -> float:
        subset = [c for c in subset if c in ap_aware]
        return float(np.mean([ap_aware[c] for c in subset])) if subset else float("nan")

    m1 = {
        "map50_all_375": mean_ap(val_classes),
        "n_classes_all": len(val_classes),
        "map50_learnable": mean_ap(learnable),
        "n_classes_learnable": len(learnable),
        "map50_emit_82": mean_ap(emit_in_val),
        "n_classes_emit_in_val": len(emit_in_val),
        "unlearnable_val_classes": sorted(unlearnable),
        "structural_ceiling": len(learnable) / len(val_classes) if val_classes else float("nan"),
    }

    # G1: matcher reproduction
    if run_reference:
        print("\n  Running Ultralytics model.val() for the G1 reference...")
        ref = reference_map50(weights, device)
        delta = abs(ref - m1["map50_all_375"])
        gates.record(
            f"G1 matcher reproduction [{name}]",
            delta <= G1_TOLERANCE,
            f"ultralytics model.val() mAP50 = {ref:.5f}\n"
            f"this script's matcher    mAP50 = {m1['map50_all_375']:.5f}\n"
            f"|delta| = {delta:.5f}  (tolerance {G1_TOLERANCE})",
            "This is the essential gate. It validates a hand-written matcher against\n"
            "an external reference. Without it every figure in M1-M7 is unfalsifiable.",
        )
        m1["ultralytics_reference_map50"] = ref
    else:
        print("\n  G1 reference skipped (--no-reference).")

    # M2: class-agnostic localisation
    m2 = {
        "ap50_class_agnostic": float(np.mean(list(ap_agnostic.values()))) if ap_agnostic else float("nan"),
        "at_operating_conf": pr_at_conf(gtrec_n, prec_n, OPERATING_CONF),
        "at_val_floor": pr_at_conf(gtrec_n, prec_n, VAL_CONF_FLOOR),
    }

    # M3: matched-box top-1 classification
    def top1(records, subset=None) -> dict:
        pool = [g for g in records
                if g["matched"] and g["conf"] >= OPERATING_CONF
                and (subset is None or g["gt_cls"] in subset)]
        correct = sum(1 for g in pool if g["pred_cls"] == g["gt_cls"])
        return {
            "n_matched": len(pool),
            "n_correct": correct,
            "top1": correct / len(pool) if pool else float("nan"),
        }

    strata = [(0, 0, "0 examples"), (1, 4, "1-4"), (5, 19, "5-19"),
              (20, 49, "20-49"), (50, 199, "50-199"), (200, 10**9, "200+")]
    m3 = {
        "all_375": top1(gtrec_n),
        "emit_82": top1(gtrec_n, emit_classes),
        "by_train_frequency": {
            label: top1(gtrec_n, {c for c in val_classes
                                 if lo <= train_counts.get(c, 0) <= hi})
            for lo, hi, label in strata
        },
    }

    # M4: confidence distributions
    correct_c, wrong_c, spurious_c = [], [], []
    aware_matched = {(p["tile"], p["pred_cls"], p["conf"]) for p in prec_a if p["matched"]}
    for p in prec_n:
        if not p["matched"]:
            spurious_c.append(p["conf"])
        elif (p["tile"], p["pred_cls"], p["conf"]) in aware_matched:
            correct_c.append(p["conf"])
        else:
            wrong_c.append(p["conf"])

    def geometry_by_band(lo, hi):
        sel = [p for p in prec_n if lo <= p["conf"] < hi]
        return {
            "n": len(sel),
            "median_w": float(stats.median([p["w"] for p in sel])) if sel else float("nan"),
            "median_h": float(stats.median([p["h"] for p in sel])) if sel else float("nan"),
        }

    m4 = {
        "n_detections_total": len(prec_n),
        "correct_label": {"n": len(correct_c), **percentiles(correct_c),
                          "mean": float(np.mean(correct_c)) if correct_c else float("nan")},
        "wrong_label": {"n": len(wrong_c), **percentiles(wrong_c),
                        "mean": float(np.mean(wrong_c)) if wrong_c else float("nan")},
        "spurious": {"n": len(spurious_c), **percentiles(spurious_c),
                     "mean": float(np.mean(spurious_c)) if spurious_c else float("nan")},
        # Are the very wide boxes confined to the low-confidence tail?
        "geometry_by_conf_band": {
            "0.001-0.05": geometry_by_band(0.001, 0.05),
            "0.05-0.15": geometry_by_band(0.05, OPERATING_CONF),
            "0.15-0.50": geometry_by_band(OPERATING_CONF, 0.50),
            "0.50-1.00": geometry_by_band(0.50, 1.01),
        },
    }

    # M5: confusion structure
    confusions = Counter(
        (g["gt_cls"], g["pred_cls"]) for g in gtrec_n
        if g["matched"] and g["conf"] >= OPERATING_CONF
        and g["pred_cls"] != g["gt_cls"]
        and g["gt_cls"] in emit_classes
    )
    m5 = {
        "n_wrong_label_pairs": int(sum(confusions.values())),
        "n_distinct_pairs": len(confusions),
        "top_confusions": [
            {"gt": int(a), "pred": int(b), "n": int(n),
             "gt_train_boxes": int(train_counts.get(a, 0)),
             "pred_train_boxes": int(train_counts.get(b, 0))}
            for (a, b), n in confusions.most_common(25)
        ],
    }

    # M6: spurious rate
    m6 = {
        "at_operating_conf": {
            "spurious": m2["at_operating_conf"]["spurious"],
            "per_tile": m2["at_operating_conf"]["spurious_per_tile"],
        },
        "at_val_floor": {
            "spurious": m2["at_val_floor"]["spurious"],
            "per_tile": m2["at_val_floor"]["spurious_per_tile"],
        },
    }

    # M7: per scale stratum
    m7 = {}
    for _, _, label in SCALE_BUCKETS:
        tiles = {s for s, b in tile_bucket.items() if b == label}
        if not tiles:
            continue
        g_sub = [g for g in gtrec_n if g["tile"] in tiles]
        p_sub = [p for p in prec_n if p["tile"] in tiles]
        pool = [g for g in g_sub if g["matched"] and g["conf"] >= OPERATING_CONF]
        corr = sum(1 for g in pool if g["pred_cls"] == g["gt_cls"])
        n_gt = len(g_sub)
        kept = [p for p in p_sub if p["conf"] >= OPERATING_CONF]
        tp_k = sum(1 for p in kept if p["matched"])
        m7[label] = {
            "n_tiles": len(tiles),
            "n_gt_boxes": n_gt,
            "recall": len(pool) / n_gt if n_gt else float("nan"),
            "precision": tp_k / len(kept) if kept else float("nan"),
            "top1_of_matched": corr / len(pool) if pool else float("nan"),
        }

    return {"checkpoint": str(weights), "M1": m1, "M2": m2, "M3": m3,
            "M4": m4, "M5": m5, "M6": m6, "M7": m7}


# Reporting

def print_panel(name: str, r: dict) -> None:
    m1, m2, m3, m4, m6, m7 = r["M1"], r["M2"], r["M3"], r["M4"], r["M6"], r["M7"]
    op, floor = m2["at_operating_conf"], m2["at_val_floor"]

    print(f"\n{'-' * 74}\nMETRIC PANEL: {name}")
    print(f"population: {op['n_gt']} val boxes over {EXPECTED_VAL_TILES} tiles, "
          f"operating conf {OPERATING_CONF}\n{'-' * 74}")

    print("\nM1  mAP50 under three denominators")
    ref = m1.get("ultralytics_reference_map50")
    print(f"      all {m1['n_classes_all']} val classes      "
          f"{m1['map50_all_375']:.4f}"
          + (f"   (ultralytics says {ref:.4f})" if ref is not None else ""))
    print(f"      {m1['n_classes_learnable']} learnable classes    "
          f"{m1['map50_learnable']:.4f}")
    print(f"      {m1['n_classes_emit_in_val']} emit classes in val  "
          f"{m1['map50_emit_82']:.4f}   <- the pipeline's actual label space")
    print(f"      structural ceiling on the all-classes figure: "
          f"{m1['structural_ceiling']:.4f}")
    print(f"      {len(m1['unlearnable_val_classes'])} val classes have zero training "
          f"boxes and can never score above 0.")

    print("\nM2  class-agnostic LOCALISATION  (labels ignored; 'is a slot present')")
    print(f"      AP50                       {m2['ap50_class_agnostic']:.4f}")
    print(f"      recall    @conf {OPERATING_CONF:<5}      {op['recall']:.4f}")
    print(f"      precision @conf {OPERATING_CONF:<5}      {op['precision']:.4f}")
    print(f"      recall    @conf {VAL_CONF_FLOOR:<5}     {floor['recall']:.4f}"
          f"   (upper bound on placement)")

    print("\nM3  matched-box TOP-1 CLASSIFICATION  (of boxes localised at IoU>=0.5)")
    print(f"      over all 375 classes       {m3['all_375']['top1']:.4f}"
          f"   ({m3['all_375']['n_correct']}/{m3['all_375']['n_matched']})")
    print(f"      over the 82 emit classes   {m3['emit_82']['top1']:.4f}"
          f"   ({m3['emit_82']['n_correct']}/{m3['emit_82']['n_matched']})")
    print("      by training frequency:")
    for label, v in m3["by_train_frequency"].items():
        if v["n_matched"]:
            print(f"        {label:<12} {v['top1']:.4f}  ({v['n_correct']}/{v['n_matched']})")
        else:
            print(f"        {label:<12}   --    (no matched boxes)")

    print("\nM4  confidence distributions  (replaces simulate_vision assumption 1)")
    for key, beta in (("correct_label", "Beta(5,2) mean 0.714"),
                      ("wrong_label", "Beta(2,3) mean 0.400"),
                      ("spurious", "assumed absent")):
        d = m4[key]
        print(f"      {key:<14} n={d['n']:<6} median={d['p50']:.4f} "
              f"mean={d['mean']:.4f}  p5={d['p5']:.4f} p95={d['p95']:.4f}"
              f"   [simulated: {beta}]")
    print("      predicted box geometry by confidence band:")
    for band, d in m4["geometry_by_conf_band"].items():
        print(f"        {band:<12} n={d['n']:<6} median {d['median_w']:.0f} x "
              f"{d['median_h']:.0f} px")

    print("\nM6  spurious detections  (replaces simulate_vision assumption 3)")
    print(f"      @conf {OPERATING_CONF}: {m6['at_operating_conf']['spurious']:>6} "
          f"({m6['at_operating_conf']['per_tile']:.2f} per tile)")
    print(f"      @conf {VAL_CONF_FLOOR}: {m6['at_val_floor']['spurious']:>6} "
          f"({m6['at_val_floor']['per_tile']:.2f} per tile)")

    print("\nM7  per tablet SCALE stratum  (median GT sign height, source px)")
    print(f"      {'bucket':<14} {'tiles':>6} {'boxes':>7} {'recall':>8} "
          f"{'prec':>8} {'top1':>8}")
    for label, v in m7.items():
        print(f"      {label:<14} {v['n_tiles']:>6} {v['n_gt_boxes']:>7} "
              f"{v['recall']:>8.4f} {v['precision']:>8.4f} {v['top1_of_matched']:>8.4f}")


# Main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", action="append", default=None,
                    help="Name from CHECKPOINTS, or a path. Repeatable. "
                         "Default: every entry in CHECKPOINTS that exists.")
    ap.add_argument("--device", default="cpu",
                    help="Torch device. Default cpu -- see the docstring on "
                         "inference determinism.")
    ap.add_argument("--no-reference", action="store_true",
                    help="Skip the Ultralytics model.val() call that anchors G1. "
                         "Faster, but leaves the matcher ungated.")
    args = ap.parse_args()

    config.ensure_dirs()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("DOMAIN 1 METRIC PANEL")
    print("=" * 74)
    print("MEASUREMENT ONLY. No training, no re-tiling, no build_vocab rebuild.")
    print("Population: the saved VAL TILES. Tiles are NOT tablets -- see the")
    print("module docstring before quoting any figure.")

    gates = Gates()
    print("\nGATES")
    print("-" * 74)

    # G3: population
    gt = load_ground_truth("val")
    n_tiles = len(gt)
    n_boxes = sum(len(g["cls"]) for g in gt.values())
    gates.record(
        "G3 population",
        n_tiles == EXPECTED_VAL_TILES and n_boxes == EXPECTED_VAL_BOXES,
        f"scanned {n_tiles} val tiles / {n_boxes} boxes "
        f"(expected {EXPECTED_VAL_TILES} / {EXPECTED_VAL_BOXES})",
        "A different population invalidates every comparison to the recorded 0.252.",
    )

    # G4: vocabulary coherence (read-only)
    from cuneiform.domain2_sequence.tokenizer import SignTokenizer

    tok = SignTokenizer()
    emit_classes = set(tok.vision_emit_classes)
    print("\n  reading Domain 1 label files for per-class train counts:")
    train_counts = count_vision_labels("train")
    starved = sorted(c for c in emit_classes if train_counts.get(c, 0) < 20)
    gates.record(
        "G4 vocabulary coherence",
        len(emit_classes) == EXPECTED_EMIT_CLASSES and not starved,
        f"vocab.json vision_emit_classes = {len(emit_classes)} "
        f"(expected {EXPECTED_EMIT_CLASSES}) | "
        f"emit classes with <20 train boxes on disk = {len(starved)} {starved[:10]}",
        "Read-only. build_vocab.main() is NOT called: re-running it can shift token\n"
        "ids and silently invalidate the 175 MB Domain 2 checkpoint.",
    )

    # G5: scale bucketing
    scale_map = tablet_scale_map()
    tile_bucket, unmapped = {}, []
    for stem in gt:
        tablet = tablet_of(stem)
        if tablet not in scale_map:
            unmapped.append(stem)
            continue
        tile_bucket[stem] = bucket_for(scale_map[tablet])
    bucketed_boxes = sum(len(gt[s]["cls"]) for s in tile_bucket)
    tablets_seen = {tablet_of(s) for s in gt}
    gates.record(
        "G5 scale bucketing",
        not unmapped and bucketed_boxes == n_boxes,
        f"{len(tablets_seen)} distinct val tablets | {len(tile_bucket)}/{n_tiles} tiles "
        f"bucketed | {bucketed_boxes}/{n_boxes} boxes accounted for | "
        f"unmapped tiles: {unmapped[:5]}",
        "A tile that matches no tablet would silently distort M7.",
    )

    if not gates.all_passed:
        print("\nA pre-inference gate FAILED. Stopping before any measurement, "
              "because the numbers would be meaningless.")
        json.dump({"gates": gates.results},
                  open(EVAL_DIR / "gates_failed.json", "w", encoding="utf-8"),
                  indent=2)
        return 1

    # Select checkpoints
    if args.checkpoint:
        selected = {}
        for c in args.checkpoint:
            selected[c] = Path(CHECKPOINTS.get(c, c))
    else:
        selected = {k: v for k, v in CHECKPOINTS.items() if v.exists()}
    missing = {k: v for k, v in selected.items() if not v.exists()}
    if missing:
        print(f"\nERROR: checkpoint(s) not found: {missing}")
        return 1
    print(f"\nEvaluating {len(selected)} checkpoint(s): {list(selected)}")

    results = {}
    for name, weights in selected.items():
        results[name] = evaluate_one(
            name, weights, gt, args.device, gates,
            train_counts, emit_classes, tile_bucket,
            run_reference=not args.no_reference,
        )

    for name, r in results.items():
        print_panel(name, r)

    out = {
        "population": {
            "split": "val (saved tiles)",
            "n_tiles": n_tiles,
            "n_boxes": n_boxes,
            "note": "Tiles, not tablets. The training grid never cuts the right/"
                    "bottom edges, so 4.0% of annotated boxes have no training tile. "
                    "A tablet-level figure is a different population.",
        },
        "settings": {
            "iou_match": IOU_MATCH,
            "operating_conf": OPERATING_CONF,
            "val_conf_floor": VAL_CONF_FLOOR,
            "nms_iou": VAL_NMS_IOU,
            "max_det": VAL_MAX_DET,
            "device": args.device,
        },
        "gates": gates.results,
        "checkpoints": results,
    }
    out_path = EVAL_DIR / "metric_panel.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"\n{'=' * 74}")
    print(f"Wrote {out_path}")
    print(f"GATES: {'ALL PASS' if gates.all_passed else 'ONE OR MORE FAILED'}")
    if not gates.all_passed:
        print("At least one gate FAILED. Do not quote any figure above.")
    print("=" * 74)
    return 0 if gates.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
