"""
My service layer: one tablet id in, one JSON-ready dict out. The FastAPI app
and the tests both sit on top of it.

I import every pipeline stage and frozen constant from live_pipeline instead of
restating them, so this file always gives the same reading as the CLI. Each
slot gets one of four states (vision, override, restored, unnamed), and the
caveats travel inside the payload so no pane can show a reading without them.

Run from the project root:
    python -m cuneiform.api.service --tablet P335940
    python -m cuneiform.api.service --tablet P335940 --json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import uuid
from functools import lru_cache
from typing import Any

from cuneiform import config
from cuneiform.domain2_sequence.tokenizer import SignTokenizer
from cuneiform.domain3_generative.generate import (
    MODEL as D3_MODEL,
    classify_citations,
    load_catalogue,
    metadata_block,
    scan_prose,
)
from cuneiform.orchestrator.evaluate_live import live_tablets as _live_tablets
from cuneiform.orchestrator.fusion import CKPT_DIR as _CKPT_DIR
from cuneiform.orchestrator.fusion import load_sequence_model
from cuneiform.orchestrator.live_adapter import (
    build_slots, load_detections, load_sign_names,
)

# The frozen configuration and pipeline stages, imported rather than restated.
from cuneiform.orchestrator.live_pipeline import (
    LIVE_CELL,
    LIVE_CONF,
    LIVE_EDGE_EXTEND,
    LIVE_LAMBDA,
    LIVE_MIN_SIZE,
    LM_FLOOR,
    MIN_BLOCK,
    build_domain3_input,
    build_live_blocks,
    call_domain3,
    run_fusion,
    segment,
    to_detections,
)

# Warm state. The Domain 2 checkpoint takes about 1.4 s to load, so I load it
# lazily once and let the FastAPI lifespan warm it at startup.

_MODEL = None
_TOK: SignTokenizer | None = None
_NAMES: dict[int, str] | None = None
_CATALOGUE = None
_DETECTOR = None


def _tokenizer() -> SignTokenizer:
    global _TOK
    if _TOK is None:
        _TOK = SignTokenizer()
    return _TOK


def _sign_names() -> dict[int, str]:
    global _NAMES
    if _NAMES is None:
        _NAMES = load_sign_names()
    return _NAMES


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_sequence_model()
    return _MODEL


def _catalogue():
    global _CATALOGUE
    if _CATALOGUE is None:
        _CATALOGUE = load_catalogue()
    return _CATALOGUE


def _detector():
    """
    Load the Domain 1 detector once. Only uploads need it, since the 30 tablets
    use saved detections.

    I import ultralytics inside the function to keep startup fast, and set
    YOLO_OFFLINE before the import because ultralytics checks for internet at
    import time and this server has to run offline.
    """
    global _DETECTOR
    if _DETECTOR is None:
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO
        _DETECTOR = YOLO(str(config.D1_WEIGHTS))
    return _DETECTOR


def warm() -> dict[str, float]:
    """
    Load everything a request would otherwise pay for, and return the seconds
    each asset took. Safe to call more than once.
    """
    timings: dict[str, float] = {}
    for name, fn in (("tokenizer", _tokenizer),
                     ("sign_names", _sign_names),
                     ("sequence_model", _model),
                     ("catalogue", _catalogue)):
        t0 = time.perf_counter()
        fn()
        timings[name] = round(time.perf_counter() - t0, 3)
    return timings


# The tablet population

def list_tablets() -> list[str]:
    """
    The ids of the 30 tablets with saved detections, sorted. I reuse the
    evaluation's own function so the dashboard and the evaluation can never
    disagree about which tablets exist.
    """
    return _live_tablets()


def tablet_exists(tablet: str) -> bool:
    """True for either population: one of the 30, or an uploaded photograph."""
    return _record_path(tablet).exists()


def is_warm() -> bool:
    """Is the Domain 2 sequence model in memory? Read by /api/health."""
    return _MODEL is not None


def sequence_checkpoint_name() -> str:
    """The name of the checkpoint Domain 2 actually loads."""
    return _CKPT_DIR.name


def detector_weights_name() -> str:
    """The name of the Domain 1 detector checkpoint."""
    return config.D1_WEIGHTS.name


def frozen_config() -> dict[str, Any]:
    """The frozen configuration as plain values."""
    return {
        "conf": LIVE_CONF,
        "cell": LIVE_CELL,
        "lambda": LIVE_LAMBDA,
        "edge_extrapolation": LIVE_EDGE_EXTEND,
        "lm_floor": LM_FLOOR,
        "min_cluster": LIVE_MIN_SIZE,
    }


@lru_cache(maxsize=64)
def tablet_metadata(tablet: str, conf: float = LIVE_CONF) -> dict[str, Any]:
    """
    Cheap per-tablet facts for the tablet list, with no model run. I count
    detections with the pipeline's own loader, and never read the ground-truth
    box counts that sit in the same summary file.
    """
    rec = _detection_record(tablet)
    w, h = rec["image_size"]                     # [W, H], width first
    return {
        "tablet": tablet,
        "width": int(w),
        "height": int(h),
        "detections_in_file": int(rec.get("n_detections",
                                          len(rec["detections"]))),
        "detections_above_conf": len(_load_detections(tablet, conf)),
        "conf": conf,
        "detector_seconds": (float(rec["seconds"])
                             if rec.get("seconds") is not None else None),
    }


def image_path(tablet: str):
    """
    The photograph to display, always in the same frame as the detections.
    For an upload that is the working image (transposed and maybe rescaled),
    not the file that was sent.
    """
    if _is_upload(tablet):
        return config.UPLOADS_DIR / f"{tablet}.jpg"
    return config.IMAGES_DIR / f"{tablet}.jpg"


def _detection_record(tablet: str) -> dict:
    path = _record_path(tablet)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# Uploaded photographs. Everything downstream only needs a tablet id, so this
# feature is just a resolver plus one ingest function. Uploads are saved in
# their own folder and never join the 30 evaluation tablets.

UPLOAD_PREFIX = "upload_"

# Reject anything bigger than this before decoding it.
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
# Reject after reading the header, before allocating pixels (decompression bombs).
MAX_UPLOAD_PIXELS = 80_000_000

# Scale rule. The detector only saw signs at museum-plate scale. Across my 30
# tablets the median sign is 42.7 to 112.7 px per tablet, with a median of
# 58.5 px. An upload inside that band is left alone; outside it I rescale once
# towards 58.5 px.
UPLOAD_SIGN_TARGET_PX = 58.5
UPLOAD_SIGN_BAND = (42.7, 112.7)
# Rescales outside this range are refused, since the measurement is probably wrong.
UPLOAD_SCALE_CLAMP = (0.25, 3.0)
# With fewer boxes than this, the median is noise and I skip the rescale.
UPLOAD_MIN_BOXES_FOR_SCALE = 25


def _is_upload(tablet: str) -> bool:
    """Decide which population an id belongs to."""
    return tablet.startswith(UPLOAD_PREFIX)


def _record_path(tablet: str):
    return ((config.UPLOADS_DIR / f"{tablet}.json") if _is_upload(tablet)
            else (config.D1_LIVE_DETECTIONS / f"{tablet}.json"))


def list_uploads() -> list[str]:
    """Uploaded tablet ids, newest first. Deliberately kept out of list_tablets()."""
    out = []
    for p in config.UPLOADS_DIR.glob(f"{UPLOAD_PREFIX}*.json"):
        out.append(p.stem)
    out.sort(key=lambda t: (config.UPLOADS_DIR / f"{t}.json").stat().st_mtime,
             reverse=True)
    return out


def upload_record(tablet: str) -> dict:
    """The stored record for one upload. Raises if it is not an upload."""
    if not _is_upload(tablet):
        raise ValueError(f"{tablet} is not an uploaded photograph.")
    return _detection_record(tablet)


def upload_caveats(record: dict) -> list[str]:
    """Public accessor for the upload caveats."""
    return _upload_caveats(record)


def _load_detections(tablet: str, conf: float) -> list[dict]:
    """
    Detections above `conf` for either population. Corpus tablets go through
    the pipeline's own loader; uploads apply the same filter to their record.
    """
    if not _is_upload(tablet):
        return load_detections(tablet, conf)
    rec = _detection_record(tablet)
    return [d for d in rec["detections"] if d["conf"] >= conf]


def _median_box_side(dets: list[dict]) -> float | None:
    """The median mean-side of a detection set, in the image's own pixels."""
    if not dets:
        return None
    sides = sorted(((d["x2"] - d["x1"]) + (d["y2"] - d["y1"])) / 2.0
                   for d in dets)
    n = len(sides)
    return (sides[n // 2] if n % 2 else (sides[n // 2 - 1] + sides[n // 2]) / 2.0)


def _plan_rescale(dets: list[dict]) -> dict[str, Any]:
    """Decide whether this photograph's signs are the size the detector expects, and say why."""
    measured = _median_box_side(dets)
    if measured is None or len(dets) < UPLOAD_MIN_BOXES_FOR_SCALE:
        return {
            "applied": False, "factor": 1.0, "measured_px": measured,
            "reason": f"only {len(dets)} signs were found above the operating "
                      f"threshold, which is too few for a reliable size "
                      f"measurement, so the photograph was left unchanged.",
        }
    low, high = UPLOAD_SIGN_BAND
    if low <= measured <= high:
        return {
            "applied": False, "factor": 1.0, "measured_px": measured,
            "reason": f"the typical sign measured {measured:.0f} px across, "
                      f"inside the {low:.0f} to {high:.0f} px range spanned by "
                      f"the 30 reference tablets, so nothing was changed.",
        }
    factor = UPLOAD_SIGN_TARGET_PX / measured
    lo, hi = UPLOAD_SCALE_CLAMP
    if not (lo <= factor <= hi):
        return {
            "applied": False, "factor": 1.0, "measured_px": measured,
            "reason": f"the typical sign measured {measured:.0f} px across, so "
                      f"far outside the {low:.0f} to {high:.0f} px range of the "
                      f"reference tablets that rescaling it would be a guess. "
                      f"The photograph was left unchanged.",
        }
    return {
        "applied": True, "factor": factor, "measured_px": measured,
        "reason": f"the typical sign measured {measured:.0f} px across, "
                  f"outside the {low:.0f} to {high:.0f} px range spanned by the "
                  f"30 reference tablets, so the photograph was resized by "
                  f"{factor:.2f} to bring it to {UPLOAD_SIGN_TARGET_PX:.0f} px "
                  f"and read again.",
    }


def _upload_caveats(record: dict) -> list[str]:
    """Caveats that apply to an uploaded photograph and not to the 30 tablets."""
    scale = record.get("upload", {}).get("scale", {})
    out = [
        "This photograph is not part of any evaluation in this project. "
        "Every accuracy figure quoted anywhere in this interface was measured "
        "on 30 other tablets, and none of it describes this one.",

        "No ground truth exists for this photograph, so nothing shown for it "
        "can be checked against a known answer, not even in principle. For the "
        "30 reference tablets there are human annotations; here there are none.",

        "The detector was fine-tuned on 81 annotated museum plates: flat, "
        "evenly lit, photographed square on. A photograph taken under other "
        "conditions is outside what it has been shown, and it has no way to "
        "report that it is out of its depth.",
    ]
    if scale.get("reason"):
        out.append("Image scale: " + scale["reason"])
    return out


class UploadRejected(ValueError):
    """The bytes were not a usable photograph. Carries a message for a person."""


def _detect(image_path_, tablet_id: str) -> dict:
    """
    Run Domain 1 on one image file with the same detect_tablet code that made
    the 30 saved records. It detects at the low floor; the 0.15 threshold is
    applied later.
    """
    from cuneiform.domain1_vision.run_yolo_inference import detect_tablet
    floor, nms_iou = _corpus_inference_settings()
    return detect_tablet(_detector(), tablet_id, conf=floor,
                         nms_iou=nms_iou, verbose=False, img_path=image_path_)


@lru_cache(maxsize=1)
def _corpus_inference_settings() -> tuple[float, float]:
    """The (conf floor, NMS IoU) the 30 saved records were made with, read from the data."""
    for tablet in list_tablets():
        rec = _detection_record(tablet)
        return float(rec["conf_floor"]), float(rec["nms_iou"])
    raise RuntimeError(
        "No reference detections found, so the settings an upload must match "
        "cannot be determined.")


def ingest_upload(data: bytes, filename: str = "") -> dict[str, Any]:
    """
    Take an uploaded photograph, detect signs in it, and turn it into a tablet.

    Steps: decode and EXIF-transpose, detect at native scale, rescale once and
    detect again if the signs are the wrong size, then save the working image,
    the original and a detection record shaped like the saved ones.
    """
    if not data:
        raise UploadRejected("The uploaded file was empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"That file is {len(data) / 1e6:.0f} MB. The limit is "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB.")

    from PIL import Image, ImageOps

    t0 = time.perf_counter()
    # I identify the file by decoding it, not by the content type the browser sent.
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = (probe.format or "").upper()
        w0, h0 = probe.size
        if w0 * h0 > MAX_UPLOAD_PIXELS:
            raise UploadRejected(
                f"That image is {w0} by {h0} pixels, which is larger than this "
                f"server will decode.")
        probe.load()
        img = probe.convert("RGB")
    except UploadRejected:
        raise
    except Exception as exc:                                     # noqa: BLE001
        # Deliberately broad: once ultralytics is loaded it patches Image.open,
        # so a bad file can raise errors PIL never would. Any failure here just
        # means "I can't read this", so it becomes a message, never a 500.
        # HEIC gets named because it is the iPhone default and can't be read offline.
        raise UploadRejected(
            "That file could not be read as an image. JPEG, PNG and TIFF "
            "photographs work. HEIC, which iPhones use by default, does not: "
            "export or convert to JPEG first."
        ) from exc

    img = ImageOps.exif_transpose(img)
    t_decode = time.perf_counter() - t0

    tablet_id = UPLOAD_PREFIX + uuid.uuid4().hex[:10]
    work_path = config.UPLOADS_DIR / f"{tablet_id}.jpg"
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Pass 1, at the photograph's own scale
    img.save(work_path, "JPEG", quality=92)
    rec = _detect(work_path, tablet_id)
    passes = [{"width": rec["image_size"][0], "height": rec["image_size"][1],
               "n_detections": rec["n_detections"],
               "median_sign_px": _median_box_side(
                   [d for d in rec["detections"] if d["conf"] >= LIVE_CONF]),
               "seconds": rec["seconds"]}]

    scale = _plan_rescale([d for d in rec["detections"] if d["conf"] >= LIVE_CONF])

    # Pass 2, only if the signs are the wrong size
    if scale["applied"]:
        f = scale["factor"]
        w1, h1 = max(1, round(img.width * f)), max(1, round(img.height * f))
        img.resize((w1, h1), Image.LANCZOS).save(work_path, "JPEG", quality=92)
        rec = _detect(work_path, tablet_id)
        passes.append({"width": w1, "height": h1,
                       "n_detections": rec["n_detections"],
                       "median_sign_px": _median_box_side(
                           [d for d in rec["detections"] if d["conf"] >= LIVE_CONF]),
                       "seconds": rec["seconds"]})

    # Keep the original bytes so the user never has to upload twice.
    ext = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif"}.get(fmt, ".bin")
    (config.UPLOADS_DIR / f"{tablet_id}.original{ext}").write_bytes(data)

    rec["upload"] = {
        "filename": (filename or "")[:200],
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original_size": [w0, h0],
        "original_format": fmt,
        "exif_transposed": [img.width, img.height] != [w0, h0],
        "scale": scale,
        "passes": passes,
        "decode_seconds": round(t_decode, 3),
    }
    # Store just the file name, not a full local path.
    rec["image"] = str(work_path.name)

    with (config.UPLOADS_DIR / f"{tablet_id}.json").open(
            "w", encoding="utf-8", newline="") as fh:
        json.dump(rec, fh, ensure_ascii=False)

    tablet_metadata.cache_clear()
    return rec


# Slot to source box

def _attach_boxes(dets: list[dict], groups, slots) -> list[int | None]:
    """
    For each slot, the index of the detection it came from, or None.

    The match is exact on (x, label, confidence), since insert_masks copies those
    through unchanged, and each detection is used once. Misses are counted so
    a failed match shows up as a number.
    """
    by_seg: list[dict[tuple, list[int]]] = []
    for idx in groups:
        table: dict[tuple, list[int]] = {}
        for i in idx:
            d = dets[int(i)]
            key = (float(d["cx"]), int(d["train_label"]), float(d["conf"]))
            table.setdefault(key, []).append(int(i))
        by_seg.append(table)

    out: list[int | None] = []
    for seg_i, s in slots:
        if s.kind != "detected" or s.label is None:
            out.append(None)
            continue
        key = (float(s.x), int(s.label), float(s.confidence))
        bucket = by_seg[seg_i].get(key) if seg_i < len(by_seg) else None
        out.append(bucket.pop(0) if bucket else None)
    return out


# The analysis
#
# The summarise path needs the live TabletInput object, not the JSON, because
# the faithfulness audit builds its reference set from the object's slots.
# Rebuilding it from the rendered text would check the prose against itself,
# so analyse_tablet_with_input() hands the object back alongside the dict.


def analyse_tablet_with_input(
    tablet: str,
    conf: float = LIVE_CONF,
    cell: float = LIVE_CELL,
    lam: float = LIVE_LAMBDA,
    edge: bool = LIVE_EDGE_EXTEND,
    lm_floor: float = LM_FLOOR,
    quiet: bool = True,
) -> tuple[dict[str, Any], Any]:
    """
    Domains 1 and 2 end to end for one tablet.

    Returns (result, tablet_input): a JSON-ready dict and the TabletInput that
    Domain 3 needs (None if nothing could be fused). The defaults are the frozen
    configuration. `quiet` hides run_fusion's progress prints.
    """
    if not tablet_exists(tablet):
        raise FileNotFoundError(
            f"No detections for {tablet}. "
            f"{len(list_tablets())} tablets are available."
        )

    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    tok = _tokenizer()
    names = _sign_names()

    record = _detection_record(tablet)
    dets = _load_detections(tablet, conf)
    groups = segment(dets, cell, LIVE_MIN_SIZE)
    kept = sum(len(g) for g in groups)
    slots = build_slots(dets, groups, extend_edges=edge)
    detections = to_detections(slots, tok)
    blocks = build_live_blocks(slots)
    timings["geometry"] = round(time.perf_counter() - t_start, 3)

    in_blocks = sum(len(b) for b in blocks)
    n_det = sum(1 for d in detections if d.kind == "detected")
    n_unk = sum(1 for d in detections if d.kind == "unknown")
    n_mis = sum(1 for d in detections if d.kind == "missed")

    counts: dict[str, Any] = {
        "detections_in_file": int(record.get("n_detections",
                                             len(record["detections"]))),
        "detections_above_conf": len(dets),
        "detections_kept": kept,
        "detections_debris": len(dets) - kept,
        "segments": len(groups),
        "slots": len(slots),
        "slots_detected": n_det,
        "slots_unknown": n_unk,
        "slots_missed": n_mis,
        "blocks": len(blocks),
        "slots_in_blocks": in_blocks,
        "slots_dropped_in_remnants": len(slots) - in_blocks,
        "min_block": MIN_BLOCK,
    }

    box_of = _attach_boxes(dets, groups, slots)
    counts["box_match_misses"] = sum(
        1 for si, (_, s) in enumerate(slots)
        if s.kind == "detected" and box_of[si] is None
    )

    # No block reached the minimum length. That is a real outcome, not an error.
    if not blocks:
        return _empty_result(tablet, record, counts, conf, cell, lam, edge,
                             lm_floor, timings, t_start), None

    t0 = time.perf_counter()
    model = _model()
    timings["model_load"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    if quiet:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            preds, lm_conf = run_fusion(model, tok, slots, detections,
                                        blocks, lam)
    else:
        preds, lm_conf = run_fusion(model, tok, slots, detections, blocks, lam)
    timings["fusion"] = round(time.perf_counter() - t0, 3)

    t, d3, restored_flags, n_demoted = build_domain3_input(
        tablet, slots, detections, preds, lm_conf, lm_floor
    )

    changed = sum(1 for si, lbl in preds.items()
                  if detections[si].kind == "detected"
                  and lbl != detections[si].train_label)
    n_restorable = sum(1 for si in range(len(slots))
                       if si in preds and not detections[si].is_informative)
    n_named = sum(1 for x in d3 if x is not None)

    fusion_stats: dict[str, Any] = {
        "vision_named_positions": n_det,
        "fusion_changed": changed,
        "restoration_candidates": n_restorable,
        "restorations_kept": n_restorable - n_demoted,
        "restorations_demoted": n_demoted,
        "named_positions": n_named,
        "total_positions": len(d3),
        "lm_floor": lm_floor,
    }
    if n_restorable:
        confs = sorted(lm_conf[si] for si in range(len(slots))
                       if si in preds and not detections[si].is_informative)
        fusion_stats["restoration_posterior"] = {
            "min": round(confs[0], 4),
            "median": round(confs[len(confs) // 2], 4),
            "max": round(confs[-1], 4),
        }

    lines = _build_lines(slots, detections, preds, lm_conf, restored_flags,
                         names, box_of, dets)

    meta_text, had_meta = metadata_block(tablet, _catalogue())

    result: dict[str, Any] = {
        "tablet": tablet,
        "config": _config_block(conf, cell, lam, edge, lm_floor),
        "image": {
            "size": list(record["image_size"]),      # [W, H], width first
            "url": f"/api/tablets/{tablet}/image",
        },
        "counts": counts,
        "fusion": fusion_stats,
        "lines": lines,
        "sequence_dump": _sequence_dump(lines),
        "detections": [
            {
                "i": i,
                "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                "train_label": int(d["train_label"]),
                "name": names.get(int(d["train_label"]),
                                  "label " + str(int(d["train_label"]))),
                "conf": float(d["conf"]),
            }
            for i, d in enumerate(dets)
        ],
        "domain3_input": {
            # t.lines holds Verbalisation objects; only the text and counts are JSON-safe.
            "lines": [{"text": v.text, "counts": v.counts()} for v in t.lines],
            "catalogue_metadata": meta_text,
            "had_metadata": bool(had_meta),
        },
        # Uploads get extra caveats, in the same list so they show in the same place.
        "caveats": (_caveats() + _upload_caveats(record)
                    if _is_upload(tablet) else _caveats()),
        "timings": timings,
    }
    timings["total"] = round(time.perf_counter() - t_start, 3)
    return result, t


def analyse_tablet(tablet: str, **kwargs) -> dict[str, Any]:
    """Domains 1 and 2 for one tablet as a plain dict, without caching."""
    result, _ = analyse_tablet_with_input(tablet, **kwargs)
    return result


@lru_cache(maxsize=32)
def _analyse_frozen_cached(tablet: str) -> tuple[dict[str, Any], Any]:
    """
    The cached (result, TabletInput) pair. I cache the object too because the
    summarise path needs it; callers must treat it as read-only.
    """
    return analyse_tablet_with_input(tablet)


def analyse_tablet_frozen(tablet: str) -> tuple[dict[str, Any], bool]:
    """
    The serving path: the frozen configuration, cached per tablet.
    Returns (result, was_cached).

    Caching is safe because detections are read from disk and fusion runs
    deterministically on CPU; a test analyses the same tablet twice to prove it.
    Only the frozen configuration is ever cached.
    """
    result, _, was_cached = analyse_tablet_frozen_with_input(tablet)
    return result, was_cached


def analyse_tablet_frozen_with_input(
    tablet: str,
) -> tuple[dict[str, Any], Any, bool]:
    """
    Like analyse_tablet_frozen, but also returns the TabletInput.
    Returns (result, tablet_input, was_cached).
    """
    hits_before = _analyse_frozen_cached.cache_info().hits
    result, tablet_input = _analyse_frozen_cached(tablet)
    was_cached = _analyse_frozen_cached.cache_info().hits > hits_before
    return result, tablet_input, was_cached


def clear_analysis_cache() -> None:
    """Drop every cached analysis, for tests that need a cold run."""
    _analyse_frozen_cached.cache_clear()
    _SUMMARY_CACHE.clear()


def _config_block(conf, cell, lam, edge, lm_floor) -> dict[str, Any]:
    """The operating point, and whether it is the frozen one."""
    return {
        "conf": conf,
        "cell": cell,
        "lambda": lam,
        "edge_extrapolation": bool(edge),
        "lm_floor": lm_floor,
        "min_cluster": LIVE_MIN_SIZE,
        "is_frozen": (
            conf == LIVE_CONF and cell == LIVE_CELL and lam == LIVE_LAMBDA
            and bool(edge) == bool(LIVE_EDGE_EXTEND) and lm_floor == LM_FLOOR
        ),
        "frozen": {
            "conf": LIVE_CONF, "cell": LIVE_CELL, "lambda": LIVE_LAMBDA,
            "edge_extrapolation": LIVE_EDGE_EXTEND, "lm_floor": LM_FLOOR,
            "min_cluster": LIVE_MIN_SIZE,
        },
    }


def _caveats() -> list[str]:
    """The caveats that must travel with every reading."""
    return [
        "This system does not translate. Domain 3 produces a genre-level "
        "gist, and its field is named tentative_reading for that reason.",
        "Domain 2's top-1 accuracy is 22.2%, so MOST restorations -- every "
        "bracketed [NAME] -- are wrong. A fluent-looking reading is not "
        "evidence that the tablet was read correctly.",
        "[x] is the system declining to guess: we hold a box, or a gap we "
        "believe holds a sign, but nothing we are entitled to call a name. "
        "It is a feature, not a gap.",
        "fusion_changed counts how many predictions fusion CHANGED, not how "
        "many it FIXED. No position in the live path has a known truth.",
        "restorations_demoted counts demotions, not correct demotions, for "
        "the same reason.",
        "The configuration shown is frozen, not tuned. conf 0.15 is the "
        "midpoint of a plateau, cell 2.0 was not selected at all, and the LM "
        "floor's precision is one bucket wide, roughly 0.06 to 0.10.",
        "Segments are physical surfaces of the tablet, ordered top to "
        "bottom on the photographic plate. That is a geometric convention, "
        "not a judgement of which face is the obverse: on 5 of these 30 "
        "tablets it puts the two faces in the opposite order to the "
        "published one, at a measured cost of +0.0093 WER.",
    ]


def _build_lines(slots, detections, preds, lm_conf, restored_flags,
                 names, box_of, dets) -> list[dict[str, Any]]:
    """
    Group slots into (segment, line) rows. `token` is rendered here, exactly as
    the CLI prints it, so the browser never re-derives it.
    """
    rows: list[dict[str, Any]] = []
    cur = (None, None)

    for si, ((seg, s), d) in enumerate(zip(slots, detections)):
        if (seg, s.line) != cur:
            cur = (seg, s.line)
            rows.append({"segment": int(seg), "line": int(s.line), "slots": []})

        if si not in preds:
            state, reason, label = "unnamed", "outside_block", None
            token = "·"
        elif d.is_informative:
            label = int(preds[si])
            if label != d.train_label:
                state, reason = "override", None
                token = names.get(label, "#" + str(label)) + "*"
            else:
                state, reason = "vision", None
                token = names.get(label, "#" + str(label))
        elif not restored_flags[si]:
            state, reason, label = "unnamed", "below_floor", None
            token = "[x]"
        else:
            label = int(preds[si])
            state, reason = "restored", None
            token = "[" + names.get(label, "?") + "]"

        detected = d.kind == "detected"
        rows[-1]["slots"].append({
            "i": si,
            "kind": s.kind,
            "x": round(float(s.x), 2),
            "state": state,
            "unnamed_reason": reason,
            "token": token,
            "label": label,
            "name": None if label is None else names.get(
                label, "label " + str(label)),
            "vision_label": int(d.train_label) if detected else None,
            "vision_name": (names.get(int(d.train_label),
                                      "label " + str(int(d.train_label)))
                            if detected else None),
            "vision_conf": round(float(s.confidence), 4) if detected else None,
            "lm_posterior": (round(float(lm_conf[si]), 4)
                             if si in lm_conf else None),
            # Index into this payload's `detections`, so the panes can link a
            # box to its position without matching coordinates.
            "detection_i": (None if box_of[si] is None
                            else int(box_of[si])),
            "box": (None if box_of[si] is None else {
                "x1": dets[box_of[si]]["x1"], "y1": dets[box_of[si]]["y1"],
                "x2": dets[box_of[si]]["x2"], "y2": dets[box_of[si]]["y2"],
            }),
        })
    return rows


def _sequence_dump(lines) -> list[dict[str, Any]]:
    """The CLI's sequence dump, one entry per line."""
    return [
        {
            "segment": r["segment"],
            "line": r["line"],
            "text": " ".join(s["token"] for s in r["slots"]),
        }
        for r in lines
    ]


def _empty_result(tablet, record, counts, conf, cell, lam, edge, lm_floor,
                  timings, t_start) -> dict[str, Any]:
    """A tablet whose slots never reached a four-slot block. Not an error."""
    timings["total"] = round(time.perf_counter() - t_start, 3)
    return {
        "tablet": tablet,
        "config": _config_block(conf, cell, lam, edge, lm_floor),
        "image": {"size": list(record["image_size"]),
                  "url": f"/api/tablets/{tablet}/image"},
        "counts": counts,
        "fusion": None,
        "lines": [],
        "sequence_dump": [],
        "detections": [],
        "domain3_input": None,
        "caveats": _caveats() + [
            "No block reached the minimum length of " + str(MIN_BLOCK) +
            " slots, so nothing was fused for this tablet."
        ] + (_upload_caveats(record) if _is_upload(tablet) else []),
        "timings": timings,
    }


# Domain 3: the tentative reading (not a translation)
#
# The ollama client ships with no timeout, so I give its shared client an
# explicit deadline. With a short connect deadline Ollama fails as a bare
# httpx timeout, with a longer one as ConnectionError, so I catch both.

# Long enough that Ollama's own "failed to connect" message comes through.
OLLAMA_CONNECT_TIMEOUT = 3.0

# Covers a whole generation, including a retry and the first-call model load.
OLLAMA_READ_TIMEOUT = 300.0

_SUMMARY_CACHE: dict[str, dict[str, Any]] = {}


class Domain3Unavailable(RuntimeError):
    """
    Base class for Domain 3 failures. Each subclass carries its HTTP status
    and a plain-language hint for the person reading it.
    """
    http_status = 502
    hint = ""


class OllamaUnreachable(Domain3Unavailable):
    http_status = 503
    hint = "Start the Ollama tray app, then try again."


class OllamaTimedOut(Domain3Unavailable):
    http_status = 504
    hint = ("Ollama accepted the request but did not answer in time. The "
            "model may still be loading into VRAM; the first call after idle "
            "takes about a minute.")


class Domain3Malformed(Domain3Unavailable):
    http_status = 502
    hint = ("Ollama answered but the response was not the JSON the schema "
            "asks for. This is a Domain 3 failure and is worth recording: "
            "structural validity is arm 1 of the Domain 3 evaluation.")


class NothingToFuse(RuntimeError):
    """No block reached the minimum length, so there is no reading to read."""


def bound_ollama_deadline(
    connect: float = OLLAMA_CONNECT_TIMEOUT,
    read: float = OLLAMA_READ_TIMEOUT,
) -> dict[str, Any]:
    """
    Give ollama's shared client an explicit deadline. Safe to call repeatedly.
    Returns what was applied, or why not, instead of failing.
    """
    try:
        import httpx
        import ollama

        client = getattr(ollama, "_client", None)
        http = getattr(client, "_client", None)
        if http is None:
            return {"applied": False,
                    "reason": "ollama._client._client not found; the client "
                              "layout changed. Calls are UNBOUNDED."}
        before = repr(getattr(http, "timeout", None))
        http.timeout = httpx.Timeout(connect=connect, read=read,
                                     write=30.0, pool=30.0)
        return {"applied": True, "connect": connect, "read": read,
                "was": before}
    except Exception as exc:                        # noqa: BLE001
        return {"applied": False, "reason": f"{type(exc).__name__}: {exc}"}


def _domain3_caveats() -> list[str]:
    """The caveats that must travel with every Domain 3 summary."""
    return [
        "THIS IS NOT A TRANSLATION. Domain 3 produces a genre-level gist from "
        "a sign sequence, which is why the field is called tentative_reading.",

        "Domain 2's top-1 accuracy is 22.2%, so most restored signs are wrong. "
        "A fluent reading here demonstrates the CHAIN, not that the tablet was "
        "read correctly.",

        "Both faithfulness checks below cover SIGN NAMES ONLY. A proper noun "
        "in the prose -- a king, a city, a date, a person -- is invisible to "
        "them, so a PASS is not evidence the prose is faithful. Read it "
        "against catalogue_metadata by eye.",

        "metadata_used is the model's own CLAIM. had_metadata is what was "
        "actually supplied. They can disagree, and when they do the claim is "
        "the unreliable one.",

        "Generation runs at temperature 0.2 with no fixed seed, so a second "
        "call can return a different summary. Unlike the analysis, this result "
        "is NOT reproducible, and that is why it is not presented as one.",
    ]


def _faithfulness(t, out: dict[str, Any]) -> dict[str, Any]:
    """
    Check whether the signs the model cited were really in what it was shown.

    There are three tiers: exact, matched after normalising the notation, and
    invented. I return the matched pairs, not just counts, so they can be checked.
    """
    supplied = t.all_names | t.displayed_names
    raw_cited = [s.strip() for s in out.get("signs_used", []) if s.strip()]
    cited = sorted(set(raw_cited))
    exact, normalised, invented = classify_citations(cited, supplied)

    if not cited:
        # Nothing cited means the check never ran, which is not a pass.
        verdict = "vacuous"
    elif invented:
        verdict = "fail"
    elif normalised:
        verdict = "pass_after_normalising"
    else:
        verdict = "pass"

    return {
        "supplied_distinct": len(supplied),
        "citations": len(raw_cited),
        "citations_distinct": len(cited),
        "exact": exact,
        "normalised": [[c, m] for c, m in normalised],
        "invented": invented,
        "verdict": verdict,
        "coverage_of_input": (
            round(len(set(cited) & supplied) / len(supplied), 4)
            if cited and supplied else None
        ),
    }


def summarise_tablet(tablet: str, regenerate: bool = False
                     ) -> tuple[dict[str, Any], bool]:
    """
    Domain 3 for one tablet: the tentative reading plus its audit.
    Returns (result, was_cached).

    The call goes through live_pipeline.call_domain3, which retries once on a
    truncated answer. Domain 3 is not deterministic, so this cache only means
    "the summary I already showed you"; `regenerate=True` makes a fresh call.
    """
    if not tablet_exists(tablet):
        raise FileNotFoundError(
            f"No detections for {tablet}. "
            f"{len(list_tablets())} tablets are available."
        )

    if not regenerate and tablet in _SUMMARY_CACHE:
        return _SUMMARY_CACHE[tablet], True

    result, t, _ = analyse_tablet_frozen_with_input(tablet)
    if t is None:
        raise NothingToFuse(
            f"{tablet} produced no block of at least {MIN_BLOCK} slots, so "
            f"there is no sign sequence for Domain 3 to read. This is a real "
            f"outcome of the pipeline, not an error."
        )

    deadline = bound_ollama_deadline()
    catalogue = _catalogue()

    t0 = time.perf_counter()
    try:
        # call_domain3 only prints when it retries; that belongs in the server log.
        out, attempts = call_domain3(t, catalogue)

    except ConnectionError as exc:
        # The usual case: Ollama down raises the builtin ConnectionError.
        raise OllamaUnreachable(str(exc)) from exc

    except ValueError as exc:
        # call_domain3 already retried once, so the retry failed too.
        raise Domain3Malformed(str(exc)) from exc

    except Exception as exc:                        # noqa: BLE001
        # Sorted by what the person has to do about it.
        import httpx

        if isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError)):
            # A short-deadline timeout, with no message from Ollama.
            raise OllamaUnreachable(
                f"{type(exc).__name__}: {exc or 'timed out'}"
            ) from exc
        if isinstance(exc, httpx.TimeoutException):
            raise OllamaTimedOut(
                f"{type(exc).__name__}: {exc or 'timed out'} "
                f"(read deadline {OLLAMA_READ_TIMEOUT:.0f}s)"
            ) from exc
        raise Domain3Unavailable(f"{type(exc).__name__}: {exc}") from exc

    seconds = round(time.perf_counter() - t0, 2)

    meta_text = result["domain3_input"]["catalogue_metadata"]
    had_meta = bool(result["domain3_input"]["had_metadata"])

    summary = {
        "tentative_reading": out.get("tentative_reading", ""),
        "confidence": out.get("confidence"),
        "signs_used": list(out.get("signs_used", [])),
        "uncertainty_notes": out.get("uncertainty_notes", ""),
        "historical_context": out.get("historical_context", ""),
        "metadata_used": bool(out.get("metadata_used", False)),
    }

    payload: dict[str, Any] = {
        "tablet": tablet,
        "model": D3_MODEL,
        "attempts": attempts,
        "seconds": seconds,
        # _hit_cap is only ever set when truncation happened.
        "hit_cap": bool(out.get("_hit_cap", False)),
        "schema_complete": all(
            k in out for k in ("tentative_reading", "confidence", "signs_used",
                               "uncertainty_notes", "historical_context",
                               "metadata_used")),
        "summary": summary,
        "metadata": {
            "had_metadata": had_meta,
            # What generate() says it supplied. Must match the analysis.
            "had_metadata_at_call": bool(out.get("_had_metadata", had_meta)),
            "claimed_metadata_used": summary["metadata_used"],
            "catalogue_metadata": meta_text,
        },
        "faithfulness": _faithfulness(t, out),
        "prose_scan": scan_prose(out, t.all_names | t.displayed_names),
        "positions": {
            "named": result["fusion"]["named_positions"],
            "total": result["fusion"]["total_positions"],
            "lines": len(result["domain3_input"]["lines"]),
        },
        "deadline": deadline,
        "caveats": _domain3_caveats(),
    }

    _SUMMARY_CACHE[tablet] = payload
    return payload, False


# Standalone check

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tablet", required=True)
    ap.add_argument("--json", action="store_true",
                    help="dump the whole payload instead of the summary.")
    args = ap.parse_args()

    t0 = time.perf_counter()
    res = analyse_tablet(args.tablet)
    elapsed = time.perf_counter() - t0

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    blob = json.dumps(res, ensure_ascii=False)

    print("=" * 78)
    print("analyse_tablet(" + repr(args.tablet) + ")")
    print("=" * 78)
    print("  elapsed           : {:.2f}s   {}".format(
        elapsed, "PASS (< 10 s)" if elapsed < 10 else "FAIL (>= 10 s)"))
    print("  COLD call: this includes the Domain 2 checkpoint load, which the")
    print("  server pays once at startup via warm(). A warm call is faster.")
    print("  JSON-serialisable : PASS ({:,} bytes)".format(len(blob)))
    print("  timings           : " + str(res["timings"]))
    print("  counts            : " + str(res["counts"]))
    print("  fusion            : " + str(res["fusion"]))
    print("\n  THE FUSED SEQUENCE")
    for row in res["sequence_dump"]:
        print("\n  segment {}  line {}".format(row["segment"], row["line"]))
        print("    " + row["text"])
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
