"""
Checks whether a downloaded best.pt is the wrong checkpoint, or the right one
trained against a stale data.yaml, by comparing its class count with my label
files. It loads no images and trains nothing.

Run from the project root:
    python -m cuneiform.domain1_vision.diagnose_nc_mismatch "D:\\Downloads\\best.pt"
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

from cuneiform import config

RULE = "=" * 70


def head(t: str) -> None:
    print("\n" + RULE)
    print(t)
    print(RULE)


def verdict(name: str, ok: bool | None, note: str = "") -> None:
    tag = "PASS" if ok is True else ("FAIL" if ok is False else "VACUOUS")
    print(f"\n  --> {tag}: {name}")
    if note:
        print(f"      {note}")


def section_a(ckpt_path: str) -> int:
    head("A. WHAT nc DID THIS CHECKPOINT ACTUALLY TRAIN WITH?")

    p = Path(ckpt_path)
    if not p.exists():
        print(f"  {p} does not exist.")
        return -1

    ck = torch.load(p, map_location="cpu", weights_only=False)
    model = ck.get("model")

    # I read the class count three ways; the detection head's nc is the one that can't be stale.
    names = getattr(model, "names", None)
    n_names = len(names) if names is not None else -1
    model_nc = getattr(model, "nc", None)

    head_nc = None
    try:
        head_nc = int(model.model[-1].nc)
    except Exception as exc:
        print(f"  (could not read detection head nc: {exc})")

    print(f"  len(model.names)        : {n_names}")
    print(f"  model.nc                : {model_nc}")
    print(f"  detection head .nc      : {head_nc}   <-- AUTHORITATIVE")

    args = ck.get("train_args") or ck.get("args") or {}
    if hasattr(args, "__dict__"):
        args = vars(args)
    print(f"\n  trained against data    : {args.get('data')}")
    print(f"  epochs / imgsz / batch  : {args.get('epochs')} / "
          f"{args.get('imgsz')} / {args.get('batch')}")
    print(f"  project / name          : {args.get('project')} / {args.get('name')}")
    print(f"  epoch saved             : {ck.get('epoch')}")
    print(f"  best_fitness            : {ck.get('best_fitness')}")
    print(f"  date                    : {ck.get('date')}")

    nc = head_nc if head_nc is not None else n_names
    verdict("checkpoint was trained with nc = 375", nc == 375,
            f"nc = {nc}. Classes {nc}..374 are NOT REPRESENTABLE by this "
            f"model's output layer -- it cannot predict them at all.")
    return nc


def section_b() -> int:
    head("B. WHAT DOES THE LOCAL data.yaml SAY NOW?")

    if not config.YOLO_DATA_YAML.exists():
        print(f"  {config.YOLO_DATA_YAML} not found.")
        return -1
    doc = yaml.safe_load(config.YOLO_DATA_YAML.read_text(encoding="utf-8"))
    nc = doc.get("nc")
    print(f"  {config.YOLO_DATA_YAML}")
    print(f"  nc      : {nc}")
    print(f"  train   : {doc.get('train')}")
    print(f"  val     : {doc.get('val')}")
    print(f"  names   : {len(doc.get('names', {}))} entries")

    verdict("the local data.yaml is the fixed one", nc == 375,
            f"nc = {nc}. If this is 375 but the checkpoint is 354, the "
            f"checkpoint predates the fix or was trained from a stale zip.")
    return nc


def section_c(nc_ckpt: int) -> None:
    head("C. HOW MUCH DATA WOULD nc=354 HAVE DISCARDED?")

    if nc_ckpt <= 0:
        verdict("data loss estimate", None, "checkpoint nc unknown.")
        return

    counts: Counter = Counter()
    tiles_total = tiles_affected = 0
    boxes_total = boxes_over = 0

    for split in ("train", "val"):
        d = config.YOLO_DIR / "labels" / split
        if not d.exists():
            print(f"  {d} missing -- skipping {split}")
            continue
        for f in d.glob("*.txt"):
            tiles_total += 1
            over = False
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                cid = int(line.split()[0])
                counts[cid] += 1
                boxes_total += 1
                if cid >= nc_ckpt:
                    boxes_over += 1
                    over = True
            tiles_affected += over

    print(f"  label files scanned : {tiles_total:,}")
    print(f"  boxes               : {boxes_total:,}")
    print(f"  max class id present: {max(counts) if counts else 'n/a'}")
    print(f"\n  boxes with class >= {nc_ckpt} : {boxes_over:,} "
          f"({100 * boxes_over / max(1, boxes_total):.2f}%)")
    print(f"  tiles containing one   : {tiles_affected:,} "
          f"({100 * tiles_affected / max(1, tiles_total):.2f}%)")

    if boxes_over:
        print(f"\n  the offending classes:")
        for cid, n in sorted((c, n) for c, n in counts.items() if c >= nc_ckpt):
            print(f"    class {cid:>4} : {n:>5} boxes")

    verdict(f"no training data sits above the checkpoint's nc", boxes_over == 0,
            f"{boxes_over} box(es) across {tiles_affected} tile(s) carry a "
            f"class this model cannot represent. Ultralytics warns and skips "
            f"these, so the recorded box counts and mAP describe a REDUCED "
            f"dataset.")


def section_d(nc_ckpt: int) -> None:
    head("D. CAN THE CHECKPOINT STILL EMIT ALL 82 EMIT CLASSES?")

    vocab_path = config.D2_DIR / "vocab.json"
    if not vocab_path.exists():
        verdict("emit-set containment", None, f"{vocab_path} not found.")
        return

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    emit = sorted(int(x) for x in vocab["vision_emit_classes"])

    print(f"  emit classes      : {len(emit)}")
    print(f"  min / max         : {min(emit)} / {max(emit)}")

    unreachable = [c for c in emit if c >= nc_ckpt] if nc_ckpt > 0 else []
    print(f"  >= checkpoint nc  : {len(unreachable)}  {unreachable}")

    verdict("every emit class is inside the checkpoint's output layer",
            not unreachable,
            f"{len(unreachable)} emit class(es) can never be predicted by this "
            f"checkpoint. The simulator assumes all 82 are reachable, so "
            f"simulated and live results would not be comparable."
            if unreachable else
            "the emit set survives nc=354 intact -- the vocabulary contract is "
            "unaffected even if the dataset was reduced.")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if len(sys.argv) < 2:
        print(__doc__)
        return

    print(RULE)
    print("nc MISMATCH DIAGNOSIS")
    print(RULE)

    nc_ckpt = section_a(sys.argv[1])
    section_b()
    section_c(nc_ckpt)
    section_d(nc_ckpt)

    print("\n" + RULE)
    print("  READING THIS:")
    print("  - D PASS + C small  -> the checkpoint is USABLE with a stated")
    print("    limitation. Proceed with live wiring; report the reduction.")
    print("  - D FAIL            -> retrain, or find the right checkpoint.")
    print("    Do not wire a detector that cannot emit its own emit set.")
    print(RULE)


if __name__ == "__main__":
    main()