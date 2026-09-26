"""
Tells my YOLO checkpoints apart by reading what is stored inside each .pt
(training args, best fitness, class count) instead of trusting the filename.
A checkpoint with nc != 375 predates my label fix and is the wrong one.

Run from the project root, passing every candidate path:
    python -m cuneiform.domain1_vision.identify_checkpoint "C:\\path\\one\\best.pt" "D:\\path\\two\\best.pt"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def describe(path: str) -> None:
    p = Path(path)
    print("\n" + "-" * 70)
    print(f"  {p}")
    print("-" * 70)
    if not p.exists():
        print("  DOES NOT EXIST")
        return

    print(f"  size      : {p.stat().st_size / 1e6:.1f} MB")
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  COULD NOT LOAD: {type(exc).__name__}: {exc}")
        return

    if not isinstance(ck, dict):
        print(f"  unexpected checkpoint type: {type(ck)}")
        return

    print(f"  keys      : {sorted(ck)[:10]}")
    print(f"  epoch     : {ck.get('epoch')}")
    print(f"  fitness   : {ck.get('best_fitness')}")
    print(f"  date      : {ck.get('date')}")

    args = ck.get("train_args") or ck.get("args") or {}
    if hasattr(args, "__dict__"):
        args = vars(args)
    for k in ("data", "epochs", "imgsz", "batch", "device", "project", "name"):
        if k in args:
            print(f"  arg {k:<8}: {args[k]}")

    model = ck.get("model")
    names = getattr(model, "names", None) if model is not None else None
    if names is not None:
        print(f"  nc        : {len(names)}   "
              f"<-- MUST BE 375 for the bugfix_v2 run")
        sample = [names[i] for i in sorted(names)[:3]] if isinstance(names, dict) \
            else list(names)[:3]
        print(f"  first names: {sample}")
    else:
        print("  nc        : could not read class names")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    print("=" * 70)
    print("YOLO CHECKPOINT IDENTIFICATION")
    print("=" * 70)
    for a in sys.argv[1:]:
        describe(a)
    print("\n" + "=" * 70)
    print("  The one we want: nc = 375, epoch around 121, best_fitness set.")
    print("  Anything with nc = 354 predates the bbox fix -- do not use it.")
    print("=" * 70)


if __name__ == "__main__":
    main()