"""
Counts how many train and val boxes a 354-class checkpoint would have lost.
Ultralytics rejects a whole label file if any class id exceeds nc, so I count
every box in those tiles, not just the offending ones.

Run from the project root:
    python -m cuneiform.domain1_vision.quantify_nc_loss
"""

from __future__ import annotations

import sys

from cuneiform import config

CKPT_NC = 354


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 70)
    print(f"DATA LOST TO nc = {CKPT_NC}  (whole tiles, not just bad boxes)")
    print("=" * 70)

    grand = {"tiles": 0, "boxes": 0, "lost_tiles": 0, "lost_boxes": 0}

    for split in ("train", "val"):
        d = config.YOLO_DIR / "labels" / split
        if not d.exists():
            print(f"\n  {d} missing -- skipping")
            continue

        tiles = boxes = lost_tiles = lost_boxes = 0
        for f in sorted(d.glob("*.txt")):
            ids = [int(l.split()[0]) for l in f.read_text().splitlines()
                   if l.strip()]
            tiles += 1
            boxes += len(ids)
            if any(c >= CKPT_NC for c in ids):
                lost_tiles += 1
                lost_boxes += len(ids)   # the whole tile is rejected

        print(f"\n  [{split}]")
        print(f"    tiles written   : {tiles:,}")
        print(f"    boxes written   : {boxes:,}")
        print(f"    tiles REJECTED  : {lost_tiles:,} "
              f"({100 * lost_tiles / max(1, tiles):.2f}%)")
        print(f"    boxes LOST      : {lost_boxes:,} "
              f"({100 * lost_boxes / max(1, boxes):.2f}%)")
        print(f"    -> trainer actually saw {tiles - lost_tiles:,} tiles / "
              f"{boxes - lost_boxes:,} boxes")

        grand["tiles"] += tiles
        grand["boxes"] += boxes
        grand["lost_tiles"] += lost_tiles
        grand["lost_boxes"] += lost_boxes

    print("\n" + "-" * 70)
    print(f"  TOTAL: {grand['lost_tiles']:,} of {grand['tiles']:,} tiles and "
          f"{grand['lost_boxes']:,} of {grand['boxes']:,} boxes rejected")
    print(f"  ({100 * grand['lost_boxes'] / max(1, grand['boxes']):.2f}% of "
          f"annotated boxes never reached the model)")
    print("\n  My recorded figures are 730/7,450 train and 289/4,422 val.")
    print("  If the val figures above differ from 289/4,422, then the recorded")
    print("  'val instances' came from the WRITTEN data, not what the trainer")
    print("  loaded, and the mAP denominator needs restating.")
    print("=" * 70)

if __name__ == "__main__":
    main()