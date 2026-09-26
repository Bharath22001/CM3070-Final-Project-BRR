"""
Trains my Domain 1 detector locally with the same settings as my Colab notebook.

It runs on the AMD GPU through native ROCm PyTorch, which Ultralytics sees as a
normal CUDA device, so none of the DirectML patches in train_yolo_baseline.py
are needed. Use the fyp_env_rocm environment for GPU training; fyp_env stays
on CPU torch for Domains 2 and 3.

Usage:
    python -m cuneiform.domain1_vision.train_yolo_local --epochs 3   # smoke test
    python -m cuneiform.domain1_vision.train_yolo_local              # full run
    python -m cuneiform.domain1_vision.train_yolo_local --device cpu # fallback
"""

import argparse

from ultralytics import YOLO

from cuneiform import config

# Local COCO weights, so Ultralytics doesn't download them again.
PRETRAINED = config.PROJECT_ROOT / "yolov8s.pt"

RUNS_DIR = config.PROJECT_ROOT / "runs"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--device",
        default="0",
        help="'0' for the ROCm GPU (default), or 'cpu' for the fallback run.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Epoch budget. Use a small value (e.g. 3) for a smoke test.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Dataloader workers. Default 0 (load in the main process), which is "
            "deliberate and matters on this machine. Windows spawns workers "
            "rather than forking them, so with --cache each worker gets its OWN "
            "copy of the decoded dataset: at workers=4 Ultralytics ran ~10 "
            "processes and drove free RAM from 15.8 GB to 1.1 GB. Since the "
            "pagefile lives on the same mechanical HDD we are caching to avoid, "
            "swapping would cost far more than single-threaded loading does. "
            "One process + one RAM cache + zero disk reads is the fast "
            "configuration here. Raise this only with --no-cache."
        ),
    )
    p.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size. 16 matches the Colab run.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Do not hold the decoded dataset in RAM. Caching is ON by default "
            "because D: is a mechanical HDD (WD Blue, SATA) and Ultralytics "
            "reports ~1.7 MB/s on these 1019 small tiles, which dominates "
            "epoch time. Decoded the whole set is ~764 MB, which one process "
            "absorbs comfortably -- but only ONE process, hence workers=0 by "
            "default. If you pass --no-cache, raise --workers to parallelise "
            "the disk reads instead."
        ),
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help=(
            "Disable mixed precision. Ultralytics runs an AMP self-check that "
            "downloads yolo11n.pt and compares fp16/fp32 outputs; if that check "
            "fails or hangs on ROCm, pass this. Note that the Colab reference "
            "run used amp=True, so this is a deviation worth recording."
        ),
    )
    p.add_argument(
        "--name",
        default="domain1_local_v1",
        help="Run name under runs/.",
    )
    args = p.parse_args()

    if not PRETRAINED.exists():
        raise SystemExit(f"Pretrained weights not found: {PRETRAINED}")
    if not config.YOLO_DATA_YAML.exists():
        raise SystemExit(f"Dataset config not found: {config.YOLO_DATA_YAML}")

    print(f"data    : {config.YOLO_DATA_YAML}")
    print(f"weights : {PRETRAINED}")
    print(f"device  : {args.device}")
    print(f"epochs  : {args.epochs}")

    model = YOLO(str(PRETRAINED))

    # Same hyperparameters as the Colab run so the two are comparable.
    # Flips stay at 0 because a mirrored wedge cluster is a different sign (or
    # none), imgsz matches the tile size so nothing is resampled, and lr0 is set
    # by hand because the auto-picked 2.8e-05 was too low for a new 375-class head.
    model.train(
        data=str(config.YOLO_DATA_YAML),
        epochs=args.epochs,
        imgsz=512,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.001,
        fliplr=0.0,
        flipud=0.0,
        patience=40,
        seed=0,
        device=args.device,
        workers=args.workers,
        amp=not args.no_amp,
        cache=False if args.no_cache else "ram",
        project=str(RUNS_DIR),
        name=args.name,
        plots=True,
    )

    print("\n--- TRAINING COMPLETE ---")
    print(f"weights: {RUNS_DIR / args.name / 'weights' / 'best.pt'}")
    print(f"metrics: {RUNS_DIR / args.name / 'results.csv'}")


if __name__ == "__main__":
    main()
