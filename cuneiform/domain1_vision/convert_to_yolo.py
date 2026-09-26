"""
Converts my tiled annotations into the folder layout and label format that
Ultralytics YOLOv8 expects.

Input:  cuneiform_data/tiled_annotations_{train,test}.csv (boxes relative to the 512 px tile)
Output: yolo_dataset/images/{train,val}/*.jpg and yolo_dataset/labels/{train,val}/*.txt,
        one "<class_id> <x_center> <y_center> <w> <h>" line per box, normalised to 0-1.

There are 375 classes (train_label 0 to 374). Class 0 is the annotators'
"broken sign" placeholder and almost never has a box.
"""

import os
import ast
import shutil

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

# Configuration
DATA_DIR = "./cuneiform_data"
TILES_DIR = os.path.join(DATA_DIR, "tiles_512")
TRAIN_CSV = os.path.join(DATA_DIR, "tiled_annotations_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "tiled_annotations_test.csv")

YOLO_DIR = "./yolo_dataset"
NUM_CLASSES = 375
TILE_SIZE = 512


def setup_directories():
    for split in ('train', 'val'):
        os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)


def convert_and_copy(csv_path, split_name):
    """Write one split's images and YOLO label files. Returns a stats dict."""
    if not os.path.exists(csv_path):
        print(f"Error: could not find {csv_path}")
        return None

    print(f"\nProcessing '{split_name}' from {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)

    if 'bbox_xyxy' not in df.columns:
        raise ValueError(
            f"{csv_path} has no 'bbox_xyxy' column. It was probably produced by the "
            f"old buggy generate_tiles.py. Re-run generate_tiles.py first."
        )

    df['bbox_xyxy'] = df['bbox_xyxy'].apply(ast.literal_eval)

    written_boxes = 0
    dropped_boxes = 0
    written_tiles = 0
    missing_tiles = 0
    widths, heights = [], []

    for tile_name, group in tqdm(df.groupby('tile_name'), desc=f"  {split_name}"):
        src_img = os.path.join(TILES_DIR, f"{tile_name}.jpg")
        if not os.path.exists(src_img):
            missing_tiles += 1
            continue

        with Image.open(src_img) as im:
            img_w, img_h = im.size

        lines = []
        for _, row in group.iterrows():
            x1, y1, x2, y2 = row['bbox_xyxy']

            # Clamp to the tile, then discard anything degenerate.
            x1 = max(0.0, float(x1))
            y1 = max(0.0, float(y1))
            x2 = min(float(img_w), float(x2))
            y2 = min(float(img_h), float(y2))

            w = x2 - x1
            h = y2 - y1
            if w <= 1.0 or h <= 1.0:
                dropped_boxes += 1
                continue

            class_id = int(row['train_label'])
            if not (0 <= class_id < NUM_CLASSES):
                raise ValueError(
                    f"train_label {class_id} is outside 0..{NUM_CLASSES - 1}. "
                    f"Raise NUM_CLASSES rather than dropping the label."
                )

            lines.append(
                f"{class_id} "
                f"{(x1 + w / 2.0) / img_w:.6f} "
                f"{(y1 + h / 2.0) / img_h:.6f} "
                f"{w / img_w:.6f} "
                f"{h / img_h:.6f}"
            )
            widths.append(w)
            heights.append(h)
            written_boxes += 1

        if not lines:
            continue

        shutil.copy(src_img, os.path.join(YOLO_DIR, 'images', split_name, f"{tile_name}.jpg"))
        with open(os.path.join(YOLO_DIR, 'labels', split_name, f"{tile_name}.txt"), 'w') as f:
            f.write("\n".join(lines) + "\n")
        written_tiles += 1

    return {
        'split': split_name,
        'tiles': written_tiles,
        'boxes': written_boxes,
        'dropped': dropped_boxes,
        'missing_tiles': missing_tiles,
        'classes': int(df['train_label'].nunique()),
        'median_w': float(np.median(widths)) if widths else 0.0,
        'median_h': float(np.median(heights)) if heights else 0.0,
        'frac_huge': float(np.mean((np.array(widths) * np.array(heights)) >
                                   (TILE_SIZE * TILE_SIZE * 0.5))) if widths else 0.0,
    }


def generate_yaml():
    yaml_content = {
        'path': os.path.abspath(YOLO_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': NUM_CLASSES,
        'names': {i: f"class_{i}" for i in range(NUM_CLASSES)},
    }
    yaml_path = os.path.join(YOLO_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"\nWrote {yaml_path} (nc={NUM_CLASSES})")


def report(stats):
    print("\n" + "=" * 60)
    print("--- CONVERSION COMPLETE ---")
    for s in stats:
        if s is None:
            continue
        print(f"\n[{s['split']}]")
        print(f"  tiles written      : {s['tiles']}")
        print(f"  boxes written      : {s['boxes']}")
        print(f"  boxes dropped      : {s['dropped']}  (degenerate after clamping)")
        print(f"  tiles missing image: {s['missing_tiles']}")
        print(f"  distinct classes   : {s['classes']}")
        print(f"  median box size    : {s['median_w']:.0f} x {s['median_h']:.0f} px")
        print(f"  boxes >50% of tile : {s['frac_huge'] * 100:.1f}%   (should be ~0%)")
    print("\n" + "=" * 60)
    print("SANITY CHECK: median box size should be roughly 50-90 px on each side.")
    print("If it is in the hundreds, the coordinate convention is wrong again.")
    print("=" * 60)


if __name__ == "__main__":
    print("Initialising YOLOv8 dataset conversion...")
    setup_directories()
    stats = [
        convert_and_copy(TRAIN_CSV, 'train'),
        convert_and_copy(TEST_CSV, 'val'),
    ]
    generate_yaml()
    report(stats)