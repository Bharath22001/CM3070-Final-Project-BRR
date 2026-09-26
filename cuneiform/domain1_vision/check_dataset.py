"""
Finds files in yolo_dataset/ that my current pipeline did not write, by
comparing against the tile list my CSVs actually produce. I used it to catch
leftovers from an earlier buggy run that shared the same tile names.
"""
import os
import pandas as pd

YOLO_DIR = "./yolo_dataset"
CSVS = {
    'train': "./cuneiform_data/tiled_annotations_train.csv",
    'val':   "./cuneiform_data/tiled_annotations_test.csv",
}

for split, csv_path in CSVS.items():
    expected = set(pd.read_csv(csv_path)['tile_name'].unique())

    img_dir = os.path.join(YOLO_DIR, 'images', split)
    lbl_dir = os.path.join(YOLO_DIR, 'labels', split)

    on_disk_imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')}
    on_disk_lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')}

    stale_imgs = on_disk_imgs - expected
    stale_lbls = on_disk_lbls - expected

    print(f"\n=== {split} ===")
    print(f"  expected from CSV : {len(expected)}")
    print(f"  images on disk    : {len(on_disk_imgs)}")
    print(f"  labels on disk    : {len(on_disk_lbls)}")
    print(f"  STALE images      : {len(stale_imgs)}")
    print(f"  STALE labels      : {len(stale_lbls)}")

    for name in sorted(stale_imgs | stale_lbls):
        lbl = os.path.join(lbl_dir, name + '.txt')
        n_boxes = sum(1 for _ in open(lbl)) if os.path.exists(lbl) else 0
        print(f"    - {name}  ({n_boxes} boxes)")