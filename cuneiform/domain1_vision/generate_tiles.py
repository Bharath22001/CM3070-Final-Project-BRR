"""
Slices the full-resolution tablet photographs into overlapping 512x512 tiles
and re-expresses every bounding box relative to the tile it lands in.

The source CSVs store `bbox` as absolute corners [x1, y1, x2, y2], not
[x, y, w, h]. Reading them the wrong way inflates every box and makes the 60%
area rule throw most of them away, so I name the output column `bbox_xyxy`
and check the convention when loading.
"""

import os
import ast
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

# Configuration
IMG_DIR = "cuneiform_data/images"
ANNOTATIONS_DIR = "cuneiform_data/cuneiform-sign-detection-dataset/annotations"

TRAIN_BBOX_CSV = os.path.join(ANNOTATIONS_DIR, "bbox_annotations_train_full.csv")
TEST_BBOX_CSV = os.path.join(ANNOTATIONS_DIR, "bbox_annotations_test_full.csv")

TILES_DIR = "cuneiform_data/tiles_512"
TRAIN_TILES_CSV_PATH = "cuneiform_data/tiled_annotations_train.csv"
TEST_TILES_CSV_PATH = "cuneiform_data/tiled_annotations_test.csv"

TILE_SIZE = 512
OVERLAP = 128
STEP_SIZE = TILE_SIZE - OVERLAP          # 384
MIN_AREA_RATIO = 0.60                    # keep a box only if >=60% of it survives the crop


def load_annotations(csv_path):
    """
    Load one annotation CSV and parse its bbox column. I check that x2 > x1
    and y2 > y1, and that the implied sign size is tens of pixels (not
    thousands), which is what gives away an [x, y, w, h] file.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    required = {'bbox', 'tablet_CDLI', 'train_label'}
    missing = required - set(df.columns)
    if missing:
        print(f"Warning: {csv_path} missing columns {missing}, skipping.")
        return pd.DataFrame()

    # The test CSV lists 134 boxes twice (all on P336150). I drop the duplicates
    # so my box counts match what the trainer actually loads.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} duplicate row(s) from {os.path.basename(csv_path)}")

    df['bbox'] = df['bbox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    arr = np.asarray(df['bbox'].tolist(), dtype=float)
    if not ((arr[:, 2] > arr[:, 0]).all() and (arr[:, 3] > arr[:, 1]).all()):
        raise ValueError(
            f"{csv_path}: bbox column is not [x1, y1, x2, y2] -- found rows where "
            f"x2 <= x1 or y2 <= y1. Do not proceed until this is understood."
        )

    widths = arr[:, 2] - arr[:, 0]
    heights = arr[:, 3] - arr[:, 1]
    print(f"  {os.path.basename(csv_path)}: {len(df)} boxes | "
          f"median sign size {np.median(widths):.0f} x {np.median(heights):.0f} px")

    if np.median(widths) > 400 or np.median(heights) > 400:
        raise ValueError(
            "Median sign size is implausibly large. The bbox convention is probably "
            "being misread again. Stop and check the source CSV."
        )

    return df


def tile_one_tablet(tablet_id, group, tiles_dir):
    """
    Slice one tablet into tiles and return the annotation rows that survive.
    I keep the boxes in a numpy array because per-row pandas lookups were far too slow.
    """
    img_path = os.path.join(IMG_DIR, f"{tablet_id}.jpg")
    if not os.path.exists(img_path):
        return [], 0

    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img_w, img_h = img.size

    boxes = np.asarray(group['bbox'].tolist(), dtype=float)   # (N, 4) as x1,y1,x2,y2
    labels = group['train_label'].to_numpy()
    mzl = group['mzl_label'].to_numpy() if 'mzl_label' in group.columns else np.full(len(group), -1)
    segm = group['segm_idx'].to_numpy() if 'segm_idx' in group.columns else np.full(len(group), -1)

    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    rows = []
    tiles_saved = 0

    for y in range(0, img_h - TILE_SIZE + 1, STEP_SIZE):
        for x in range(0, img_w - TILE_SIZE + 1, STEP_SIZE):
            tx2, ty2 = x + TILE_SIZE, y + TILE_SIZE

            # Intersection of every box with this tile, vectorised.
            ix1 = np.maximum(boxes[:, 0], x)
            iy1 = np.maximum(boxes[:, 1], y)
            ix2 = np.minimum(boxes[:, 2], tx2)
            iy2 = np.minimum(boxes[:, 3], ty2)

            iw = np.clip(ix2 - ix1, 0, None)
            ih = np.clip(iy2 - iy1, 0, None)
            inter = iw * ih

            keep = (inter / box_areas) >= MIN_AREA_RATIO
            if not keep.any():
                continue

            tile_name = f"{tablet_id}_tile_{x}_{y}"
            img.crop((x, y, tx2, ty2)).save(os.path.join(tiles_dir, f"{tile_name}.jpg"))
            tiles_saved += 1

            for i in np.flatnonzero(keep):
                # Clipped box, relative to the tile's top-left corner.
                rows.append({
                    'tile_name': tile_name,
                    'tablet_CDLI': tablet_id,
                    'segm_idx': int(segm[i]),
                    'train_label': int(labels[i]),
                    'mzl_label': int(mzl[i]),
                    'bbox_xyxy': str([
                        float(ix1[i] - x), float(iy1[i] - y),
                        float(ix2[i] - x), float(iy2[i] - y),
                    ]),
                })

    return rows, tiles_saved


def tile_split(df, split_name, exclude_tablet_ids=None):
    if df.empty:
        print(f"No annotations to process for split '{split_name}'.")
        return []

    exclude_tablet_ids = exclude_tablet_ids or set()
    grouped = df.groupby('tablet_CDLI')

    all_rows = []
    total_tiles = 0
    skipped_leakage = 0
    missing_images = 0

    print(f"\nSlicing '{split_name}': {len(grouped)} tablets into "
          f"{TILE_SIZE}x{TILE_SIZE} tiles (step {STEP_SIZE}px)...")

    for tablet_id, group in tqdm(grouped):
        if tablet_id in exclude_tablet_ids:
            skipped_leakage += 1
            continue

        rows, tiles_saved = tile_one_tablet(tablet_id, group, TILES_DIR)
        if tiles_saved == 0 and not rows:
            missing_images += 1
        all_rows.extend(rows)
        total_tiles += tiles_saved

    print(f"'{split_name}' complete: {total_tiles} tiles, {len(all_rows)} boxes.")
    if skipped_leakage:
        print(f"  Skipped {skipped_leakage} tablet(s) present in both source splits.")
    if missing_images:
        print(f"  {missing_images} tablet(s) had no downloaded image and were skipped.")

    return all_rows


def process_images():
    os.makedirs(TILES_DIR, exist_ok=True)

    print("Loading source annotations...")
    train_df = load_annotations(TRAIN_BBOX_CSV)
    test_df = load_annotations(TEST_BBOX_CSV)

    train_tablets = set(train_df['tablet_CDLI'].unique()) if not train_df.empty else set()
    test_tablets = set(test_df['tablet_CDLI'].unique()) if not test_df.empty else set()
    overlap = train_tablets & test_tablets

    if overlap:
        print(f"\nWARNING: {len(overlap)} tablet(s) appear in BOTH source CSVs: {sorted(overlap)}")
        print("Excluding them from TRAIN so the evaluation split stays clean.")

    train_rows = tile_split(train_df, "train", exclude_tablet_ids=overlap)
    test_rows = tile_split(test_df, "test")

    train_out = pd.DataFrame(train_rows)
    test_out = pd.DataFrame(test_rows)
    train_out.to_csv(TRAIN_TILES_CSV_PATH, index=False)
    test_out.to_csv(TEST_TILES_CSV_PATH, index=False)

    print("\n" + "=" * 60)
    print("--- TILING COMPLETE ---")
    print(f"Train tiles: {train_out['tile_name'].nunique() if not train_out.empty else 0}"
          f" | boxes: {len(train_out)}")
    print(f"Test tiles:  {test_out['tile_name'].nunique() if not test_out.empty else 0}"
          f" | boxes: {len(test_out)}")

    if not train_out.empty:
        combined = pd.concat([train_out, test_out]) if not test_out.empty else train_out
        print(f"Distinct sign classes present: {combined['train_label'].nunique()}")
        print(f"Highest train_label id: {combined['train_label'].max()}")

    print(f"\nSaved: {TRAIN_TILES_CSV_PATH}")
    print(f"Saved: {TEST_TILES_CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    process_images()