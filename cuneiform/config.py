"""
I keep every path and shared constant for the project in this one file.

Paths are built from this file's own location, so a script finds the same
files whether I run it from the project root, VS Code or a notebook.
"""

from pathlib import Path

# This file sits at <root>/cuneiform/config.py, so the root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source data (read-only)
DATA_DIR = PROJECT_ROOT / "cuneiform_data"
DATASET_DIR = DATA_DIR / "cuneiform-sign-detection-dataset"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
TRANSLITERATIONS_DIR = DATASET_DIR / "transliterations"
IMAGES_DIR = DATA_DIR / "images"
TILES_DIR = DATA_DIR / "tiles_512"

TRAIN_BBOX_CSV = ANNOTATIONS_DIR / "bbox_annotations_train_full.csv"
TEST_BBOX_CSV = ANNOTATIONS_DIR / "bbox_annotations_test_full.csv"

TILED_TRAIN_CSV = DATA_DIR / "tiled_annotations_train.csv"
TILED_TEST_CSV = DATA_DIR / "tiled_annotations_test.csv"

# Domain 1 (vision)
YOLO_DIR = PROJECT_ROOT / "yolo_dataset"
YOLO_DATA_YAML = YOLO_DIR / "data.yaml"

# Generated artifacts (safe to delete and rebuild)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
D2_DIR = ARTIFACTS_DIR / "domain2"
D3_DIR = ARTIFACTS_DIR / "domain3"

D2_CORPUS_JSONL = D2_DIR / "corpus_lines.jsonl"
D2_CORPUS_STATS = D2_DIR / "corpus_stats.json"

D3_OGSL_DIR = D3_DIR / "ogsl"
D3_OGSL_ZIP = D3_OGSL_DIR / "osl-master.zip"
D3_OSL_ASL = D3_OGSL_DIR / "osl-master" / "00lib" / "osl.asl"
D3_MZL_NAMES = D3_DIR / "mzl_signnames.json"
D3_SIGN_NAMES = D3_DIR / "sign_names.json"
D3_SIGN_LOOKUP = D3_DIR / "sign_lookup.json"  
D3_ORACC_DIR = D3_DIR / "oracc"

# Domain 1 weights and saved detections
D1_DIR = ARTIFACTS_DIR / "domain1"
D1_WEIGHTS_DIR = D1_DIR / "weights"
D1_WEIGHTS = D1_WEIGHTS_DIR / "domain1_best.pt"
D1_LIVE_DETECTIONS = D1_DIR / "live_detections"

# Photographs uploaded through the dashboard. I keep these apart from
# D1_LIVE_DETECTIONS on purpose: that folder defines my 30 evaluation tablets,
# so an upload saved there would quietly change every evaluation result.
# Each upload stores <id>.json (detections), <id>.jpg (the working image the
# boxes are drawn on) and <id>.original.<ext> (the untouched upload).
UPLOADS_DIR = ARTIFACTS_DIR / "uploads"

# Tile size and step. These have to match the grid my training tiles were cut
# with; the inference grid only adds a flush final tile on top.
D1_TILE_SIZE = 512
D1_TILE_STEP = 384

# Number of classes in the deployed detector's head.
D1_CHECKPOINT_NC = 375

# Shared constants. train_label ids run from 0 to 374.
NUM_TRAIN_LABELS = 375

# train_label 0 is the annotators' "broken / unreadable sign" placeholder,
# not a real sign class.
BROKEN_LABEL = 0


def ensure_dirs() -> None:
    """Create the artifact directories if they do not already exist."""
    for d in (ARTIFACTS_DIR, D2_DIR, D3_DIR, D3_OGSL_DIR, D3_ORACC_DIR,
              D1_DIR, D1_WEIGHTS_DIR, D1_LIVE_DETECTIONS, UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)

def check_environment() -> None:
    """
    Warn me if the HuggingFace cache is not on D:.

    I only warn here because HuggingFace reads HF_HOME at import time, so the
    real fix is running start.ps1 before anything else.
    """
    import os

    hf = os.environ.get("HF_HOME", "")
    if not hf.upper().startswith("D:"):
        print("  WARNING: HF_HOME is not set to a D: path "
              f"(currently: {hf or 'unset'}).")
        print("  Model downloads may land on C:. Run .\\start.ps1 first.")