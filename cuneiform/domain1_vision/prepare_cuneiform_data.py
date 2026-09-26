import os
import ast
import pandas as pd
import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from transformers import AutoImageProcessor

# Directory setup
DATA_DIR = "./cuneiform_data"
IMG_DIR = os.path.join(DATA_DIR, "images")
# TILES_DIR is shared with the training and testing scripts
TILES_DIR = os.path.join(DATA_DIR, "tiles_512")

# PyTorch dataset
class CuneiformDataset(Dataset):
    """
    PyTorch Dataset for my tablet photographs (or tiles). Returns raw PIL images
    and COCO boxes for the Hugging Face AutoImageProcessor to process.
    """
    def __init__(self, bbox_csv, img_dir, augment=False):
        self.data = pd.read_csv(bbox_csv)
        self.img_dir = img_dir
        self.augment = augment
        
        # Albumentations augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                # Simulate dynamic museum lighting and flash photography
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                
                # Simulate camera angle and zoom; rotated corners are padded black
                A.ShiftScaleRotate(
                    shift_limit=0.05, 
                    scale_limit=0.1, 
                    rotate_limit=15, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0, 
                    p=0.2
                ),
                
                # Simulate camera lens grain
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.2))
        
        # Parse the stringified coordinate lists
        self.data['bbox'] = self.data['bbox'].apply(ast.literal_eval)
        
        # Group rows by image (now tile names)
        self.image_groups = self.data.groupby('tablet_CDLI')
        
        # Drop entries whose image is missing
        all_keys = list(self.image_groups.groups.keys())
        self.image_keys = []
        
        for key in all_keys:
            img_path = os.path.join(self.img_dir, f"{key}.jpg")
            if os.path.exists(img_path):
                self.image_keys.append(key)
        
        print(f"Found {len(self.image_keys)} valid images out of {len(all_keys)} CSV entries.")

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        p_number = self.image_keys[idx]
        img_path = os.path.join(self.img_dir, f"{p_number}.jpg")
        
        # Load the image and respect its EXIF rotation
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        img_width, img_height = image.size

        group = self.image_groups.get_group(p_number)
        
        # Extract raw boxes and categories
        bboxes = []
        category_ids = []
        
        for _, row in group.iterrows():
            x_min, y_min, w, h = row['bbox']
            
            # Convert normalised coordinates to absolute if needed
            if w <= 1.0 and h <= 1.0:
                x_min = x_min * img_width
                y_min = y_min * img_height
                w = w * img_width
                h = h * img_height
                
            # Clamp so no box runs off the image
            x_min = max(0.0, float(x_min))
            y_min = max(0.0, float(y_min))
            x_max = min(float(img_width), x_min + float(w))
            y_max = min(float(img_height), y_min + float(h))
            w = x_max - x_min
            h = y_max - y_min

            if w > 1.0 and h > 1.0:
                bboxes.append([x_min, y_min, w, h])
                category_ids.append(int(row['train_label']))
                
        # Apply augmentation
        if self.augment and len(bboxes) > 0:
            image_np = np.array(image)
            try:
                # Albumentations recomputes the boxes along with the image
                transformed = self.transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
                image_np = transformed['image']
                bboxes = transformed['bboxes']
                category_ids = transformed['category_ids']
                
                # Convert back to PIL for the Hugging Face processor
                image = Image.fromarray(image_np)
            except Exception:
                # Fall back to the original image if the transform drops too many boxes
                pass
                
        # Final size, in case augmentation shifted it
        final_w, final_h = image.size
                
        annotations = []
        for bbox, cat_id in zip(bboxes, category_ids):
            x, y, w, h = bbox
            
            # Rotations can push box corners outside the image, which made my
            # loss explode, so I clamp every coordinate back to the edge.
            x_min = max(0.0, float(x))
            y_min = max(0.0, float(y))
            x_max = min(float(final_w), float(x + w))
            y_max = min(float(final_h), float(y + h))
            
            clean_w = x_max - x_min
            clean_h = y_max - y_min
            
            # Only keep the box if it survived the clamping with a valid, visible area
            if clean_w > 1.0 and clean_h > 1.0:
                annotations.append({
                    "bbox": [x_min, y_min, clean_w, clean_h],
                    "category_id": int(cat_id),
                    "area": float(clean_w * clean_h),
                    "iscrowd": 0
                })
            
        target = {
            "image_id": idx,
            "annotations": annotations
        }

        return image, target

# Data collator
processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    batch_encoding = processor(images=images, annotations=targets, return_tensors="pt")
    return batch_encoding

# Quick test
if __name__ == "__main__":
    # Test against the tiled dataset
    train_bbox_path = "cuneiform_data/tiled_annotations_train.csv"
    
    if os.path.exists(train_bbox_path):
        # Turn augmentation on to check it works
        dataset = CuneiformDataset(bbox_csv=train_bbox_path, img_dir=TILES_DIR, augment=True)
        print(f"\nPipeline Ready: PyTorch Dataset initialized with {len(dataset)} tile images.")
        
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(dataloader))
        
        print(f"Processed Pixel Values Shape: {batch['pixel_values'].shape}")
        print(f"First Image Bounding Boxes Shape: {batch['labels'][0]['boxes'].shape}")
    else:
        print(f"Error: Could not find {train_bbox_path}. Please verify your paths.")