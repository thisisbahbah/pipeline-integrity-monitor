"""
prepare_dataset.py
------------------
Splits CODEBRIM (or synthetic) images into train / val / test sets,
applies augmentation to the training split, and writes a manifest CSV.

Splits: 70% train | 15% val | 15% test (stratified per class)

Output:
  data/processed/train/<class>/<image>.jpg
  data/processed/val/<class>/<image>.jpg
  data/processed/test/<class>/<image>.jpg
  data/manifest.csv   ← columns: split, class, filepath, label_idx

Usage:
    python scripts/prepare_dataset.py [--img-size 224]
"""

import shutil
import random
import csv
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

RAW_DIR     = Path("data/raw/CODEBRIM")
PROC_DIR    = Path("data/processed")
MANIFEST    = Path("data/manifest.csv")

CLASSES = [
    "background",
    "crack",
    "spallation",
    "exposed_bars",
    "corrosion_stain",
    "efflorescence",
]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15 (remainder)

SEED = 42

TRAIN_AUGMENT = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.2),
])

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(class_dir: Path) -> List[Path]:
    return [
        p for p in sorted(class_dir.rglob("*"))
        if p.suffix.lower() in IMG_EXTENSIONS
    ]


def split_list(items: List, train_r: float, val_r: float, seed: int) -> Tuple[List, List, List]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def process_split(
    images: List[Path],
    split: str,
    cls: str,
    label_idx: int,
    img_size: int,
    rows: list,
    augment: bool = False,
) -> None:
    out_dir = PROC_DIR / split / cls
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(images, desc=f"  [{split}] {cls}", leave=False):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [warn] Could not read {img_path}, skipping")
            continue

        img = resize_to_square(img, img_size)

        if augment:
            img = TRAIN_AUGMENT(image=img)["image"]

        out_path = out_dir / img_path.name
        # Avoid filename collisions in synthetic data across augmented variants
        if out_path.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            out_path = out_dir / f"{stem}_aug{suffix}"

        cv2.imwrite(str(out_path), img)
        rows.append({
            "split":     split,
            "class":     cls,
            "filepath":  str(out_path),
            "label_idx": label_idx,
        })


def main(img_size: int = 224) -> None:
    if not RAW_DIR.exists():
        print(
            f"[prepare] {RAW_DIR} not found.\n"
            "  Run: python scripts/download_data.py  first."
        )
        return

    print(f"[prepare] Image size: {img_size}×{img_size}")
    print(f"[prepare] Splits: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {1-TRAIN_RATIO-VAL_RATIO:.0%} test\n")

    manifest_rows = []

    for label_idx, cls in enumerate(CLASSES):
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            print(f"[prepare] Class dir missing: {cls_dir} — skipping")
            continue

        images = find_images(cls_dir)
        if not images:
            print(f"[prepare] No images found in {cls_dir} — skipping")
            continue

        train_imgs, val_imgs, test_imgs = split_list(images, TRAIN_RATIO, VAL_RATIO, SEED)
        print(f"  {cls}: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")

        process_split(train_imgs, "train", cls, label_idx, img_size, manifest_rows, augment=True)
        process_split(val_imgs,   "val",   cls, label_idx, img_size, manifest_rows, augment=False)
        process_split(test_imgs,  "test",  cls, label_idx, img_size, manifest_rows, augment=False)

    # Write manifest
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "filepath", "label_idx"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    total = len(manifest_rows)
    train_n = sum(1 for r in manifest_rows if r["split"] == "train")
    val_n   = sum(1 for r in manifest_rows if r["split"] == "val")
    test_n  = sum(1 for r in manifest_rows if r["split"] == "test")

    print(f"\n[prepare] Manifest written: {MANIFEST}")
    print(f"[prepare] Total images: {total}  ({train_n} train / {val_n} val / {test_n} test)")
    print("[prepare] Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()
    main(img_size=args.img_size)