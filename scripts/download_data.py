"""
download_data.py
----------------
Downloads the CODEBRIM dataset from Kaggle.
If Kaggle credentials are unavailable, falls back to generating a synthetic
dataset using albumentations so the full pipeline still runs end-to-end.

Usage:
    python scripts/download_data.py [--synthetic]

CODEBRIM (COncrete DEfect BRidge IMage dataset):
    - 6 classes: background, crack, spallation, exposed_bars, corrosion_stain, efflorescence
    - Kaggle slug: arnav3105/codebrim-concrete-bridge-defects
"""

import argparse
import os
import sys
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path("data/raw")
CLASSES = [
    "background",
    "crack",
    "spallation",
    "exposed_bars",
    "corrosion_stain",
    "efflorescence",
]

KAGGLE_DATASET = "arnav3105/codebrim-concrete-bridge-defects"


def check_kaggle_credentials() -> bool:
    """Return True if kaggle.json is present and readable."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def download_from_kaggle() -> bool:
    """
    Attempt Kaggle download. Returns True on success, False on failure.

    Manual fallback (if Kaggle CLI fails):
    1. Go to: https://www.kaggle.com/datasets/arnav3105/codebrim-concrete-bridge-defects
    2. Click Download (requires free Kaggle account)
    3. Unzip into data/raw/ so the structure is:
       data/raw/CODEBRIM/<ClassName>/<image>.jpg
    """
    try:
        import kaggle  # noqa: F401

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[download] Fetching {KAGGLE_DATASET} from Kaggle...")
        os.system(
            f'kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --unzip'
        )
        print(f"[download] Dataset saved to {RAW_DIR}")
        return True
    except Exception as e:
        print(f"[download] Kaggle download failed: {e}")
        return False


def generate_synthetic_dataset(
    n_per_class: int = 200,
    img_size: int = 224,
) -> None:
    """
    Generate synthetic concrete-texture images per class using albumentations.
    This lets the entire ML pipeline run without real data.
    Images are visually distinct per class via colour and texture cues.
    """
    try:
        import albumentations as A
        import cv2
    except ImportError:
        print("[synthetic] albumentations / opencv not found. Install requirements first.")
        sys.exit(1)

    print(f"[synthetic] Generating {n_per_class} images per class × {len(CLASSES)} classes...")

    # Base colour per class (BGR for OpenCV)
    class_colours = {
        "background":       (180, 180, 180),
        "crack":            (140, 140, 140),
        "spallation":       (160, 140, 120),
        "exposed_bars":     (80, 100, 130),
        "corrosion_stain":  (60, 80, 160),
        "efflorescence":    (200, 195, 185),
    }

    # always_apply removed in albumentations 2.x — use p=1.0 instead
    augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.7),
        A.GridDistortion(p=0.3),
        A.ElasticTransform(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.RandomCrop(height=img_size, width=img_size, p=1.0),
    ])

    rng = np.random.default_rng(42)

    for cls in CLASSES:
        cls_dir = RAW_DIR / "CODEBRIM" / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        base_colour = class_colours[cls]

        for i in tqdm(range(n_per_class), desc=f"  {cls}", leave=False):
            # Start from a noisy concrete-like base
            base = np.full((img_size + 32, img_size + 32, 3), base_colour, dtype=np.uint8)
            noise = rng.integers(0, 40, base.shape, dtype=np.uint8)
            img = cv2.add(base, noise)

            # Class-specific texture overlays
            # All numpy int64 values cast to int — cv2 stubs require plain Python int
            if cls == "crack":
                for _ in range(int(rng.integers(3, 8))):
                    x1, y1, x2, y2 = (int(v) for v in rng.integers(0, img_size, 4))
                    thickness = int(rng.integers(1, 3))
                    cv2.line(img, (x1, y1), (x2, y2), (30, 30, 30), thickness)

            elif cls == "corrosion_stain":
                for _ in range(int(rng.integers(2, 5))):
                    cx, cy = (int(v) for v in rng.integers(20, img_size - 20, 2))
                    ax1, ax2 = int(rng.integers(10, 40)), int(rng.integers(10, 40))
                    cv2.ellipse(img, (cx, cy), (ax1, ax2), 0, 0, 360, (30, 60, 180), -1)

            elif cls == "spallation":
                for _ in range(int(rng.integers(3, 7))):
                    pts = rng.integers(0, img_size, (5, 1, 2)).astype(np.int32)
                    cv2.fillPoly(img, [pts], (100, 90, 80))

            augmented = augment(image=img)["image"]

            # Save as RGB PIL image
            pil_img = Image.fromarray(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
            pil_img.save(cls_dir / f"{cls}_{i:04d}.jpg", quality=90)

    total = n_per_class * len(CLASSES)
    print(f"[synthetic] Done — {total} images written to {RAW_DIR}/CODEBRIM/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip Kaggle; generate synthetic data instead",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=200,
        help="Images per class for synthetic mode (default: 200)",
    )
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_dataset(n_per_class=args.n_per_class)
        return

    if not check_kaggle_credentials():
        print(
            "[download] kaggle.json not found at ~/.kaggle/kaggle.json\n"
            "  To set up Kaggle credentials:\n"
            "    1. Go to https://www.kaggle.com/account\n"
            "    2. Scroll to 'API' → 'Create New API Token'\n"
            "    3. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json\n"
            "    4. Run: chmod 600 ~/.kaggle/kaggle.json  (on Git Bash)\n\n"
            "  Falling back to synthetic data generation...\n"
        )
        generate_synthetic_dataset(n_per_class=args.n_per_class)
        return

    success = download_from_kaggle()
    if not success:
        print("[download] Falling back to synthetic data generation...")
        generate_synthetic_dataset(n_per_class=args.n_per_class)


if __name__ == "__main__":
    main()