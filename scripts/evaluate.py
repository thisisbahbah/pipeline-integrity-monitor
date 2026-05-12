"""
evaluate.py
-----------
Evaluates the saved best model on the test set.

Outputs (all written to outputs/):
  - metrics.json             ← per-class precision/recall/F1 + macro avg
  - confusion_matrix.png     ← seaborn heatmap
  - gradcam_samples/         ← 2 GradCAM overlay images per class

Usage:
    python scripts/evaluate.py --model models/best_model.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torchvision import transforms

from scripts.gradcam import GradCAM, get_target_layer, overlay_heatmap
from scripts.train import DefectDataset, build_model

CLASSES = [
    "background",
    "crack",
    "spallation",
    "exposed_bars",
    "corrosion_stain",
    "efflorescence",
]
NUM_CLASSES = len(CLASSES)
OUTPUT_DIR  = Path("outputs")
CAM_DIR     = OUTPUT_DIR / "gradcam_samples"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DENORM = transforms.Normalize(
    mean=[-m / s for m, s in zip(MEAN, STD)],
    std=[1 / s for s in STD],
)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalised (3, H, W) tensor to a PIL RGB image."""
    denorm = DENORM(tensor.cpu().clone())
    denorm = denorm.clamp(0, 1)
    arr = (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def safe_gradcam_samples(
    model: torch.nn.Module,
    model_name: str,
    test_ds: DefectDataset,
    device: torch.device,
    n_per_class: int = 2,
) -> None:
    """Save GradCAM overlay images for n_per_class images per class."""
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    target_layer = get_target_layer(model, model_name)
    cam = GradCAM(model, target_layer)

    # Group sample indices by class
    class_samples: dict[int, list[int]] = {i: [] for i in range(NUM_CLASSES)}
    for idx, (_, label) in enumerate(test_ds.samples):
        class_samples[label].append(idx)

    for cls_idx, cls_name in enumerate(CLASSES):
        indices = class_samples[cls_idx][:n_per_class]
        if not indices:
            print(f"  [gradcam] No test samples for class '{cls_name}', skipping")
            continue

        for i, sample_idx in enumerate(indices):
            img_tensor, label = test_ds[sample_idx]
            input_t = img_tensor.unsqueeze(0).to(device)

            heatmap = cam.generate(input_t, class_idx=cls_idx)
            original_pil = tensor_to_pil(img_tensor)
            overlay = overlay_heatmap(original_pil, heatmap, alpha=0.45)

            # Side-by-side figure
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(original_pil)
            axes[0].set_title("Original", fontsize=10)
            axes[0].axis("off")

            axes[1].imshow(overlay)
            axes[1].set_title(f"GradCAM — {cls_name}", fontsize=10)
            axes[1].axis("off")

            fig.tight_layout()
            out_path = CAM_DIR / f"{cls_name}_{i+1}.png"
            fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  [gradcam] Saved {out_path}")

    cam.remove_hooks()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="models/best_model.pth")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch",    type=int, default=32)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[evaluate] Model not found: {model_path}")
        print("  Run: python scripts/train.py --model resnet18 --epochs 20  first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Device: {device}")

    # Load checkpoint
    ckpt       = torch.load(str(model_path), map_location=device)
    model_name = ckpt.get("model_name", "resnet18")
    model      = build_model(model_name, NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[evaluate] Loaded {model_name} from epoch {ckpt.get('epoch', '?')}")

    # Test dataset
    test_ds = DefectDataset("test", img_size=args.img_size, augment=False)
    loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=0
    )
    print(f"[evaluate] Test set: {len(test_ds)} images\n")

    # ── Inference ──────────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # ── Metrics ────────────────────────────────────────────────────────────────
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Accuracy:  {float(report['accuracy']):.4f}\n")  # type: ignore[arg-type]
    print("Per-class:")
    for cls in CLASSES:
        r: dict = report[cls]  # type: ignore[index]
        print(f"  {cls:<18} P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  n={int(r['support'])}")

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[evaluate] Metrics saved: {metrics_path}")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(str(cm_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved: {cm_path}")

    # ── GradCAM samples ────────────────────────────────────────────────────────
    print("\n[evaluate] Generating GradCAM visualisations...")
    safe_gradcam_samples(model, model_name, test_ds, device, n_per_class=2)

    print("\n[evaluate] Done. All outputs in outputs/")


if __name__ == "__main__":
    main()