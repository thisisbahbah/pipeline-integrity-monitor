"""
train.py
--------
Fine-tunes a pretrained CNN on the CODEBRIM defect dataset.
Phased approach:  ResNet-18 (fast baseline)  →  EfficientNet-B0 (better accuracy)

Features:
  - Checkpoint saved every 5 epochs (resumable sessions)
  - Best model saved by val F1 (not val loss)
  - MLflow experiment tracking (metrics, params, model artifact)
  - Class-weighted loss for imbalanced splits

Usage:
    python scripts/train.py --model resnet18 --epochs 20
    python scripts/train.py --model efficientnet_b0 --epochs 20
    python scripts/train.py --model resnet18 --resume models/checkpoints/resnet18_epoch_10.pth
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, Dict, cast

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
CKPT_DIR  = MODEL_DIR / "checkpoints"
BEST_PATH = MODEL_DIR / "best_model.pth"

CLASSES = [
    "background",
    "crack",
    "spallation",
    "exposed_bars",
    "corrosion_stain",
    "efflorescence",
]
NUM_CLASSES = len(CLASSES)

# ─── Dataset ──────────────────────────────────────────────────────────────────

class DefectDataset(Dataset):
    """Loads images from data/processed/<split>/<class>/ directories."""

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, split: str, img_size: int = 224, augment: bool = False):
        self.split   = split
        self.samples = []  # list of (path, label_idx)

        base = PROC_DIR / split
        if not base.exists():
            raise FileNotFoundError(
                f"Processed split not found: {base}\n"
                "Run: python scripts/prepare_dataset.py"
            )

        for label_idx, cls in enumerate(CLASSES):
            cls_dir = base / cls
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.rglob("*.jpg")):
                self.samples.append((str(img_path), label_idx))

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return cast(torch.Tensor, self.transform(img)), label


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build and return a fine-tuned model.
    Only the final classification head is replaced; backbone is unfrozen
    after 3 warm-up epochs for gradual fine-tuning.
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        classifier = cast(nn.Linear, model.classifier[1])
        in_features = classifier.in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose resnet18 or efficientnet_b0.")

    return model


def get_class_weights(dataset: DefectDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    counts = torch.zeros(NUM_CLASSES)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = counts.sum() / (NUM_CLASSES * counts.clamp(min=1))
    return weights


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in tqdm(loader, desc="  eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    accuracy  = float(sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels))

    return {
        "loss":     float(total_loss / len(all_labels)),
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs",   type=int, default=20)
    parser.add_argument("--batch",    type=int, default=32)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resume",   type=str, default=None, help="Path to checkpoint .pth")
    parser.add_argument("--workers",  type=int, default=0, help="DataLoader workers (0 = main process)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device} | Model: {args.model} | Epochs: {args.epochs}")

    # Data
    train_ds = DefectDataset("train", img_size=args.img_size, augment=True)
    val_ds   = DefectDataset("val",   img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))

    print(f"[train] Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # Model
    model = build_model(args.model, NUM_CLASSES, pretrained=True).to(device)

    # Class-weighted loss
    weights   = get_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch   = 0
    best_val_f1   = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        print(f"[train] Resumed from epoch {ckpt['epoch']} (best val F1: {best_val_f1:.4f})")

    # MLflow experiment
    mlflow.set_experiment("pipeline-integrity-monitor")

    with mlflow.start_run(run_name=f"{args.model}_{int(time.time())}"):
        mlflow.log_params({
            "model":    args.model,
            "epochs":   args.epochs,
            "batch":    args.batch,
            "lr":       args.lr,
            "img_size": args.img_size,
            "device":   str(device),
        })

        print("\n[train] Starting training...\n")

        for epoch in range(start_epoch, args.epochs):
            # Warm-up: freeze backbone for first 3 epochs, then unfreeze
            if epoch == 0:
                print("[train] Warm-up: freezing backbone (epochs 0-2)")
                if args.model == "resnet18":
                    for name, p in model.named_parameters():
                        if "fc" not in name:
                            p.requires_grad = False
                elif args.model == "efficientnet_b0":
                    for name, p in model.named_parameters():
                        if "classifier" not in name:
                            p.requires_grad = False
            elif epoch == 3:
                print("[train] Epoch 3: unfreezing backbone (full fine-tune)")
                for p in model.parameters():
                    p.requires_grad = True

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics           = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            lr_now = scheduler.get_last_lr()[0]

            print(
                f"  Epoch {epoch+1:3d}/{args.epochs}"
                f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                f"  val_loss={val_metrics['loss']:.4f}"
                f"  val_f1={val_metrics['macro_f1']:.4f}"
                f"  val_acc={val_metrics['accuracy']:.4f}"
                f"  lr={lr_now:.2e}"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_metrics["loss"],
                "val_f1":     val_metrics["macro_f1"],
                "val_acc":    val_metrics["accuracy"],
                "lr":         lr_now,
            }, step=epoch)

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                ckpt_path = CKPT_DIR / f"{args.model}_epoch_{epoch+1:03d}.pth"
                torch.save({
                    "epoch":           epoch,
                    "model_name":      args.model,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_f1":          val_metrics["macro_f1"],
                    "best_val_f1":     best_val_f1,
                    "classes":         CLASSES,
                }, str(ckpt_path))
                print(f"  [ckpt] Saved: {ckpt_path}")

            # Save best model
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                torch.save({
                    "epoch":       epoch,
                    "model_name":  args.model,
                    "model_state": model.state_dict(),
                    "val_f1":      best_val_f1,
                    "classes":     CLASSES,
                }, str(BEST_PATH))
                print(f"  [best] New best val F1: {best_val_f1:.4f} → {BEST_PATH}")

        mlflow.log_metric("best_val_f1", best_val_f1)
        mlflow.log_artifact(str(BEST_PATH), artifact_path="model")

        print(f"\n[train] Training complete. Best val F1: {best_val_f1:.4f}")
        print(f"[train] Best model saved to: {BEST_PATH}")


if __name__ == "__main__":
    main()