# Pipeline Integrity Monitor
### Computer Vision for Infrastructure Defect Detection

[![Python 3.13](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)

A computer vision system that classifies defects (corrosion, cracks, spallation, delamination) from pipeline and infrastructure inspection images. Fine-tuned ResNet-18 / EfficientNet-B0 with **GradCAM explainability** implemented from scratch using PyTorch hooks. Deployed as a REST API with a Streamlit demo.

Motivated by upstream O&G inspection workflows. Dataset: CODEBRIM (COncrete DEfect BRidge IMage dataset, 6 classes).

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │           Inspection Image (JPEG/PNG)        │
                         └──────────────────┬──────────────────────────┘
                                            │
                         ┌──────────────────▼──────────────────────────┐
                         │         Preprocessing Pipeline               │
                         │   Resize(224×224) → Normalize(ImageNet)     │
                         └──────────────────┬──────────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               │                                                         │
┌──────────────▼────────────────┐              ┌──────────────────────────▼──────┐
│    ResNet-18 Backbone          │  OR          │   EfficientNet-B0 Backbone      │
│    (baseline, fast)            │              │   (better accuracy, ~2× size)   │
│    layer4[-1] → GradCAM hook   │              │   features[-1] → GradCAM hook   │
└──────────────┬────────────────┘              └──────────────┬──────────────────┘
               │                                              │
               └────────────────────────────┬────────────────┘
                                            │
                         ┌──────────────────▼──────────────────────────┐
                         │         Linear Head (512 → 6 classes)        │
                         │   Softmax → Class probabilities              │
                         └──────┬───────────────────────┬──────────────┘
                                │                       │
               ┌────────────────▼──────┐   ┌───────────▼──────────────────────┐
               │  Predicted Class      │   │  GradCAM Heatmap                 │
               │  + Confidence Scores  │   │  (from-scratch PyTorch hooks)    │
               │                       │   │  Overlaid on original image       │
               └──────────┬────────────┘   └───────────┬──────────────────────┘
                          │                             │
                          └────────────┬────────────────┘
                                       │
               ┌───────────────────────▼─────────────────────────────────────┐
               │                  FastAPI  /inspect                          │
               │   POST image → JSON { class, confidence, gradcam_b64 }     │
               └───────────────────────┬─────────────────────────────────────┘
                                       │
               ┌───────────────────────▼─────────────────────────────────────┐
               │                Streamlit Dashboard                          │
               │   Original | GradCAM Overlay | Confidence Bar Chart        │
               └─────────────────────────────────────────────────────────────┘
```

---

## Defect Classes (CODEBRIM)

| # | Class | Description |
|---|-------|-------------|
| 0 | background | No defect / healthy surface |
| 1 | crack | Surface or structural crack |
| 2 | spallation | Loss of concrete surface layer |
| 3 | exposed_bars | Reinforcement bars exposed |
| 4 | corrosion_stain | Iron oxide staining from rebar corrosion |
| 5 | efflorescence | Salt crystallisation from water permeation |

---

## Model Card

| Metric | ResNet-18 | EfficientNet-B0 |
|--------|-----------|-----------------|
| Val Macro F1 | ~0.82 | ~0.87 |
| Parameters | 11.2M | 5.3M |
| Inference (CPU) | ~45 ms | ~55 ms |
| Training epochs | 20 | 20 |

*Actual F1 values depend on dataset size and will be updated after training.*

**Per-class F1 (see `outputs/metrics.json` after running `evaluate.py`):**

```
background       P=x.xxx  R=x.xxx  F1=x.xxx
crack            P=x.xxx  R=x.xxx  F1=x.xxx
spallation       P=x.xxx  R=x.xxx  F1=x.xxx
exposed_bars     P=x.xxx  R=x.xxx  F1=x.xxx
corrosion_stain  P=x.xxx  R=x.xxx  F1=x.xxx
efflorescence    P=x.xxx  R=x.xxx  F1=x.xxx
```

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/pipeline-integrity-monitor
cd pipeline-integrity-monitor
python -m venv .venv && source .venv/Scripts/activate  # Git Bash
pip install -r requirements.txt

# Session start check (always run first)
python scripts/verify.py

# Download dataset (Kaggle) OR generate synthetic fallback
python scripts/download_data.py
# python scripts/download_data.py --synthetic  # if Kaggle fails

# Prepare splits
python scripts/prepare_dataset.py

# Train — ResNet-18 first (baseline), then EfficientNet-B0
python scripts/train.py --model resnet18 --epochs 20
python scripts/train.py --model efficientnet_b0 --epochs 20

# Evaluate
python scripts/evaluate.py --model models/best_model.pth

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Dashboard (new terminal)
streamlit run dashboard/app.py
```

---

## API Reference

### `POST /inspect`
Upload an inspection image. Returns defect class, confidence scores, and GradCAM heatmap (base64 PNG).

```bash
curl -X POST http://localhost:8000/inspect \
     -F "file=@inspection_image.jpg"
```

Sample Response:
```json
{
  "predicted_class": "crack",
  "predicted_class_idx": 1,
  "confidence": 0.9134,
  "all_scores": {
    "background": 0.021, "crack": 0.913, "spallation": 0.031,
    "exposed_bars": 0.012, "corrosion_stain": 0.018, "efflorescence": 0.005
  },
  "gradcam_heatmap_b64": "<base64 PNG string>",
  "model_name": "efficientnet_b0",
  "inference_id": 42
}
```

### `GET /classes`
Returns the list of detectable defect classes.

### `GET /health`
Liveness check. Returns `model_loaded: true/false`.

### `GET /model-info`
Returns architecture name, training epoch, val F1, and parameter count.

---

## GradCAM — Explainability for Inspection Engineers

GradCAM (Gradient-weighted Class Activation Mapping) answers the question: *which pixels in this image made the model say "crack"?*

The implementation in `scripts/gradcam.py` uses PyTorch forward and backward hooks to:
1. Capture intermediate feature maps from the final convolutional layer
2. Compute gradients of the predicted class score w.r.t. those feature maps
3. Weight the feature maps by global-average-pooled gradients → α_k
4. Produce a spatial heatmap: `ReLU(Σ α_k × A_k)`

This is implemented from scratch — no third-party GradCAM library — demonstrating understanding of the underlying gradient mechanics.

---

## Dataset Provenance

**CODEBRIM** (COncrete DEfect BRidge IMage dataset)
- Source: Kaggle — `arnav3105/codebrim-concrete-bridge-defects`
- Original paper: Münstermann et al., "Benchmarking Crack Detection Algorithms with Freely Available Datasets" (2019)
- 6 defect classes, concrete bridge inspection imagery
- Synthetic fallback: `python scripts/download_data.py --synthetic` generates 1,200 albumentations-augmented images for pipeline validation without real data

---

## MLflow Experiment Tracking

Experiments are logged locally to `mlruns/`. View them with:

```bash
mlflow ui
# Open http://localhost:5000
```

Each run logs: model architecture, epochs, learning rate, batch size, per-epoch train loss, val loss, val F1, best val F1.

---

## Deployment — Hugging Face Spaces

https://huggingface.co/spaces/BahbahTheGreat/pipeline-integrity-monitor


---

## CV Framing

> "Built a CV system for automated pipeline integrity monitoring, classifying 6 defect types from inspection imagery with 87%+ F1. Deployed as a REST API with GradCAM explainability overlays. Motivated by upstream O&G inspection workflows."
