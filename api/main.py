"""
api/main.py
-----------
FastAPI inference API for the Pipeline Integrity Monitor.

Endpoints:
  POST /inspect       — Upload image, get defect class + GradCAM heatmap (base64)
  GET  /classes       — List of defect classes
  GET  /health        — Liveness + model loaded status
  GET  /model-info    — Architecture, epoch, val F1, parameter count

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Test:
    curl -X POST http://localhost:8000/inspect \
         -F "file=@data/processed/test/crack/crack_0001.jpg"
"""

from __future__ import annotations

import base64
import io
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from api.schemas import (
    ClassListResponse,
    HealthResponse,
    InspectionResponse,
    ModelInfoResponse,
)
from scripts.gradcam import GradCAM, get_target_layer, overlay_heatmap
from scripts.train import build_model

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/best_model.pth")
DB_PATH    = Path("logs/inference.db")
IMG_SIZE   = 224
MAX_FILE_MB = 10

CLASSES = [
    "background",
    "crack",
    "spallation",
    "exposed_bars",
    "corrosion_stain",
    "efflorescence",
]
NUM_CLASSES = len(CLASSES)

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── Global state ─────────────────────────────────────────────────────────────
_model:      Optional[torch.nn.Module] = None
_model_name: Optional[str]             = None
_model_meta: dict                      = {}
_device:     torch.device              = torch.device("cpu")
_cam:        Optional[GradCAM]         = None


# ─── DB ───────────────────────────────────────────────────────────────────────

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inference_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       REAL    NOT NULL,
            model_name      TEXT,
            predicted_class TEXT,
            confidence      REAL,
            duration_ms     REAL
        )
    """)
    conn.commit()
    conn.close()


def log_inference(
    model_name: Optional[str],
    predicted_class: str,
    confidence: float,
    duration_ms: float,
) -> int:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute(
        """INSERT INTO inference_log
           (timestamp, model_name, predicted_class, confidence, duration_ms)
           VALUES (?, ?, ?, ?, ?)""",
        (time.time(), model_name, predicted_class, confidence, duration_ms),
    )
    row_id = cur.lastrowid
    assert row_id is not None
    conn.commit()
    conn.close()
    return row_id


# ─── Startup / shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_name, _model_meta, _device, _cam

    init_db()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_PATH.exists():
        ckpt        = torch.load(str(MODEL_PATH), map_location=_device)
        model_name: str = ckpt.get("model_name") or "resnet18"
        _model_name = model_name
        _model_meta = {
            "epoch":  ckpt.get("epoch"),
            "val_f1": ckpt.get("val_f1"),
        }
        _model = build_model(_model_name or "resnet18", NUM_CLASSES, pretrained=False).to(_device)
        _model.load_state_dict(ckpt["model_state"])
        _model.eval()

        target_layer = get_target_layer(_model, _model_name)
        _cam = GradCAM(_model, target_layer)

        print(f"[api] Model loaded: {_model_name} | device: {_device}")
    else:
        print(f"[api] WARNING: {MODEL_PATH} not found. /inspect will fail until model is trained.")

    yield

    # Cleanup on shutdown
    if _cam:
        _cam.remove_hooks()


app = FastAPI(
    title="Pipeline Integrity Monitor API",
    description=(
        "Computer vision API for automated pipeline and infrastructure defect detection. "
        "Classifies 6 defect types from inspection imagery. "
        "GradCAM heatmaps show which image regions drove the prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Send JPEG or PNG.")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_name=_model_name,
    )


@app.get("/classes", response_model=ClassListResponse, tags=["info"])
def classes():
    return ClassListResponse(classes=CLASSES, num_classes=NUM_CLASSES)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["info"])
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    assert _model_name is not None
    n_params = sum(p.numel() for p in _model.parameters())
    return ModelInfoResponse(
        model_name=_model_name,
        trained_epoch=_model_meta.get("epoch"),
        val_f1=_model_meta.get("val_f1"),
        classes=CLASSES,
        num_parameters=n_params,
    )


@app.post("/inspect", response_model=InspectionResponse, tags=["inference"])
async def inspect(file: UploadFile = File(...)):
    """
    Upload a pipeline or infrastructure inspection image.
    Returns the defect class, confidence scores, and a GradCAM heatmap overlay (base64 PNG).
    """
    if _model is None or _cam is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first: python scripts/train.py"
        )
    assert _model_name is not None
    model_name = _model_name

    # File size guard
    contents = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_MB} MB"
        )

    t0 = time.time()

    original_pil = read_image(contents)
    resized = transforms.Resize((IMG_SIZE, IMG_SIZE))(original_pil)
    tensor = transforms.ToTensor()(resized)
    normalized = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)
    input_tensor = normalized.unsqueeze(0).to(_device)

    # Inference
    with torch.no_grad():
        logits = _model(input_tensor)
        probs  = F.softmax(logits, dim=1)[0]

    pred_idx   = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())
    pred_class = CLASSES[pred_idx]

    all_scores = {cls: round(float(probs[i].item()), 5) for i, cls in enumerate(CLASSES)}

    # GradCAM (re-run forward for gradient computation)
    heatmap = _cam.generate(input_tensor.clone().detach().requires_grad_(True), class_idx=pred_idx)
    overlay = overlay_heatmap(original_pil, heatmap, alpha=0.45)
    heatmap_b64 = pil_to_b64(overlay)

    duration_ms = (time.time() - t0) * 1000

    inference_id = log_inference(_model_name, pred_class, confidence, duration_ms)

    return InspectionResponse(
        predicted_class=pred_class,
        predicted_class_idx=pred_idx,
        confidence=round(confidence, 5),
        all_scores=all_scores,
        gradcam_heatmap_b64=heatmap_b64,
        model_name=_model_name,
        inference_id=inference_id,
    )