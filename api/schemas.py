"""
schemas.py
----------
Pydantic v2 request and response models for the Pipeline Integrity Monitor API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class InspectionResponse(BaseModel):
    """Response from POST /inspect"""

    predicted_class: str = Field(
        description="Name of the predicted defect class"
    )
    predicted_class_idx: int = Field(
        description="Integer index of the predicted class"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Softmax confidence for the predicted class"
    )
    all_scores: Dict[str, float] = Field(
        description="Softmax score for every class, keyed by class name"
    )
    gradcam_heatmap_b64: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded PNG of the GradCAM overlay on the input image. "
            "Decode with: Image.open(BytesIO(base64.b64decode(v)))"
        )
    )
    model_name: str = Field(description="Architecture used for inference")
    inference_id: int = Field(description="SQLite row ID of the logged inference")


class ClassListResponse(BaseModel):
    """Response from GET /classes"""
    classes: List[str]
    num_classes: int


class HealthResponse(BaseModel):
    """Response from GET /health"""
    status: str
    model_loaded: bool
    model_name: Optional[str]


class ModelInfoResponse(BaseModel):
    """Response from GET /model-info"""
    model_name: str
    trained_epoch: Optional[int]
    val_f1: Optional[float]
    classes: List[str]
    num_parameters: int