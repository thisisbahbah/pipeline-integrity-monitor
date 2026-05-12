"""
dashboard/app.py
----------------
Streamlit demo for the Pipeline Integrity Monitor.

Features:
  - Upload a pipeline/infrastructure inspection image
  - Calls the FastAPI /inspect endpoint
  - Displays: original image | GradCAM overlay | classification bar chart
  - Graceful fallback mode if API is unavailable (shows placeholder UI)
  - No .iloc[-1] on empty DataFrames (safe_last pattern applied)

Run:
    streamlit run dashboard/app.py

Environment:
    API_URL=http://localhost:8000   (default)
"""

from __future__ import annotations

import base64
import io
import os
from typing import Optional, Dict

import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

CLASSES = [
    "background",
    "crack",
    "spallation",
    "exposed_bars",
    "corrosion_stain",
    "efflorescence",
]

CLASS_COLOURS = {
    "background":      "#94a3b8",
    "crack":           "#ef4444",
    "spallation":      "#f97316",
    "exposed_bars":    "#eab308",
    "corrosion_stain": "#dc2626",
    "efflorescence":   "#3b82f6",
}

DEFECT_DESCRIPTIONS = {
    "background":      "No defect detected. Background / healthy surface.",
    "crack":           "Surface or structural crack — risk of water ingress and fatigue propagation.",
    "spallation":      "Concrete spalling — loss of surface layer exposing aggregate.",
    "exposed_bars":    "Reinforcement bars exposed — severe corrosion risk, structural concern.",
    "corrosion_stain": "Corrosion staining — iron oxide leaching, indicates rebar corrosion onset.",
    "efflorescence":   "Efflorescence — salt crystallisation from water permeation.",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_last(lst: list, default=None):
    """Return the last element of a list, or default if empty."""
    return lst[-1] if lst else default


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def call_inspect(image_bytes: bytes) -> Optional[dict]:
    try:
        r = requests.post(
            f"{API_URL}/inspect",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API error {r.status_code}: {r.text}")
            return None
    except requests.ConnectionError:
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def make_confidence_chart(all_scores: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of all class confidence scores."""
    classes = list(all_scores.keys())
    scores  = [all_scores[c] * 100 for c in classes]
    colours = [CLASS_COLOURS.get(c, "#64748b") for c in classes]

    fig = go.Figure(go.Bar(
        y=classes,
        x=scores,
        orientation="h",
        marker_color=colours,
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=None,
        xaxis=dict(title="Confidence (%)", range=[0, 110]),
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=10, r=30, t=10, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


# ─── Fallback UI ──────────────────────────────────────────────────────────────

def render_fallback():
    st.warning(
        "⚠️ API is not reachable. Start the FastAPI server first:\n\n"
        "```bash\nuvicorn api.main:app --host 0.0.0.0 --port 8000\n```\n\n"
        "The dashboard will refresh automatically once the API is live."
    )
    st.markdown("---")
    st.subheader("Defect classes this model detects:")
    for cls, desc in DEFECT_DESCRIPTIONS.items():
        st.markdown(f"**{cls.replace('_', ' ').title()}** — {desc}")


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Pipeline Integrity Monitor",
        page_icon="🔬",
        layout="wide",
    )

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style='margin-bottom:0'>🔬 Pipeline Integrity Monitor</h1>
        <p style='color:#6b7280; margin-top:4px'>
        Computer vision defect detection for infrastructure inspection imagery.
        Powered by fine-tuned ResNet-18 / EfficientNet-B0 with GradCAM explainability.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "Upload a pipeline or concrete inspection image. "
            "The model classifies the defect type and generates a **GradCAM** heatmap "
            "showing which image regions drove the prediction."
        )
        st.markdown("---")
        st.markdown("**Defect classes:**")
        for cls, desc in DEFECT_DESCRIPTIONS.items():
            with st.expander(cls.replace("_", " ").title()):
                st.caption(desc)

        st.markdown("---")
        api_ok = check_api_health()
        if api_ok:
            st.success("API: online ✓")
        else:
            st.error("API: offline ✗")

        if st.button("↺ Refresh"):
            st.rerun()

    # ── API check ───────────────────────────────────────────────────────────
    if not check_api_health():
        render_fallback()
        return

    # ── Upload ──────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload inspection image (JPEG or PNG, max 10 MB)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    if uploaded is None:
        st.info("Upload an inspection image above to begin analysis.")
        return

    img_bytes = uploaded.read()

    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    with col1:
        st.markdown("**Original image**")
        original_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(original_pil, use_column_width=True)

    with st.spinner("Running defect classification and GradCAM..."):
        result = call_inspect(img_bytes)

    if result is None:
        st.error("API call failed. Check that uvicorn is running.")
        return

    pred_class  = result["predicted_class"]
    confidence  = result["confidence"]
    all_scores  = result["all_scores"]
    heatmap_b64 = result.get("gradcam_heatmap_b64")

    # ── GradCAM overlay ─────────────────────────────────────────────────────
    with col2:
        st.markdown("**GradCAM attention map**")
        if heatmap_b64:
            overlay_pil = b64_to_pil(heatmap_b64)
            st.image(overlay_pil, use_column_width=True)
            st.caption(
                "Warm colours (red/yellow) show which regions most influenced the prediction. "
                "Cool colours (blue) are low-influence zones."
            )
        else:
            st.info("GradCAM not available for this prediction.")

    # ── Result panel ────────────────────────────────────────────────────────
    with col3:
        st.markdown("**Classification result**")

        colour = CLASS_COLOURS.get(pred_class, "#64748b")
        bg     = "#fef2f2" if pred_class != "background" else "#f0fdf4"

        st.markdown(
            f"""
            <div style="
                background:{bg};
                border-left: 4px solid {colour};
                border-radius: 6px;
                padding: 12px 16px;
                margin-bottom: 12px;
            ">
                <div style="font-size:13px; color:#6b7280;">Predicted defect</div>
                <div style="font-size:22px; font-weight:600; color:{colour};">
                    {pred_class.replace('_', ' ').title()}
                </div>
                <div style="font-size:13px; color:#374151; margin-top:4px;">
                    {DEFECT_DESCRIPTIONS.get(pred_class, '')}
                </div>
            </div>
            <div style="
                font-size:13px; color:#6b7280; margin-bottom:4px;
            ">Confidence</div>
            <div style="
                font-size:28px; font-weight:700; color:{colour};
            ">{confidence*100:.1f}%</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**All class scores**")
        fig = make_confidence_chart(all_scores)
        st.plotly_chart(fig, use_container_width=True)

    # ── Inference metadata ──────────────────────────────────────────────────
    st.divider()
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Model", result.get("model_name", "—"))
    meta_col2.metric("Confidence", f"{confidence*100:.1f}%")
    meta_col3.metric("Inference ID", result.get("inference_id", "—"))

    # ── Raw JSON (collapsible) ───────────────────────────────────────────────
    with st.expander("Raw API response"):
        display = {k: v for k, v in result.items() if k != "gradcam_heatmap_b64"}
        st.json(display)


if __name__ == "__main__":
    main()