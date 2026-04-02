"""Streamlit dashboard for interactive anomaly detection.

Launch with: streamlit run src/demo/app.py
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from torch.utils.data import DataLoader

from src.config import load_config, resolve_device
from src.data.mvtec import MVTEC_CLASSES, MVTecTestDataset, MVTecTrainDataset, download_mvtec_class
from src.demo.viz import overlay_heatmap, tensor_to_image
from src.models.patchcore import PatchCore
from src.preprocessing.transforms import get_image_transform, get_mask_transform

BACKBONES = {"WideResNet50": "wide_resnet50_2"}

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("PatchCore Anomaly Detection")


# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    category = st.selectbox("MVTec Category", MVTEC_CLASSES, index=0)
    backbone_key = st.selectbox("Backbone", list(BACKBONES.keys()))
    backbone = BACKBONES[backbone_key]
    use_cache = st.checkbox("Use cached model", value=True)
    coreset_ratio = st.slider("Coreset fraction", 0.01, 1.0, 0.1, step=0.01)
    k_nn = st.slider("k-NN", 1, 10, 3)

    build_btn = st.button("Build Memory Bank", type="primary")
    run_btn = st.button("Run Test Set")


# --- Session state ---
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.threshold = 0.0
    st.session_state.scores = []
    st.session_state.labels = []
    st.session_state.images_processed = 0


# --- Build model ---
if build_btn:
    cfg = load_config()
    device = resolve_device(cfg.inference.device)

    with st.spinner("Building memory bank..."):
        download_mvtec_class(category, cfg.dataset.data_dir)
        transform = get_image_transform(
            image_size=cfg.model.image_size, resize=cfg.model.resize, backbone=backbone
        )
        train_ds = MVTecTrainDataset(cfg.dataset.data_dir, category, transform=transform)
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)

        model = PatchCore(
            backbone=backbone,
            coreset_ratio=coreset_ratio,
            k_nearest=k_nn,
            image_size=cfg.model.image_size,
            device=device,
        )

        cache_path = f"./cache/memory_banks/{category}_{backbone}_f{coreset_ratio:.3f}.npz"
        import os

        if use_cache and os.path.exists(cache_path):
            model.load(cache_path)
            st.info(f"Loaded from cache: {cache_path}")
        else:
            progress = st.progress(0)
            model.fit(train_dl)
            progress.progress(100)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            model.save(cache_path)

        threshold = model.calibrate_threshold(
            train_dl, percentile=cfg.inference.threshold_percentile
        )
        st.session_state.model = model
        st.session_state.threshold = threshold
        st.session_state.scores = []
        st.session_state.labels = []
        st.session_state.images_processed = 0

    st.success(
        f"Model ready | Memory bank: {model.memory_bank.shape[0]} patches | Threshold: {threshold:.4f}"
    )


# --- Run test set ---
if run_btn and st.session_state.model is not None:
    cfg = load_config()
    model = st.session_state.model
    threshold = st.session_state.threshold

    transform = get_image_transform(
        image_size=cfg.model.image_size, resize=cfg.model.resize, backbone=backbone
    )
    mask_transform = get_mask_transform(image_size=cfg.model.image_size, resize=cfg.model.resize)
    test_ds = MVTecTestDataset(
        cfg.dataset.data_dir,
        category,
        image_size=cfg.model.image_size,
        transform=transform,
        mask_transform=mask_transform,
    )
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    scores = []
    labels = []

    col_img, col_heat = st.columns(2)
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

    for i, (images, _masks, label) in enumerate(test_dl):
        score, segm_map = model.predict(images)
        scores.append(score.item())
        labels.append(label.item())

        # Display last image
        img_np = tensor_to_image(images.squeeze(0), backbone=backbone)
        heatmap = overlay_heatmap(img_np, segm_map.squeeze().numpy())

        with col_img:
            st.image(img_np, caption=f"Original (label={label.item()})", use_container_width=True)
        with col_heat:
            st.image(heatmap, caption=f"Score: {score.item():.4f}", use_container_width=True)

        # KPIs
        with metrics_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Images", i + 1)
            m2.metric("Mean Score", f"{np.mean(scores):.4f}")
            m3.metric("Anomalies", sum(1 for s in scores if s > threshold))
            m4.metric("Threshold", f"{threshold:.4f}")

    st.session_state.scores = scores
    st.session_state.labels = labels
    st.session_state.images_processed = len(scores)

    # Final charts
    if scores:
        from sklearn.metrics import roc_auc_score

        if len(set(labels)) > 1:
            auroc = roc_auc_score(labels, scores)
            st.metric("Image-level AUROC", f"{auroc:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=scores, mode="markers+lines", name="Score"))
            fig.add_hline(
                y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold"
            )
            fig.update_layout(title="Anomaly Scores", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(x=scores, nbins=30, title="Score Distribution")
            fig.add_vline(x=threshold, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

elif run_btn:
    st.warning("Build the memory bank first.")
