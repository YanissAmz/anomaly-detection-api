"""FastAPI application for anomaly detection inference."""

from __future__ import annotations

import base64
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import (
    BuildRequest,
    BuildResponse,
    HealthResponse,
    HeatmapResponse,
    PredictionResponse,
)
from src.config import load_config, resolve_device
from src.data.mvtec import MVTecTrainDataset, download_mvtec_class
from src.models.patchcore import PatchCore
from src.preprocessing.transforms import get_image_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelState:
    model: PatchCore | None = None
    loaded: bool = False
    threshold: float = 0.5
    category: str | None = None
    transform = None
    cfg = None


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.cfg = load_config()
    logger.info("Anomaly Detection API ready. Use POST /build to load a model.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Anomaly Detection API",
    description="Visual anomaly detection using PatchCore on MVTec AD",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if state.loaded else "model_not_loaded",
        model_loaded=state.loaded,
        category=state.category,
    )


@app.post("/build", response_model=BuildResponse)
def build(req: BuildRequest):
    """Train PatchCore model on a MVTec AD category."""
    cfg = state.cfg or load_config()
    device = resolve_device(cfg.inference.device)

    # Update config from request
    cfg.dataset.category = req.category
    cfg.model.backbone = req.backbone
    cfg.model.coreset_ratio = req.coreset_ratio
    cfg.model.eps_coreset = req.eps_coreset
    cfg.model.k_nearest = req.k_nearest

    # Check cache
    cache_path = Path(cfg.cache.dir) / f"{req.category}_{req.backbone}_f{req.coreset_ratio:.3f}.npz"

    model = PatchCore(
        backbone=req.backbone,
        coreset_ratio=req.coreset_ratio,
        eps_coreset=req.eps_coreset,
        k_nearest=req.k_nearest,
        image_size=cfg.model.image_size,
        device=device,
    )

    if req.use_cache and cache_path.exists():
        logger.info("Loading model from cache: %s", cache_path)
        model.load(cache_path)
    else:
        logger.info("Training PatchCore on '%s'...", req.category)
        download_mvtec_class(req.category, cfg.dataset.data_dir)
        transform = get_image_transform(
            image_size=cfg.model.image_size,
            resize=cfg.model.resize,
            backbone=req.backbone,
        )
        train_ds = MVTecTrainDataset(cfg.dataset.data_dir, req.category, transform=transform)
        from torch.utils.data import DataLoader

        train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
        model.fit(train_dl)

        if cfg.cache.enabled:
            model.save(cache_path)

    # Calibrate threshold
    transform = get_image_transform(
        image_size=cfg.model.image_size,
        resize=cfg.model.resize,
        backbone=req.backbone,
    )
    train_ds = MVTecTrainDataset(cfg.dataset.data_dir, req.category, transform=transform)
    from torch.utils.data import DataLoader

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
    threshold = model.calibrate_threshold(train_dl, percentile=cfg.inference.threshold_percentile)

    state.model = model
    state.loaded = True
    state.threshold = threshold
    state.category = req.category
    state.transform = transform

    return BuildResponse(
        status="ok",
        category=req.category,
        backbone=req.backbone,
        memory_bank_size=model.memory_bank.shape[0],
        threshold=round(threshold, 4),
    )


def _read_and_transform_image(contents: bytes) -> torch.Tensor:
    """Read image bytes, apply transform, return tensor with batch dim."""
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    if state.transform is None:
        cfg = state.cfg or load_config()
        state.transform = get_image_transform(
            image_size=cfg.model.image_size,
            resize=cfg.model.resize,
            backbone=cfg.model.backbone,
        )
    tensor = state.transform(image)
    return tensor.unsqueeze(0)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),  # noqa: B008
    threshold: float | None = None,
):
    """Upload an image and get anomaly detection results."""
    if not state.loaded or state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call POST /build first.")

    contents = await file.read()
    try:
        tensor = _read_and_transform_image(contents)
    except Exception as err:
        raise HTTPException(status_code=400, detail="Invalid image file") from err

    score, _segm_map = state.model.predict(tensor)
    thresh = threshold if threshold is not None else state.threshold
    return PredictionResponse(
        anomaly_score=round(score.item(), 4),
        is_anomalous=score.item() > thresh,
        threshold=round(thresh, 4),
    )


@app.post("/predict/heatmap", response_model=HeatmapResponse)
async def predict_heatmap(
    file: UploadFile = File(...),  # noqa: B008
    threshold: float | None = None,
):
    """Upload an image and get anomaly score + heatmap overlay."""
    if not state.loaded or state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call POST /build first.")

    contents = await file.read()
    try:
        tensor = _read_and_transform_image(contents)
    except Exception as err:
        raise HTTPException(status_code=400, detail="Invalid image file") from err

    score, segm_map = state.model.predict(tensor)
    thresh = threshold if threshold is not None else state.threshold

    # Generate heatmap overlay
    from src.demo.viz import overlay_heatmap
    from src.preprocessing.transforms import denormalize

    img_tensor = denormalize(tensor.squeeze(0).clone())
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    heatmap_np = overlay_heatmap(img_np, segm_map.squeeze().numpy())

    # Encode as base64 PNG
    heatmap_img = Image.fromarray(heatmap_np)
    buffer = io.BytesIO()
    heatmap_img.save(buffer, format="PNG")
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode()

    return HeatmapResponse(
        anomaly_score=round(score.item(), 4),
        is_anomalous=score.item() > thresh,
        threshold=round(thresh, 4),
        heatmap_base64=heatmap_b64,
    )
