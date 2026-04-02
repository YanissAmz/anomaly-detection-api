"""FastAPI application for anomaly detection inference."""

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionResponse(BaseModel):
    anomaly_score: float
    is_anomalous: bool
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelState:
    model = None
    loaded = False
    threshold = 0.5


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading PatchCore model...")
    # Model loading will be implemented when training is complete
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
        status="ok" if state.loaded else "model_not_loaded", model_loaded=state.loaded
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):  # noqa: B008
    """Upload an image and get anomaly detection results."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    try:
        _image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as err:
        raise HTTPException(status_code=400, detail="Invalid image file") from err

    # Preprocessing and inference will be implemented
    score = 0.0  # placeholder
    return PredictionResponse(
        anomaly_score=score,
        is_anomalous=score > state.threshold,
        threshold=state.threshold,
    )
