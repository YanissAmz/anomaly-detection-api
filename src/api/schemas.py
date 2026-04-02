"""Pydantic schemas for the anomaly detection API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    category: str | None = None


class BuildRequest(BaseModel):
    category: str = Field("bottle", description="MVTec AD class to train on")
    backbone: str = Field("wide_resnet50_2", description="Backbone model name")
    coreset_ratio: float = Field(0.1, ge=0.001, le=1.0, description="Coreset fraction")
    eps_coreset: float = Field(0.90, ge=0.01, le=1.0)
    k_nearest: int = Field(3, ge=1, le=20)
    use_cache: bool = Field(True, description="Load from cache if available")


class BuildResponse(BaseModel):
    status: str
    category: str
    backbone: str
    memory_bank_size: int
    threshold: float


class PredictionResponse(BaseModel):
    anomaly_score: float
    is_anomalous: bool
    threshold: float


class HeatmapResponse(BaseModel):
    anomaly_score: float
    is_anomalous: bool
    threshold: float
    heatmap_base64: str
