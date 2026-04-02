"""Configuration loading for anomaly detection pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml


@dataclass
class ModelConfig:
    backbone: str = "wide_resnet50_2"
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    coreset_ratio: float = 0.1
    eps_coreset: float = 0.90
    k_nearest: int = 3
    image_size: int = 224
    resize: int = 256
    backbone_image_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "wide_resnet50_2": 224,
            "RN50": 224,
            "RN50x4": 288,
            "RN50x16": 384,
            "RN101": 224,
        }
    )


@dataclass
class DatasetConfig:
    name: str = "mvtec"
    category: str = "bottle"
    data_dir: str = "./data/mvtec"
    download: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 1
    num_workers: int = 4


@dataclass
class InferenceConfig:
    threshold_percentile: int = 99
    device: str = "auto"


@dataclass
class CacheConfig:
    dir: str = "./cache/memory_banks"
    enabled: bool = True


@dataclass
class ServeConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class DemoConfig:
    server_port: int = 7860


@dataclass
class Settings:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)


def load_config(path: str | Path = "configs/default.yaml") -> Settings:
    """Load configuration from YAML file with environment variable overrides."""
    path = Path(path)
    if not path.exists():
        return Settings()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    settings = Settings(
        model=ModelConfig(**raw.get("model", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        training=TrainingConfig(**raw.get("training", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
        cache=CacheConfig(**raw.get("cache", {})),
        serve=ServeConfig(**raw.get("serve", {})),
        demo=DemoConfig(**raw.get("demo", {})),
    )

    # Environment variable overrides
    if env_device := os.getenv("DEVICE"):
        settings.inference.device = env_device
    if env_category := os.getenv("MVTEC_CATEGORY"):
        settings.dataset.category = env_category
    if env_cache_dir := os.getenv("CACHE_DIR"):
        settings.cache.dir = env_cache_dir

    return settings


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_image_size(cfg: Settings) -> int:
    """Get the appropriate image size for the configured backbone."""
    return cfg.model.backbone_image_sizes.get(cfg.model.backbone, cfg.model.image_size)
