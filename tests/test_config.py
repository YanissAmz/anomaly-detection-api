import tempfile
from pathlib import Path

import torch

from src.config import Settings, get_image_size, load_config, resolve_device


class TestLoadConfig:
    def test_default_config(self):
        cfg = load_config("configs/default.yaml")
        assert cfg.model.backbone == "wide_resnet50_2"
        assert cfg.model.coreset_ratio == 0.1
        assert cfg.model.k_nearest == 3
        assert cfg.inference.threshold_percentile == 99
        assert cfg.inference.device == "auto"

    def test_missing_file_returns_defaults(self):
        cfg = load_config("nonexistent.yaml")
        assert isinstance(cfg, Settings)
        assert cfg.model.backbone == "wide_resnet50_2"

    def test_partial_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  backbone: resnet18\n")
            f.flush()
            cfg = load_config(f.name)
        assert cfg.model.backbone == "resnet18"
        # Other fields should still have defaults
        assert cfg.dataset.category == "bottle"
        Path(f.name).unlink()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DEVICE", "cpu")
        cfg = load_config("configs/default.yaml")
        assert cfg.inference.device == "cpu"


class TestResolveDevice:
    def test_cpu(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_auto_returns_device(self):
        device = resolve_device("auto")
        assert isinstance(device, torch.device)


class TestGetImageSize:
    def test_known_backbone(self):
        cfg = load_config("configs/default.yaml")
        cfg.model.backbone = "RN50x4"
        assert get_image_size(cfg) == 288

    def test_default_backbone(self):
        cfg = load_config("configs/default.yaml")
        assert get_image_size(cfg) == 224
