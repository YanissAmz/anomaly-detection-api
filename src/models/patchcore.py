"""PatchCore anomaly detection model.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022).

Ported from the PFE implementation with the following fixes:
- AvgPool2d and AdaptiveAvgPool2d moved out of the fit() loop
- Proper device management (no hardcoded CPU)
- Weighted K-NN scoring from the original paper
- Configurable via Settings dataclass
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageFilter
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms

from src.models.coreset import get_coreset

logger = logging.getLogger(__name__)


def _gaussian_blur(segm_map: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """Apply Gaussian blur to a segmentation map for smoother heatmaps."""
    max_value = segm_map.max()
    if max_value == 0:
        return segm_map
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    blur_kernel = ImageFilter.GaussianBlur(radius=radius)
    blurred = to_pil(segm_map[0] / max_value).filter(blur_kernel)
    return to_tensor(blurred) * max_value


class PatchCore(nn.Module):
    """PatchCore anomaly detection using pre-trained feature extraction,
    coreset subsampling, and weighted K-NN scoring.

    Args:
        backbone: backbone model name (wide_resnet50_2, resnet18)
        coreset_ratio: fraction of training patches to keep (0.0-1.0)
        eps_coreset: epsilon for SparseRandomProjection
        k_nearest: k for K-NN scoring
        image_size: input image size (for segmentation map interpolation)
        device: torch device
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        coreset_ratio: float = 0.1,
        eps_coreset: float = 0.90,
        k_nearest: int = 3,
        image_size: int = 224,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.backbone_name = backbone
        self.coreset_ratio = coreset_ratio
        self.eps_coreset = eps_coreset
        self.k_nearest = k_nearest
        self.image_size = image_size
        self.device = torch.device(device)

        # Feature extraction
        self.features: list[torch.Tensor] = []
        self.model = self._build_backbone(backbone)
        self._register_hooks()
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        # Memory bank (set after fit())
        self.memory_bank: torch.Tensor | None = None

        # Pooling layers (initialized lazily on first forward pass)
        self.avg: nn.AvgPool2d | None = None
        self.resize: nn.AdaptiveAvgPool2d | None = None
        self._fmap_size: int | None = None

    def _build_backbone(self, name: str) -> nn.Module:
        weights_map = {
            "wide_resnet50_2": models.Wide_ResNet50_2_Weights.DEFAULT,
            "resnet18": models.ResNet18_Weights.DEFAULT,
        }
        if name not in weights_map:
            raise ValueError(f"Unknown backbone '{name}'. Available: {list(weights_map)}")
        return getattr(models, name)(weights=weights_map[name])

    def _register_hooks(self) -> None:
        def hook(_module, _input, output):
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        self.features = []
        self.model(x)
        return self.features

    def _init_pooling(self, fmap_size: int) -> None:
        """Initialize pooling layers based on feature map size (called once)."""
        if self.avg is None:
            self.avg = nn.AvgPool2d(3, stride=1, padding=1)
            self.resize = nn.AdaptiveAvgPool2d(fmap_size)
            self._fmap_size = fmap_size

    def _extract_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch features from images.

        Returns:
            (N_patches, C) tensor where N_patches = H * W per image
        """
        feature_maps = self(images.to(self.device))
        self._init_pooling(feature_maps[0].shape[-2])
        resized = [self.resize(self.avg(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized, dim=1)  # (B, C1+C2, H, W)
        return patch.reshape(patch.shape[1], -1).T  # (H*W, C)

    def fit(self, train_dataloader) -> None:
        """Build memory bank from normal training images."""
        all_patches = []
        for images, _ in train_dataloader:
            patches = self._extract_patches(images)
            all_patches.append(patches.cpu())

        self.memory_bank = torch.cat(all_patches, dim=0)
        logger.info("Memory bank built: %d patches", self.memory_bank.shape[0])

        # Coreset subsampling
        if self.coreset_ratio < 1.0:
            target_n = max(1, int(self.coreset_ratio * self.memory_bank.shape[0]))
            logger.info("Coreset subsampling: %d -> %d patches", self.memory_bank.shape[0], target_n)
            coreset_idx = get_coreset(self.memory_bank, n=target_n, eps=self.eps_coreset)
            self.memory_bank = self.memory_bank[coreset_idx]
            logger.info("Memory bank after coreset: %d patches", self.memory_bank.shape[0])

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute anomaly score and segmentation map.

        Args:
            images: (B, C, H, W) tensor (B should be 1 for segmentation maps)

        Returns:
            (score, segm_map): scalar anomaly score and (1, 1, H, W) segmentation map
        """
        if self.memory_bank is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        patches = self._extract_patches(images)
        memory = self.memory_bank.to(patches.device)

        # Per-patch minimum distance to memory bank
        distances = torch.cdist(patches, memory, p=2.0)
        dist_score, dist_score_idxs = torch.min(distances, dim=1)

        # Find the most anomalous patch
        s_idx = torch.argmax(dist_score)
        s_star = torch.max(dist_score)

        # Weighted scoring using K-NN neighborhood (from the paper)
        m_star = memory[dist_score_idxs[s_idx]].unsqueeze(0)
        knn_dists = torch.cdist(m_star, memory, p=2.0)
        _, nn_idxs = knn_dists.topk(k=self.k_nearest, largest=False)

        m_star_neighbourhood = memory[nn_idxs[0, 1:]]
        m_test_star = patches[s_idx].unsqueeze(0)
        w_denominator = torch.linalg.norm(m_test_star - m_star_neighbourhood, dim=1)
        norm = torch.sqrt(torch.tensor(patches.shape[1], dtype=torch.float32))
        w = 1 - (torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm)))
        score = w * s_star

        # Build segmentation map
        fmap_size = self._fmap_size
        segm_map = dist_score.view(1, 1, fmap_size, fmap_size)
        segm_map = nn.functional.interpolate(
            segm_map,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        segm_map = _gaussian_blur(segm_map.squeeze(0))

        return score.cpu(), segm_map.cpu()

    def evaluate(self, test_dataloader) -> tuple[float, float]:
        """Evaluate on test set.

        Returns:
            (image_level_rocauc, pixel_level_rocauc)
        """
        image_preds: list[float] = []
        image_labels: list[int] = []
        pixel_preds: list[np.ndarray] = []
        pixel_labels: list[np.ndarray] = []

        for images, masks, labels in test_dataloader:
            image_labels.append(labels.item())
            pixel_labels.extend(masks.flatten().numpy())

            score, segm_map = self.predict(images)
            image_preds.append(score.item())
            pixel_preds.extend(segm_map.flatten().numpy())

        image_rocauc = roc_auc_score(image_labels, image_preds)
        pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)

        return image_rocauc, pixel_rocauc

    def calibrate_threshold(self, train_dataloader, percentile: int = 99) -> float:
        """Compute anomaly threshold from training scores.

        Runs predict on training (good) images and returns the Nth
        percentile score as the anomaly threshold.
        """
        scores = []
        for images, _ in train_dataloader:
            score, _ = self.predict(images)
            scores.append(score.item())
        return float(np.percentile(scores, percentile))

    def save(self, path: str | Path) -> None:
        """Save memory bank and metadata to .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            memory_bank=self.memory_bank.numpy() if self.memory_bank is not None else None,
            fmap_size=self._fmap_size,
            backbone=self.backbone_name,
            image_size=self.image_size,
        )
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load memory bank from .npz file."""
        data = np.load(str(path), allow_pickle=True)
        self.memory_bank = torch.tensor(data["memory_bank"])
        fmap_size = int(data["fmap_size"])
        self._init_pooling(fmap_size)
        logger.info("Model loaded from %s (%d patches)", path, self.memory_bank.shape[0])
