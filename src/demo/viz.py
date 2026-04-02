"""Visualization utilities for anomaly detection.

Shared between the API heatmap endpoint and the Streamlit dashboard.
"""

from __future__ import annotations

import cv2
import numpy as np


def overlay_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay anomaly heatmap on the original image.

    Args:
        image: (H, W, 3) uint8 RGB image
        anomaly_map: (H, W) float anomaly scores
        alpha: blending factor (0 = only image, 1 = only heatmap)
        colormap: OpenCV colormap

    Returns:
        (H, W, 3) uint8 RGB overlay image
    """
    # Resize anomaly map to match image if needed
    if anomaly_map.shape[:2] != image.shape[:2]:
        anomaly_map = cv2.resize(anomaly_map, (image.shape[1], image.shape[0]))

    # Normalize to 0-255
    if anomaly_map.max() > anomaly_map.min():
        norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    else:
        norm_map = np.zeros_like(anomaly_map)
    heatmap = cv2.applyColorMap((norm_map * 255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    return overlay


def tensor_to_image(tensor, backbone: str = "wide_resnet50_2") -> np.ndarray:
    """Convert a normalized tensor to a displayable uint8 RGB numpy array."""
    from src.preprocessing.transforms import denormalize

    img = denormalize(tensor.clone(), backbone=backbone)
    img = img.permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)
