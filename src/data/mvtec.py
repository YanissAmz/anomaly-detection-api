"""MVTec AD dataset loading and downloading.

Supports all 15 industrial object categories with automatic download,
train/test split, and ground truth mask loading for evaluation.
"""

from __future__ import annotations

import logging
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.config import Settings
from src.preprocessing.transforms import get_image_transform, get_mask_transform

logger = logging.getLogger(__name__)

MVTEC_CLASSES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

CLASS_DOWNLOAD_LINKS = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz",
}


def download_mvtec_class(cls: str, target_dir: str | Path) -> None:
    """Download and extract a single MVTec AD class."""
    if cls not in CLASS_DOWNLOAD_LINKS:
        raise ValueError(f"Unknown class '{cls}'. Available: {MVTEC_CLASSES}")

    target_dir = Path(target_dir)
    if (target_dir / cls).is_dir():
        logger.info("Class '%s' already exists in '%s'", cls, target_dir)
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    url = CLASS_DOWNLOAD_LINKS[cls]
    archive_path = target_dir / f"{cls}.tar.xz"

    logger.info("Downloading '%s' from MVTec AD...", cls)
    urllib.request.urlretrieve(url, archive_path)

    logger.info("Extracting '%s'...", cls)
    with tarfile.open(archive_path) as tar:
        tar.extractall(target_dir)
    archive_path.unlink()
    logger.info("Class '%s' ready.", cls)


class MVTecTrainDataset(ImageFolder):
    """MVTec AD training dataset (only 'good' images)."""

    def __init__(
        self,
        root: str | Path,
        cls: str,
        transform=None,
    ):
        super().__init__(root=str(Path(root) / cls / "train"), transform=transform)
        self.cls = cls


class MVTecTestDataset(Dataset):
    """MVTec AD test dataset with ground truth masks.

    Returns (image, mask, label) tuples where:
    - label=0 for good samples, label=1 for defective
    - mask is the ground truth segmentation (zeros for good samples)
    """

    def __init__(
        self,
        root: str | Path,
        cls: str,
        image_size: int = 224,
        transform=None,
        mask_transform=None,
    ):
        self.root = Path(root) / cls / "test"
        self.cls = cls
        self.image_size = image_size
        self.transform = transform
        self.mask_transform = mask_transform

        # Collect all image paths
        self.samples: list[tuple[str, int]] = []
        for defect_type in sorted(self.root.iterdir()):
            if not defect_type.is_dir():
                continue
            label = 0 if defect_type.name == "good" else 1
            for img_path in sorted(defect_type.glob("*.png")):
                self.samples.append((str(img_path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")

        if label == 0:
            mask = Image.new("L", (self.image_size, self.image_size))
        else:
            mask_path = path.replace("test", "ground_truth")
            mask_path = mask_path.replace(".png", "_mask.png")
            mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            mask = mask[:1]  # keep only first channel
        else:
            import torch

            mask = torch.zeros(1, self.image_size, self.image_size)

        return image, mask, label


def get_dataloaders(
    cfg: Settings,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders from config."""
    image_transform = get_image_transform(
        image_size=cfg.model.image_size,
        resize=cfg.model.resize,
        backbone=cfg.model.backbone,
    )
    mask_transform = get_mask_transform(
        image_size=cfg.model.image_size,
        resize=cfg.model.resize,
    )

    data_dir = cfg.dataset.data_dir
    cls = cfg.dataset.category

    if cfg.dataset.download:
        download_mvtec_class(cls, data_dir)

    train_ds = MVTecTrainDataset(data_dir, cls, transform=image_transform)
    test_ds = MVTecTestDataset(
        data_dir,
        cls,
        image_size=cfg.model.image_size,
        transform=image_transform,
        mask_transform=mask_transform,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    return train_dl, test_dl
