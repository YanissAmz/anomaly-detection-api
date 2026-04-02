"""Train PatchCore on MVTec AD categories and save memory banks.

Usage:
    python scripts/train.py                          # train on default category (bottle)
    python scripts/train.py --category hazelnut      # train on specific category
    python scripts/train.py --category all           # train on all 15 categories
    python scripts/train.py --config configs/custom.yaml
"""

from __future__ import annotations

import argparse
import logging
import time

from torch.utils.data import DataLoader

from src.config import load_config, resolve_device
from src.data.mvtec import MVTEC_CLASSES, MVTecTrainDataset, download_mvtec_class
from src.models.patchcore import PatchCore
from src.preprocessing.transforms import get_image_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train_category(category: str, cfg) -> None:
    device = resolve_device(cfg.inference.device)
    logger.info("Training on '%s' (backbone=%s, device=%s)", category, cfg.model.backbone, device)

    download_mvtec_class(category, cfg.dataset.data_dir)

    transform = get_image_transform(
        image_size=cfg.model.image_size,
        resize=cfg.model.resize,
        backbone=cfg.model.backbone,
    )
    train_ds = MVTecTrainDataset(cfg.dataset.data_dir, category, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)

    model = PatchCore(
        backbone=cfg.model.backbone,
        coreset_ratio=cfg.model.coreset_ratio,
        eps_coreset=cfg.model.eps_coreset,
        k_nearest=cfg.model.k_nearest,
        image_size=cfg.model.image_size,
        device=device,
    )

    t0 = time.time()
    model.fit(train_dl)
    elapsed = time.time() - t0
    logger.info(
        "Training done in %.1fs | Memory bank: %d patches", elapsed, model.memory_bank.shape[0]
    )

    # Save
    cache_path = (
        f"{cfg.cache.dir}/{category}_{cfg.model.backbone}_f{cfg.model.coreset_ratio:.3f}.npz"
    )
    model.save(cache_path)

    # Calibrate threshold
    threshold = model.calibrate_threshold(train_dl, percentile=cfg.inference.threshold_percentile)
    logger.info("Threshold (p%d): %.4f", cfg.inference.threshold_percentile, threshold)


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec AD")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--category", default=None, help="MVTec category (or 'all')")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.category:
        cfg.dataset.category = args.category

    if cfg.dataset.category == "all":
        for cat in MVTEC_CLASSES:
            train_category(cat, cfg)
    else:
        train_category(cfg.dataset.category, cfg)


if __name__ == "__main__":
    main()
