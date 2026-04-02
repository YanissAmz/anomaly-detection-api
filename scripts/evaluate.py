"""Evaluate PatchCore on MVTec AD test set.

Usage:
    python scripts/evaluate.py                          # evaluate default category
    python scripts/evaluate.py --category hazelnut
    python scripts/evaluate.py --category all
"""

from __future__ import annotations

import argparse
import logging

from torch.utils.data import DataLoader

from src.config import load_config, resolve_device
from src.data.mvtec import (
    MVTEC_CLASSES,
    MVTecTestDataset,
    download_mvtec_class,
)
from src.models.patchcore import PatchCore
from src.preprocessing.transforms import get_image_transform, get_mask_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_category(category: str, cfg) -> tuple[float, float]:
    device = resolve_device(cfg.inference.device)
    cache_path = (
        f"{cfg.cache.dir}/{category}_{cfg.model.backbone}_f{cfg.model.coreset_ratio:.3f}.npz"
    )

    model = PatchCore(
        backbone=cfg.model.backbone,
        coreset_ratio=cfg.model.coreset_ratio,
        k_nearest=cfg.model.k_nearest,
        image_size=cfg.model.image_size,
        device=device,
    )

    try:
        model.load(cache_path)
    except FileNotFoundError:
        logger.error("No cached model for '%s'. Run train.py first.", category)
        return 0.0, 0.0

    transform = get_image_transform(
        image_size=cfg.model.image_size, resize=cfg.model.resize, backbone=cfg.model.backbone
    )
    mask_transform = get_mask_transform(image_size=cfg.model.image_size, resize=cfg.model.resize)

    download_mvtec_class(category, cfg.dataset.data_dir)
    test_ds = MVTecTestDataset(
        cfg.dataset.data_dir,
        category,
        image_size=cfg.model.image_size,
        transform=transform,
        mask_transform=mask_transform,
    )
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    image_auroc, pixel_auroc = model.evaluate(test_dl)
    logger.info(
        "%s | Image AUROC: %.4f | Pixel AUROC: %.4f",
        category,
        image_auroc,
        pixel_auroc,
    )
    return image_auroc, pixel_auroc


def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore on MVTec AD")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.category:
        cfg.dataset.category = args.category

    if cfg.dataset.category == "all":
        results = {}
        for cat in MVTEC_CLASSES:
            results[cat] = evaluate_category(cat, cfg)

        print("\n" + "=" * 55)
        print(f"{'Category':15} {'Image AUROC':>12} {'Pixel AUROC':>12}")
        print("-" * 55)
        for cat, (img, pxl) in results.items():
            print(f"{cat:15} {img:>12.4f} {pxl:>12.4f}")
        print("=" * 55)

        valid = [(i, p) for i, p in results.values() if i > 0]
        if valid:
            avg_img = sum(i for i, _ in valid) / len(valid)
            avg_pxl = sum(p for _, p in valid) / len(valid)
            print(f"{'AVERAGE':15} {avg_img:>12.4f} {avg_pxl:>12.4f}")
    else:
        evaluate_category(cfg.dataset.category, cfg)


if __name__ == "__main__":
    main()
