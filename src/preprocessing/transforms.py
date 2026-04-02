"""Image preprocessing transforms for PatchCore anomaly detection.

Single source of truth for all transforms — eliminates the duplication
found in the PFE repo where transforms were created in 4+ separate places.
"""

from torchvision import transforms

# ImageNet normalization (for vanilla backbones: WideResNet50, ResNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP normalization
CLIP_MEAN = (0.481, 0.457, 0.408)
CLIP_STD = (0.268, 0.261, 0.275)


def _is_clip_backbone(backbone: str) -> bool:
    return backbone in ("RN50", "RN50x4", "RN50x16", "RN101")


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_image_transform(
    image_size: int = 224,
    resize: int = 256,
    backbone: str = "wide_resnet50_2",
) -> transforms.Compose:
    """Build the image transform pipeline.

    Uses Resize -> CenterCrop -> ToTensor -> Normalize, matching the
    PatchCore paper's preprocessing. Normalization constants depend
    on the backbone (ImageNet for vanilla, CLIP for CLIP-based).
    """
    if _is_clip_backbone(backbone):
        return transforms.Compose(
            [
                transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_mask_transform(
    image_size: int = 224,
    resize: int = 256,
) -> transforms.Compose:
    """Build the ground truth mask transform pipeline.

    Uses nearest-neighbor interpolation to preserve binary mask values.
    """
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def denormalize(tensor, backbone: str = "wide_resnet50_2"):
    """Reverse normalization for visualization.

    Returns tensor with values clamped to [0, 1].
    """
    if _is_clip_backbone(backbone):
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    for t, m, s in zip(tensor, mean, std, strict=True):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)
