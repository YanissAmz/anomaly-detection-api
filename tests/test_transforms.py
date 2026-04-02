import torch
from PIL import Image

from src.preprocessing.transforms import (
    denormalize,
    get_image_transform,
    get_mask_transform,
)


class TestImageTransform:
    def test_vanilla_output_shape(self):
        transform = get_image_transform(image_size=224, resize=256)
        img = Image.new("RGB", (300, 300))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_clip_output_shape(self):
        transform = get_image_transform(image_size=288, resize=288, backbone="RN50x4")
        img = Image.new("RGB", (400, 400))
        tensor = transform(img)
        assert tensor.shape == (3, 288, 288)

    def test_center_crop_from_rectangular(self):
        transform = get_image_transform(image_size=224, resize=256)
        img = Image.new("RGB", (500, 300))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)


class TestMaskTransform:
    def test_output_shape(self):
        transform = get_mask_transform(image_size=224, resize=256)
        mask = Image.new("L", (300, 300))
        tensor = transform(mask)
        assert tensor.shape == (1, 224, 224)

    def test_binary_values_preserved(self):
        """Nearest interpolation should preserve binary mask values."""
        transform = get_mask_transform(image_size=224, resize=256)
        mask = Image.new("L", (256, 256), color=255)
        tensor = transform(mask)
        assert tensor.max() == 1.0
        assert tensor.min() == 1.0


class TestDenormalize:
    def test_output_range(self):
        tensor = torch.randn(3, 10, 10)
        result = denormalize(tensor.clone())
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_clip_backbone(self):
        tensor = torch.randn(3, 10, 10)
        result = denormalize(tensor.clone(), backbone="RN50")
        assert result.min() >= 0.0
        assert result.max() <= 1.0
