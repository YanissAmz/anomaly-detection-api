import torch

from src.preprocessing.transforms import denormalize, get_inference_transform, get_train_transform


class TestTransforms:
    def test_train_output_shape(self):
        transform = get_train_transform(224)
        from PIL import Image

        img = Image.new("RGB", (300, 300))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_inference_output_shape(self):
        transform = get_inference_transform(256)
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        tensor = transform(img)
        assert tensor.shape == (3, 256, 256)

    def test_denormalize_range(self):
        tensor = torch.randn(3, 10, 10)
        result = denormalize(tensor)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
