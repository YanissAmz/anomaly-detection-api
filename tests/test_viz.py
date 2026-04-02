import numpy as np

from src.demo.viz import overlay_heatmap


class TestOverlayHeatmap:
    def test_output_shape(self):
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        amap = np.random.rand(224, 224).astype(np.float32)
        result = overlay_heatmap(image, amap)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8

    def test_resize_mismatch(self):
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        amap = np.random.rand(28, 28).astype(np.float32)
        result = overlay_heatmap(image, amap)
        assert result.shape == (224, 224, 3)

    def test_zero_map(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        amap = np.zeros((100, 100), dtype=np.float32)
        result = overlay_heatmap(image, amap)
        assert result.shape == (100, 100, 3)
