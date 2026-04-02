import tempfile
from pathlib import Path

from PIL import Image

from src.data.mvtec import MVTEC_CLASSES, MVTecTestDataset


class TestMVTecClasses:
    def test_all_15_classes(self):
        assert len(MVTEC_CLASSES) == 15
        assert "bottle" in MVTEC_CLASSES
        assert "zipper" in MVTEC_CLASSES


class TestMVTecTestDataset:
    def _create_mini_dataset(self, tmp: Path, cls: str = "bottle"):
        """Create a minimal MVTec-like directory structure for testing."""
        # Test good
        good_dir = tmp / cls / "test" / "good"
        good_dir.mkdir(parents=True)
        Image.new("RGB", (256, 256)).save(good_dir / "000.png")

        # Test defect
        defect_dir = tmp / cls / "test" / "broken_large"
        defect_dir.mkdir(parents=True)
        Image.new("RGB", (256, 256)).save(defect_dir / "000.png")

        # Ground truth mask
        gt_dir = tmp / cls / "ground_truth" / "broken_large"
        gt_dir.mkdir(parents=True)
        Image.new("L", (256, 256), color=255).save(gt_dir / "000_mask.png")

        return tmp

    def test_sample_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._create_mini_dataset(Path(tmp))
            ds = MVTecTestDataset(root, "bottle", image_size=224)
            assert len(ds) == 2

    def test_labels_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._create_mini_dataset(Path(tmp))
            ds = MVTecTestDataset(root, "bottle", image_size=224)
            labels = [ds[i][2] for i in range(len(ds))]
            # Should have both good (0) and defective (1) samples
            assert 0 in labels
            assert 1 in labels
