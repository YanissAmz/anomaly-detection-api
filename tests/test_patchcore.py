import torch

from src.models.patchcore import PatchCore, _gaussian_blur


class TestGaussianBlur:
    def test_output_shape(self):
        x = torch.rand(1, 32, 32)
        result = _gaussian_blur(x)
        assert result.shape == (1, 32, 32)

    def test_zeros_unchanged(self):
        x = torch.zeros(1, 16, 16)
        result = _gaussian_blur(x)
        assert result.sum() == 0


class TestPatchCoreInit:
    def test_default_init(self):
        model = PatchCore(device="cpu")
        assert model.memory_bank is None
        assert model.k_nearest == 3

    def test_hooks_registered(self):
        model = PatchCore(device="cpu")
        x = torch.randn(1, 3, 224, 224)
        features = model(x)
        assert len(features) == 2  # layer2 and layer3

    def test_feature_map_shapes(self):
        model = PatchCore(device="cpu")
        x = torch.randn(1, 3, 224, 224)
        features = model(x)
        # WideResNet50: layer2 -> 28x28, layer3 -> 14x14
        assert features[0].shape[2] == 28
        assert features[1].shape[2] == 14


class TestPatchCoreExtract:
    def test_extract_patches_shape(self):
        model = PatchCore(device="cpu")
        x = torch.randn(1, 3, 224, 224)
        patches = model._extract_patches(x)
        # 28*28 = 784 patches, channels = 512 (layer2) + 1024 (layer3) = 1536
        assert patches.shape == (784, 1536)

    def test_pooling_initialized(self):
        model = PatchCore(device="cpu")
        assert model.avg is None
        x = torch.randn(1, 3, 224, 224)
        model._extract_patches(x)
        assert model.avg is not None
        assert model._fmap_size == 28


class TestPatchCoreSaveLoad:
    def test_roundtrip(self, tmp_path):
        model = PatchCore(device="cpu")
        model.memory_bank = torch.randn(100, 1536)
        model._init_pooling(28)

        save_path = tmp_path / "test_model.npz"
        model.save(save_path)

        model2 = PatchCore(device="cpu")
        model2.load(save_path)

        assert model2.memory_bank.shape == (100, 1536)
        assert model2._fmap_size == 28
        assert torch.allclose(model.memory_bank, model2.memory_bank)
