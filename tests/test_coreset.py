import torch

from src.models.coreset import get_coreset


class TestGetCoreset:
    def test_reduces_size(self):
        bank = torch.randn(1000, 128)
        idx = get_coreset(bank, n=50, eps=0.90)
        assert idx.shape == (50,)

    def test_indices_valid(self):
        bank = torch.randn(500, 64)
        idx = get_coreset(bank, n=30, eps=0.90)
        assert idx.max() < 500
        assert idx.min() >= 0

    def test_indices_unique(self):
        bank = torch.randn(200, 64)
        idx = get_coreset(bank, n=20, eps=0.90)
        assert len(idx.unique()) == 20

    def test_n_larger_than_bank(self):
        bank = torch.randn(10, 32)
        idx = get_coreset(bank, n=100, eps=0.90)
        assert idx.shape == (10,)
