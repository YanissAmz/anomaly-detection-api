"""Coreset subsampling for PatchCore memory bank.

Uses SparseRandomProjection for dimensionality reduction followed by
greedy farthest-point sampling. Ported from the PFE implementation
with GPU acceleration support.
"""

from __future__ import annotations

import logging

import torch
from sklearn.random_projection import SparseRandomProjection

logger = logging.getLogger(__name__)


def get_coreset(
    memory_bank: torch.Tensor,
    n: int,
    eps: float = 0.90,
) -> torch.Tensor:
    """Select a coreset of size n from the memory bank.

    Uses SparseRandomProjection to reduce dimensionality, then greedily
    selects the farthest points to build a representative subset.

    Args:
        memory_bank: (N, D) tensor of patch features
        n: target coreset size
        eps: epsilon parameter for SparseRandomProjection

    Returns:
        (n,) tensor of selected indices
    """
    if n >= memory_bank.shape[0]:
        return torch.arange(memory_bank.shape[0])

    # Dimensionality reduction for faster distance computation
    try:
        projector = SparseRandomProjection(eps=eps)
        projected = torch.tensor(projector.fit_transform(memory_bank.cpu().numpy()))
    except ValueError:
        logger.warning("SparseRandomProjection failed (try increasing eps). Using raw features.")
        projected = memory_bank.cpu()

    # Move to GPU if available for faster greedy selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    projected = projected.to(device)

    # Greedy farthest-point sampling
    coreset_idx: list[torch.Tensor] = []
    idx = 0
    last_item = projected[idx : idx + 1]
    coreset_idx.append(torch.tensor(idx))
    min_distances = torch.linalg.norm(projected - last_item, dim=1, keepdim=True)

    for _ in range(n - 1):
        distances = torch.linalg.norm(projected - last_item, dim=1, keepdim=True)
        min_distances = torch.minimum(distances, min_distances)
        idx = torch.argmax(min_distances).item()

        last_item = projected[idx : idx + 1]
        min_distances[idx] = 0
        coreset_idx.append(torch.tensor(idx))

    return torch.stack(coreset_idx)
