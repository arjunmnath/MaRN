"""LoRA-style low-rank modulation decoded from a latent vector."""

from torch import Tensor

from marn.modulation.base import BaseModulation


class LowRankModulation(BaseModulation):
    """Decode ``z`` into factors and apply ``W' = W + alpha/rank * AB``."""

    def __init__(self, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.rank = rank
        self.alpha = alpha

    def latent_dim_for(self, weights: Tensor) -> int:
        if weights.ndim != 2:
            raise ValueError("Low-rank modulation expects a matrix")
        rows, columns = weights.shape
        return self.rank * (rows + columns)

    def modulate(self, weights: Tensor, latent: Tensor) -> Tensor:
        expected = self.latent_dim_for(weights)
        if latent.ndim != 1 or latent.numel() != expected:
            raise ValueError(f"Expected a 1D latent vector with {expected} values")
        rows, columns = weights.shape
        split = rows * self.rank
        left = latent[:split].view(rows, self.rank)
        right = latent[split:].view(self.rank, columns)
        return weights + (self.alpha / self.rank) * (left @ right)
