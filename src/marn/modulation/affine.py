"""Parameter-free latent-conditioned affine modulation."""

import torch
from torch import Tensor

from marn.modulation.base import BaseModulation


class AffineModulation(BaseModulation):
    """Apply per-column latent scale and shift to a matrix."""

    def __init__(self, scale: float = 0.01, shift: float = 0.01) -> None:
        super().__init__()
        if scale < 0 or shift < 0:
            raise ValueError("scale and shift must be non-negative")
        self.scale = scale
        self.shift = shift

    def modulate(self, weights: Tensor, latent: Tensor) -> Tensor:
        if weights.ndim != 2 or latent.ndim != 1:
            raise ValueError("Affine modulation expects a matrix and a 1D latent vector")
        if weights.shape[1] != latent.numel():
            raise ValueError(
                f"Latent length {latent.numel()} does not match weight input width "
                f"{weights.shape[1]}"
            )
        scale = 1.0 + self.scale * torch.tanh(latent)
        shift = self.shift * latent
        return weights * scale.unsqueeze(0) + shift.unsqueeze(0)
