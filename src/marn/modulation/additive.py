"""Paper-compatible additive mapping-weight modulation."""

from torch import Tensor

from marn.modulation.base import BaseModulation


class AdditiveModulation(BaseModulation):
    """Apply ``W' = W + alpha * z`` across the rows of a matrix."""

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha

    def modulate(self, weights: Tensor, latent: Tensor) -> Tensor:
        if weights.ndim != 2 or latent.ndim != 1:
            raise ValueError("Additive modulation expects a matrix and a 1D latent vector")
        if weights.shape[1] != latent.numel():
            raise ValueError(
                f"Latent length {latent.numel()} does not match weight input width "
                f"{weights.shape[1]}"
            )
        return weights + self.alpha * latent.unsqueeze(0)
