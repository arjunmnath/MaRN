"""Interface for latent-conditioned tensor modulation."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseModulation(nn.Module, ABC):
    """Transform a fixed tensor using a latent vector."""

    def forward(self, weights: Tensor, latent: Tensor) -> Tensor:
        return self.modulate(weights, latent)

    @abstractmethod
    def modulate(self, weights: Tensor, latent: Tensor) -> Tensor:
        """Return modulated weights without mutating either input."""
