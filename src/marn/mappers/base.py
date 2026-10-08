"""Interface for latent-to-parameter descriptor mappers."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseMapper(nn.Module, ABC):
    """Map one latent vector to a flat generated descriptor."""

    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        if latent_dim <= 0 or output_dim <= 0:
            raise ValueError("latent_dim and output_dim must be positive")
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self._last_modulated_weight: Tensor | None = None

    def _validate_latent(self, latent: Tensor) -> None:
        if latent.ndim != 1 or latent.numel() != self.latent_dim:
            raise ValueError(
                f"Expected a 1D latent vector with {self.latent_dim} values, "
                f"got shape {tuple(latent.shape)}"
            )

    @abstractmethod
    def forward(self, latent: Tensor) -> Tensor:
        """Return a one-dimensional descriptor of length ``output_dim``."""
