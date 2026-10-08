"""Base contract and factories for parameter generators."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

from torch import Tensor, nn

from marn.mappers import BaseMapper, MLPMapper
from marn.modulation import AdditiveModulation
from marn.runtime import ParameterSpec, ParameterTree

MapperFactory = Callable[[int, int], BaseMapper]


def default_mapper_factory(latent_dim: int, output_dim: int) -> BaseMapper:
    return MLPMapper(
        latent_dim,
        output_dim,
        modulation=AdditiveModulation(),
    )


class ParameterGenerator(nn.Module, ABC):
    """Generate a complete named parameter tree for one target specification."""

    def __init__(self, parameter_spec: ParameterSpec) -> None:
        super().__init__()
        self.parameter_spec = parameter_spec

    @abstractmethod
    def generate_parameters(self, latents: dict[str, Tensor] | None = None) -> ParameterTree:
        """Generate and validate a complete target parameter tree."""

    @abstractmethod
    def named_latent_vectors(self) -> Iterator[tuple[str, Tensor]]:
        """Yield stable latent names and their trainable tensors."""

    def initialize_parameter_scaling(self, target_model: nn.Module) -> None:
        """Compute and register parameter scales from the original target model."""
        pass

    def modulated_mapper_weights(self) -> dict[str, Tensor]:
        """Return a mapping from latent names to their mapper's modulated weights."""
        return {}

    def forward(self, latents: dict[str, Tensor] | None = None) -> ParameterTree:
        return self.generate_parameters(latents)
