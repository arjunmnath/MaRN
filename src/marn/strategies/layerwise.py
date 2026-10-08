"""Layer-wise strategy."""

from dataclasses import dataclass

from marn.generators import LayerwiseGenerator, MapperFactory, ParameterGenerator
from marn.runtime import ParameterSpec
from marn.strategies.base import GenerationStrategy


@dataclass(frozen=True)
class LayerwiseStrategy(GenerationStrategy):
    def build(
        self, spec: ParameterSpec, latent_dim: int, mapper_factory: MapperFactory
    ) -> ParameterGenerator:
        return LayerwiseGenerator(spec, latent_dim, mapper_factory=mapper_factory)
