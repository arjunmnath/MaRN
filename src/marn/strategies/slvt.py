"""Single-vector strategy."""

from dataclasses import dataclass

from marn.generators import MapperFactory, ParameterGenerator, SingleVectorGenerator
from marn.runtime import ParameterSpec
from marn.strategies.base import GenerationStrategy


@dataclass(frozen=True)
class SLVTStrategy(GenerationStrategy):
    max_projection_elements: int = 100_000_000
    allow_large: bool = False

    def build(
        self, spec: ParameterSpec, latent_dim: int, mapper_factory: MapperFactory
    ) -> ParameterGenerator:
        return SingleVectorGenerator(
            spec,
            latent_dim,
            mapper_factory=mapper_factory,
            max_projection_elements=self.max_projection_elements,
            allow_large=self.allow_large,
        )
