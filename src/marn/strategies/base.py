"""Strategy interface for constructing parameter generators."""

from abc import ABC, abstractmethod

from marn.generators import MapperFactory, ParameterGenerator
from marn.runtime import ParameterSpec


class GenerationStrategy(ABC):
    @abstractmethod
    def build(
        self, spec: ParameterSpec, latent_dim: int, mapper_factory: MapperFactory
    ) -> ParameterGenerator:
        """Build a generator for ``spec``."""
