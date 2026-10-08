"""Explicit grouped strategy."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from marn.generators import GroupedGenerator, MapperFactory, ParameterGenerator
from marn.runtime import ParameterSpec
from marn.strategies.base import GenerationStrategy


@dataclass(frozen=True)
class GroupedStrategy(GenerationStrategy):
    groups: Mapping[str, Sequence[str]]

    def build(
        self, spec: ParameterSpec, latent_dim: int, mapper_factory: MapperFactory
    ) -> ParameterGenerator:
        return GroupedGenerator(spec, self.groups, latent_dim, mapper_factory=mapper_factory)
