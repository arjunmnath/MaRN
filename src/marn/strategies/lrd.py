"""Low-Rank Decomposition (LRD) strategy."""

from __future__ import annotations

from marn.generators import MapperFactory, ParameterGenerator, LRDGenerator
from marn.runtime import ParameterSpec
from marn.strategies.base import GenerationStrategy


class LRDStrategy(GenerationStrategy):
    """LRD strategy that builds an LRDGenerator.

    Use this strategy to wrap a target model with target-side Low-Rank Decomposition
    to reduce the mapper output size and parameter counts for large linear layers.

    Example::

        strategy = LRDStrategy(rank=16)
        model = MappingModel(target, latent_dim=128, strategy=strategy)

    Args:
        rank: Decomposition rank. Used for factoring 2D weight parameters.
    """

    def __init__(self, rank: int) -> None:
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank

    def build(
        self,
        spec: ParameterSpec,
        latent_dim: int,
        mapper_factory: MapperFactory,
    ) -> ParameterGenerator:
        return LRDGenerator(
            parameter_spec=spec,
            latent_dim=latent_dim,
            rank=self.rank,
            mapper_factory=mapper_factory,
        )
