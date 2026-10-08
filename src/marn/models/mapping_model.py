"""User-facing model that trains a target through low-dimensional parameter mappings."""

from __future__ import annotations

from typing import Any

from torch import nn, Tensor
from marn.generators.base import (
    MapperFactory,
    ParameterGenerator,
    default_mapper_factory,
)
from marn.models.forward_result import ForwardResult
from marn.models.target_model import TargetModel
from marn.strategies.base import GenerationStrategy
from marn.strategies.layerwise import LayerwiseStrategy
from marn.strategies.slvt import SLVTStrategy

_STRATEGY_ALIASES: dict[str, GenerationStrategy] = {
    "layerwise": LayerwiseStrategy(),
    "slvt": SLVTStrategy(),
}


class MappingModel(nn.Module):
    """Train a target model through a low-dimensional parameter manifold.

    The target model is frozen and never directly optimized.  Trainable latent
    vectors are projected by fixed-weight mappers into a complete set of target
    parameters, which are then used for stateless forward execution via
    ``torch.func.functional_call``.

    Example::

        model = MappingModel(
            target_model=ResNet18(),
            latent_dim=2048,
            strategy="layerwise",
        )
        result = model(inputs)          # ForwardResult
        result.predictions              # target output
        result.generated_parameters     # ParameterTree for loss computation
        result.latent_vectors           # dict[str, Tensor] for stability/alignment

    Args:
        target_model: Any ``nn.Module`` to train through parameter mapping.
            Tied parameters and parametrized modules are rejected until their
            aliasing contracts are supported.
        latent_dim: Dimensionality of each trainable latent vector.
        strategy: Generation strategy — ``"layerwise"`` (default), ``"slvt"``,
            or a ``GenerationStrategy`` instance for custom grouping.
        mapper_factory: Callable ``(latent_dim, output_dim) -> BaseMapper``.
            Defaults to an orthogonal MLP mapper with additive modulation.
    """

    def __init__(
        self,
        target_model: nn.Module,
        latent_dim: int = 256,
        strategy: str | GenerationStrategy = "layerwise",
        *,
        mapper_factory: MapperFactory | None = None,
        parameter_scaling: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.target = TargetModel(target_model)
        self.latent_dim = latent_dim

        resolved_strategy = self._resolve_strategy(strategy)
        factory = mapper_factory or default_mapper_factory
        self.generator: ParameterGenerator = resolved_strategy.build(
            self.target.parameter_spec,
            latent_dim,
            factory,
        )

        if parameter_scaling:
            self.generator.initialize_parameter_scaling(target_model)

    @staticmethod
    def _resolve_strategy(strategy: str | GenerationStrategy) -> GenerationStrategy:
        if isinstance(strategy, GenerationStrategy):
            return strategy
        if isinstance(strategy, str):
            resolved = _STRATEGY_ALIASES.get(strategy)
            if resolved is None:
                valid = ", ".join(sorted(_STRATEGY_ALIASES))
                raise ValueError(f"Unknown strategy {strategy!r}; choose from: {valid}")
            return resolved
        raise TypeError(f"strategy must be a string or GenerationStrategy, got {type(strategy)}")

    def forward(
        self, *args: Any, latents: dict[str, Tensor] | None = None, **kwargs: Any
    ) -> ForwardResult:
        """Generate target parameters and run a stateless forward pass.

        All positional and keyword arguments are forwarded to the target model.

        Returns:
            A ``ForwardResult`` containing predictions, the generated parameter
            tree, and a dictionary of named latent vectors.
        """
        generated_parameters = self.generator.generate_parameters(latents)
        predictions = self.target(generated_parameters, *args, **kwargs)
        latent_vectors = dict(self.generator.named_latent_vectors())
        return ForwardResult(
            predictions=predictions,
            generated_parameters=generated_parameters,
            latent_vectors=latent_vectors,
        )

    def modulated_mapper_weights(self) -> dict[str, Tensor]:
        """Return a mapping from latent names to their mapper's modulated weights."""
        if hasattr(self.generator, "modulated_mapper_weights"):
            return self.generator.modulated_mapper_weights()
        return {}

    @property
    def trainable_parameter_count(self) -> int:
        """Number of trainable parameters (latent vectors only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def target_parameter_count(self) -> int:
        """Total number of target model parameters."""
        return self.target.parameter_spec.total_numel

    @property
    def compression_ratio(self) -> float:
        """Ratio of target parameters to trainable mapping parameters.

        A value of 10.0 means the mapping uses 10× fewer trainable parameters
        than direct training of the target.  Returns ``float('inf')`` when
        there are zero trainable parameters.
        """
        trainable = self.trainable_parameter_count
        if trainable == 0:
            return float("inf")
        return self.target_parameter_count / trainable
