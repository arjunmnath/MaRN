"""Fine-tuning generation strategy."""

from __future__ import annotations

from torch import Tensor

from marn.generators import MapperFactory, ParameterGenerator, FineTuningGenerator
from marn.runtime import ParameterSpec
from marn.strategies.base import GenerationStrategy


class FineTuningStrategy(GenerationStrategy):
    """Fine-tuning strategy that builds a FineTuningGenerator.

    Use this strategy to wrap a target model for parameter-efficient fine-tuning
    based on the pre-trained weights.

    Example::

        strategy = FineTuningStrategy(
            pretrained_parameters=dict(target.named_parameters()),
            L=250,
            alpha=0.01,
        )
        model = MappingModel(target, latent_dim=64, strategy=strategy)

    Args:
        pretrained_parameters: Dict mapping parameter names to their original
            pretrained PyTorch Tensors.
        L: Block size (default: 250). Each generated element modulates L consecutive
            flattened parameter values.
        alpha: Scaling factor for the applied modulation.
    """

    def __init__(
        self,
        pretrained_parameters: dict[str, Tensor],
        L: int = 250,
        alpha: float = 0.01,
    ) -> None:
        self.pretrained_parameters = pretrained_parameters
        self.L = L
        self.alpha = alpha

    def build(
        self,
        spec: ParameterSpec,
        latent_dim: int,
        mapper_factory: MapperFactory,
    ) -> ParameterGenerator:
        return FineTuningGenerator(
            parameter_spec=spec,
            latent_dim=latent_dim,
            pretrained_parameters=self.pretrained_parameters,
            L=self.L,
            alpha=self.alpha,
            mapper_factory=mapper_factory,
        )
