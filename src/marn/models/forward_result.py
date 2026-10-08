"""Typed forward-pass result for MappingModel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from marn.runtime.parameter_tree import ParameterTree


@dataclass
class ForwardResult:
    """Structured output from ``MappingModel.forward()``.

    Carries autograd-live tensors so downstream losses and the trainer can
    access generated parameters and latent vectors without positional
    argument gymnastics.

    Attributes:
        predictions: Raw output from the target model forward pass.
        generated_parameters: The ``ParameterTree`` produced by the generator.
        latent_vectors: Mapping from stable latent names to their trainable tensors.
    """

    predictions: Any
    generated_parameters: ParameterTree
    latent_vectors: dict[str, Tensor]
