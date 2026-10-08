"""Training context passed to loss components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from marn.runtime.parameter_tree import ParameterTree


@dataclass
class TrainingContext:
    """Carry all information loss components need in one object.

    Built by the trainer from ``ForwardResult`` plus batch targets.  Individual
    loss components read only the fields they need, so the trainer does not
    have to know which losses are active.

    Attributes:
        predictions: Raw output from the target model forward pass.
        targets: Ground-truth labels or regression targets from the batch.
        latent_vectors: Mapping from stable latent names to their trainable
            tensors (directly from ``ForwardResult.latent_vectors``).
        generated_parameters: The ``ParameterTree`` used in this forward pass.
        perturbed_predictions: Target output from perturbed latent vectors.
            Populated only when stability loss is active and the model has
            been re-run with noise-injected latents.
        mapper_weights: Optional mapping from latent names to their mapper's
            modulated weight matrices.  Used by alignment loss to compute
            cosine distance between latent vectors and weight summaries.
    """

    predictions: Any
    targets: Any
    latent_vectors: dict[str, Tensor]
    generated_parameters: ParameterTree
    perturbed_predictions: Any | None = None
    mapper_weights: dict[str, Tensor] = field(default_factory=dict)
