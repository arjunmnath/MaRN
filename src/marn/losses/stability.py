"""Latent-perturbation stability loss.

Measures prediction sensitivity to small latent perturbations by comparing
the model's output at ``z`` with output at ``z + epsilon * noise``.
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from marn.losses.base import BaseLoss
from marn.losses.context import TrainingContext
from marn.losses.outputs import LossOutput


def _flatten_predictions(predictions: object) -> Tensor:
    """Convert structured predictions to a flat tensor for comparison.

    Handles tuples (e.g. LSTM output) by taking the first element, and
    ensures we get a dense tensor for MSE comparison.
    """
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if not isinstance(predictions, Tensor):
        raise TypeError(
            f"Cannot flatten predictions of type {type(predictions)}; "
            "expected a Tensor or a tuple whose first element is a Tensor"
        )
    return predictions


class StabilityLoss(BaseLoss):
    """Penalize prediction sensitivity to small latent perturbations.

    The loss compares predictions at ``z`` and ``z + epsilon * N(0, 1)``
    using mean squared error.  When ``num_samples > 1``, multiple random
    perturbations are averaged for a lower-variance estimate.

    This loss reads ``context.predictions`` and ``context.perturbed_predictions``.
    The ``perturbed_predictions`` field must be populated externally (e.g. by
    the MappingModel or trainer) before calling this loss.

    Args:
        epsilon: Standard deviation of the Gaussian perturbation noise.
        num_samples: Number of independent perturbations to average.
            Each sample doubles the cost of the stability term.
    """

    def __init__(self, epsilon: float = 0.01, num_samples: int = 1) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        self.epsilon = epsilon
        self.num_samples = num_samples

    def forward(self, context: TrainingContext) -> LossOutput:
        if context.perturbed_predictions is None:
            raise ValueError(
                "StabilityLoss requires perturbed_predictions in the training context. "
                "Ensure the model or trainer populates this field before computing loss."
            )
        original = _flatten_predictions(context.predictions)
        perturbed = _flatten_predictions(context.perturbed_predictions)
        if original.shape != perturbed.shape:
            original = original.expand_as(perturbed)
        sq_diff = F.mse_loss(original, perturbed, reduction="none")
        if sq_diff.ndim > 1:
            dim_to_sum = list(range(1, sq_diff.ndim))
            value = sq_diff.sum(dim=dim_to_sum).mean()
        else:
            value = sq_diff.mean()
        return LossOutput(
            total=value,
            components={"stability": value},
            metrics={"stability": value.detach().item()},
        )
