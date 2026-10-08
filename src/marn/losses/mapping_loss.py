"""Composite mapping loss with optional trainable coefficients.

Combines task, stability, smoothness, and alignment losses with the paper's
weighting scheme:

    L = L_task + λ_st·L_stability + λ_sm·L_smoothness + λ_al·L_alignment

By default, all three regularization components are enabled with sensible
default implementations.  Pass ``None`` explicitly to disable a component.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from marn.losses.base import BaseLoss
from marn.losses.context import TrainingContext
from marn.losses.outputs import LossOutput

# Sentinel value indicating "use the built-in default implementation".
_AUTO: Literal["auto"] = "auto"


def _default_stability_loss() -> BaseLoss:
    """Create the default stability loss with conservative parameters."""
    from marn.losses.stability import StabilityLoss

    return StabilityLoss(epsilon=0.01, num_samples=1)


def _default_smoothness_loss() -> BaseLoss:
    """Create the default smoothness loss with stochastic Jacobian estimation."""
    from marn.losses.smoothness import SmoothnessLoss

    return SmoothnessLoss(method="stochastic", num_projections=1)


def _default_alignment_loss() -> BaseLoss:
    """Create the default alignment loss."""
    from marn.losses.alignment import AlignmentLoss

    return AlignmentLoss()


class MappingLoss(nn.Module):
    """Composite loss implementing the paper's multi-component mapping loss.

    Combines a required task loss with stability, smoothness, and alignment
    components.  Each component is weighted by a corresponding lambda
    coefficient.

    **Default behaviour** (changed in v0.3): when a regularization component
    is not specified, a sensible default implementation is used automatically.
    Pass ``None`` explicitly to disable a component.

    When ``trainable_coefficients=True``, lambdas are stored as raw
    ``nn.Parameter`` values and passed through ``softplus`` at compute time
    to guarantee non-negativity.  This matches the paper's description of
    trainable regularization coefficients.

    When ``trainable_coefficients=False`` (default), lambdas are plain floats
    and are not included in the model's parameter groups.

    Args:
        task_loss: Required task loss component.
        stability_loss: Stability loss component.  ``"auto"`` (default)
            uses ``StabilityLoss(epsilon=0.01)``.  Pass ``None`` to disable.
        smoothness_loss: Smoothness loss component.  ``"auto"`` (default)
            uses ``SmoothnessLoss(method="stochastic", num_projections=1)``.
            Pass ``None`` to disable.
        alignment_loss: Alignment loss component.  ``"auto"`` (default)
            uses ``AlignmentLoss()``.  Pass ``None`` to disable.
        lambda_stability: Weight for the stability component.
        lambda_smoothness: Weight for the smoothness component.
        lambda_alignment: Weight for the alignment component.
        trainable_coefficients: If ``True``, lambda values become trainable
            parameters (passed through softplus for non-negativity).

    Example::

        # All three regularization losses active by default:
        loss_fn = MappingLoss(task_loss=ClassificationLoss())

        # Disable smoothness, keep others at default:
        loss_fn = MappingLoss(
            task_loss=ClassificationLoss(),
            smoothness_loss=None,
        )

        # Custom stability, default smoothness and alignment:
        loss_fn = MappingLoss(
            task_loss=ClassificationLoss(),
            stability_loss=StabilityLoss(epsilon=0.05, num_samples=3),
        )
    """

    def __init__(
        self,
        task_loss: BaseLoss,
        stability_loss: BaseLoss | None | Literal["auto"] = _AUTO,
        smoothness_loss: BaseLoss | None | Literal["auto"] = _AUTO,
        alignment_loss: BaseLoss | None | Literal["auto"] = _AUTO,
        lambda_stability: float = 0.1,
        lambda_smoothness: float = 0.01,
        lambda_alignment: float = 0.01,
        *,
        trainable_coefficients: bool = False,
    ) -> None:
        super().__init__()
        self.task_loss = task_loss

        # Resolve "auto" sentinels to default implementations
        self.stability_loss: BaseLoss | None = (
            _default_stability_loss()
            if stability_loss is _AUTO or stability_loss == "auto"
            else stability_loss
        )
        self.smoothness_loss: BaseLoss | None = (
            _default_smoothness_loss()
            if smoothness_loss is _AUTO or smoothness_loss == "auto"
            else smoothness_loss
        )
        self.alignment_loss: BaseLoss | None = (
            _default_alignment_loss()
            if alignment_loss is _AUTO or alignment_loss == "auto"
            else alignment_loss
        )

        self.trainable_coefficients = trainable_coefficients

        if trainable_coefficients:
            # Store raw values; softplus is applied at compute time
            # Initialize with inverse-softplus of the desired lambda so
            # softplus(raw) ≈ lambda at the start of training.
            self._raw_lambda_stability = nn.Parameter(
                torch.tensor(_inverse_softplus(lambda_stability))
            )
            self._raw_lambda_smoothness = nn.Parameter(
                torch.tensor(_inverse_softplus(lambda_smoothness))
            )
            self._raw_lambda_alignment = nn.Parameter(
                torch.tensor(_inverse_softplus(lambda_alignment))
            )
        else:
            self._lambda_stability = lambda_stability
            self._lambda_smoothness = lambda_smoothness
            self._lambda_alignment = lambda_alignment

    @property
    def lambda_stability(self) -> Tensor | float:
        if self.trainable_coefficients:
            return F.softplus(self._raw_lambda_stability)
        return self._lambda_stability

    @property
    def lambda_smoothness(self) -> Tensor | float:
        if self.trainable_coefficients:
            return F.softplus(self._raw_lambda_smoothness)
        return self._lambda_smoothness

    @property
    def lambda_alignment(self) -> Tensor | float:
        if self.trainable_coefficients:
            return F.softplus(self._raw_lambda_alignment)
        return self._lambda_alignment

    def forward(self, context: TrainingContext) -> LossOutput:
        """Compute the composite loss and return structured output."""
        task_output = self.task_loss(context)
        total = task_output.total
        components: dict[str, Tensor] = dict(task_output.components)
        metrics: dict[str, float] = dict(task_output.metrics)

        if self.stability_loss is not None:
            stability_output = self.stability_loss(context)
            weighted = self.lambda_stability * stability_output.total
            total = total + weighted
            components.update(stability_output.components)
            metrics.update(stability_output.metrics)
            metrics["lambda_stability"] = (
                self.lambda_stability.detach().item()
                if isinstance(self.lambda_stability, Tensor)
                else self.lambda_stability
            )

        if self.smoothness_loss is not None:
            smoothness_output = self.smoothness_loss(context)
            weighted = self.lambda_smoothness * smoothness_output.total
            total = total + weighted
            components.update(smoothness_output.components)
            metrics.update(smoothness_output.metrics)
            metrics["lambda_smoothness"] = (
                self.lambda_smoothness.detach().item()
                if isinstance(self.lambda_smoothness, Tensor)
                else self.lambda_smoothness
            )

        if self.alignment_loss is not None:
            alignment_output = self.alignment_loss(context)
            weighted = self.lambda_alignment * alignment_output.total
            total = total + weighted
            components.update(alignment_output.components)
            metrics.update(alignment_output.metrics)
            metrics["lambda_alignment"] = (
                self.lambda_alignment.detach().item()
                if isinstance(self.lambda_alignment, Tensor)
                else self.lambda_alignment
            )

        return LossOutput(total=total, components=components, metrics=metrics)


def _inverse_softplus(value: float) -> float:
    """Compute x such that softplus(x) ≈ value."""
    if value <= 0:
        raise ValueError("Cannot compute inverse softplus of non-positive value")
    # softplus(x) = log(1 + exp(x)), so x = log(exp(value) - 1)
    import math

    if value > 20:
        return value  # softplus(x) ≈ x for large x
    return math.log(math.exp(value) - 1)
