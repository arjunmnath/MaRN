"""Task losses: generic callable, classification, and regression."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from marn.losses.base import BaseLoss
from marn.losses.context import TrainingContext
from marn.losses.outputs import LossOutput


class TaskLoss(BaseLoss):
    """Wrap an arbitrary ``(predictions, targets) -> scalar`` callable.

    Example::

        loss = TaskLoss(lambda pred, tgt: F.l1_loss(pred, tgt))
    """

    def __init__(self, loss_fn: Callable[[Any, Any], Tensor]) -> None:
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, context: TrainingContext) -> LossOutput:
        value = self.loss_fn(context.predictions, context.targets)
        return LossOutput(
            total=value,
            components={"task": value},
            metrics={"task": value.detach().item()},
        )


class ClassificationLoss(BaseLoss):
    """Cross-entropy task loss for classification targets.

    Expects ``context.predictions`` to be raw logits of shape ``(N, C)`` and
    ``context.targets`` to be integer class indices of shape ``(N,)``.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, context: TrainingContext) -> LossOutput:
        value = F.cross_entropy(
            context.predictions,
            context.targets,
            label_smoothing=self.label_smoothing,
        )
        # Compute accuracy for metrics
        with torch.no_grad():
            predicted = context.predictions.argmax(dim=-1)
            correct = (predicted == context.targets).float().mean().item()
        return LossOutput(
            total=value,
            components={"task": value},
            metrics={"task": value.detach().item(), "accuracy": correct},
        )


class RegressionLoss(BaseLoss):
    """Mean squared error task loss for regression targets.

    Expects ``context.predictions`` and ``context.targets`` to be tensors
    of the same shape.
    """

    def forward(self, context: TrainingContext) -> LossOutput:
        value = F.mse_loss(context.predictions, context.targets)
        return LossOutput(
            total=value,
            components={"task": value},
            metrics={"task": value.detach().item()},
        )
