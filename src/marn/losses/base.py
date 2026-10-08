"""Abstract base for loss components."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from marn.losses.context import TrainingContext
from marn.losses.outputs import LossOutput


class BaseLoss(nn.Module, ABC):
    """Contract for individual loss components.

    Every loss receives a ``TrainingContext`` and returns a ``LossOutput``.
    Components should read only the context fields they need and must not
    mutate the context.
    """

    @abstractmethod
    def forward(self, context: TrainingContext) -> LossOutput:
        """Compute this loss component and return structured output."""
