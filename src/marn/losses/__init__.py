"""Loss components for mapping network training."""

from marn.losses.alignment import AlignmentLoss
from marn.losses.base import BaseLoss
from marn.losses.context import TrainingContext
from marn.losses.mapping_loss import MappingLoss
from marn.losses.outputs import LossOutput
from marn.losses.smoothness import SmoothnessLoss
from marn.losses.stability import StabilityLoss
from marn.losses.task import ClassificationLoss, RegressionLoss, TaskLoss

__all__ = [
    "AlignmentLoss",
    "BaseLoss",
    "ClassificationLoss",
    "LossOutput",
    "MappingLoss",
    "RegressionLoss",
    "SmoothnessLoss",
    "StabilityLoss",
    "TaskLoss",
    "TrainingContext",
]
