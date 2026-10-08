"""Pydantic configuration models for the loss system."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TaskLossConfig(BaseModel):
    """Configuration for the primary task loss.

    Attributes:
        type: Registry key or convenience alias — ``"classification"``,
            ``"regression"``, or a custom registered key.
        label_smoothing: Label smoothing factor for classification loss.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "classification"
    label_smoothing: float = 0.0


class LossConfig(BaseModel):
    """Configuration for the composite mapping loss.

    Controls which auxiliary losses are enabled and their weighting coefficients.

    Example::

        config = LossConfig(
            enable_stability=True,
            lambda_stability=0.1,
            smoothness_method="stochastic",
        )
    """

    model_config = ConfigDict(extra="forbid")

    task: TaskLossConfig = Field(default_factory=TaskLossConfig)
    lambda_stability: float = 0.1
    lambda_smoothness: float = 0.01
    lambda_alignment: float = 0.01
    stability_epsilon: float = 0.01
    stability_num_samples: int = 1
    smoothness_method: str = "stochastic"
    smoothness_projections: int = 4
    trainable_coefficients: bool = False
    enable_stability: bool = False
    enable_smoothness: bool = False
    enable_alignment: bool = False
