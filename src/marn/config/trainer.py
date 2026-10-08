"""Pydantic configuration model for the training loop."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainerConfig(BaseModel):
    """Configuration for ``MappingTrainer``.

    Covers optimizer, scheduler, gradient management, AMP, device selection,
    and reproducibility.

    Example::

        config = TrainerConfig(
            max_epochs=50,
            learning_rate=3e-4,
            gradient_clip_norm=1.0,
            amp_enabled=True
        )
    """

    model_config = ConfigDict(extra="forbid")

    max_epochs: int = 100
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str | None = None
    scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    gradient_clip_norm: float | None = None
    gradient_clip_value: float | None = None
    accumulation_steps: int = 1
    amp_enabled: bool = False
    device: str = "auto"
    seed: int | None = None
