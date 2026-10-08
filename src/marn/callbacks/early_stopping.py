"""Early stopping callback."""

from __future__ import annotations

import math
from typing import Any

from marn.callbacks.base import Callback


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Name of the metric to monitor (e.g. ``"val_loss"``).
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``"min"`` to minimize the metric, ``"max"`` to maximize it.

    Example::

        trainer = MappingTrainer(
            ...,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float = math.inf if mode == "min" else -math.inf
        self._wait: int = 0

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self._best - self.min_delta
        return current > self._best + self.min_delta

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            return

        if self._is_improvement(current):
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                trainer.should_stop = True
