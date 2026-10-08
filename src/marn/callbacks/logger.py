"""Structured metric logging callback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from marn.callbacks.base import Callback

if TYPE_CHECKING:
    from marn.losses.outputs import LossOutput

logger = logging.getLogger("marn.trainer")


class MetricLogger(Callback):
    """Log training and validation metrics using the standard ``logging`` module.

    Args:
        log_every_n_batches: Log batch metrics every N batches.

    Example::

        trainer = MappingTrainer(
            ...,
            callbacks=[MetricLogger(log_every_n_batches=50)],
        )
    """

    def __init__(self, log_every_n_batches: int = 10) -> None:
        self.log_every_n_batches = log_every_n_batches

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        logger.info("Epoch %d/%d started", epoch + 1, trainer.config.max_epochs)

    def on_batch_end(self, trainer: Any, batch_idx: int, loss_output: LossOutput) -> None:
        if (batch_idx + 1) % self.log_every_n_batches == 0:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in loss_output.metrics.items())
            logger.info("  batch %d: %s", batch_idx + 1, metrics_str)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("Epoch %d/%d: %s", epoch + 1, trainer.config.max_epochs, metrics_str)

    def on_validation_end(self, trainer: Any, metrics: dict[str, float]) -> None:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("  validation: %s", metrics_str)
