"""Observer callback base class for training lifecycle hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marn.losses.outputs import LossOutput


class Callback:
    """Base class for training lifecycle callbacks.

    All hooks are no-ops by default — subclass and override the ones you need.
    Callbacks receive the trainer instance so they can inspect or modify state
    (e.g. set ``trainer.should_stop = True`` for early termination).

    Hook execution order within each event:

    1. ``on_fit_start``
    2. For each epoch:
       a. ``on_epoch_start``
       b. For each batch: ``on_batch_start`` → ``on_batch_end``
       c. ``on_validation_start`` → ``on_validation_end``
       d. ``on_epoch_end``
    3. ``on_fit_end``
    """

    def on_fit_start(self, trainer: Any) -> None:
        """Called once at the beginning of :meth:`fit`."""

    def on_fit_end(self, trainer: Any) -> None:
        """Called once at the end of :meth:`fit`."""

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch with aggregated metrics."""

    def on_batch_start(self, trainer: Any, batch_idx: int) -> None:
        """Called before processing each batch."""

    def on_batch_end(self, trainer: Any, batch_idx: int, loss_output: LossOutput) -> None:
        """Called after processing each batch with the loss output."""

    def on_validation_start(self, trainer: Any) -> None:
        """Called before the validation loop."""

    def on_validation_end(self, trainer: Any, metrics: dict[str, float]) -> None:
        """Called after the validation loop with validation metrics."""
