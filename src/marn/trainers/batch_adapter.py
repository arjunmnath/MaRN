"""Configurable batch adapters for different dataloader output formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BatchAdapter(ABC):
    """Unpack dataloader batches into (model_inputs, targets).

    Different datasets and dataloaders produce batches in different formats.
    A batch adapter normalizes them into the ``(inputs, targets)`` tuple
    that the trainer loop expects.
    """

    @abstractmethod
    def unpack(self, batch: Any) -> tuple[tuple[Any, ...], Any]:
        """Return ``((input_arg1, ...), targets)`` from a dataloader batch.

        The first element is a tuple of positional arguments to pass to the
        model's ``forward`` method.  The second is the ground-truth targets
        for loss computation.
        """


class TupleBatchAdapter(BatchAdapter):
    """Unpack ``(inputs, targets)`` tuples — the standard PyTorch convention.

    Also handles ``(inputs, targets, *extra)`` by ignoring extra elements.
    """

    def unpack(self, batch: Any) -> tuple[tuple[Any, ...], Any]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError(
                f"TupleBatchAdapter expects a tuple/list of length >= 2, "
                f"got {type(batch).__name__} of length {len(batch) if isinstance(batch, (tuple, list)) else '?'}"
            )
        inputs, targets = batch[0], batch[1]
        return (inputs,), targets


class MappingBatchAdapter(BatchAdapter):
    """Unpack dict batches with configurable key names.

    Example::

        adapter = MappingBatchAdapter(input_key="image", target_key="label")
        # batch = {"image": tensor, "label": tensor}
        inputs, targets = adapter.unpack(batch)
    """

    def __init__(
        self,
        input_key: str = "input",
        target_key: str = "target",
    ) -> None:
        self.input_key = input_key
        self.target_key = target_key

    def unpack(self, batch: Any) -> tuple[tuple[Any, ...], Any]:
        if not isinstance(batch, dict):
            raise ValueError(
                f"MappingBatchAdapter expects a dict batch, got {type(batch).__name__}"
            )
        if self.input_key not in batch:
            raise KeyError(
                f"Batch dict missing input key {self.input_key!r}; "
                f"available keys: {sorted(batch.keys())}"
            )
        if self.target_key not in batch:
            raise KeyError(
                f"Batch dict missing target key {self.target_key!r}; "
                f"available keys: {sorted(batch.keys())}"
            )
        return (batch[self.input_key],), batch[self.target_key]
