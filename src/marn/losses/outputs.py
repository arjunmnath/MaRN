"""Typed containers for loss computation results."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class LossOutput:
    """Structured output from a loss computation.

    Attributes:
        total: Weighted scalar loss for ``backward()``.
        components: Named component scalars (e.g. ``{"task": ..., "stability": ...}``).
            Each value is a weighted, autograd-live scalar.
        metrics: Unweighted scalar floats for logging and diagnostics.  These
            are detached from the graph and safe to serialize.
    """

    total: Tensor
    components: dict[str, Tensor] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
