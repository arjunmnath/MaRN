"""Generic typed registry for pluggable components."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Named registry for pluggable mapping-network components.

    Each registry maps string keys to classes.  Built-in implementations are
    pre-registered; users add custom types with :meth:`register`.

    Example::

        from marn.registry import MAPPER_REGISTRY

        MAPPER_REGISTRY.register("my_mapper", MyCustomMapper)
        cls = MAPPER_REGISTRY.get("my_mapper")
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, type[T]] = {}

    def register(self, key: str, cls: type[T]) -> None:
        """Register a component class under *key*.

        Raises:
            ValueError: If *key* is already registered.
        """
        if key in self._entries:
            raise ValueError(
                f"{self._name}: {key!r} is already registered as {self._entries[key].__name__}"
            )
        self._entries[key] = cls

    def get(self, key: str) -> type[T]:
        """Return the class registered under *key*.

        Raises:
            KeyError: If *key* has not been registered.
        """
        try:
            return self._entries[key]
        except KeyError:
            available = ", ".join(sorted(self._entries)) or "(none)"
            raise KeyError(
                f"{self._name}: {key!r} is not registered; available: {available}"
            ) from None

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def available(self) -> tuple[str, ...]:
        """Return all registered keys in sorted order."""
        return tuple(sorted(self._entries))

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, keys={self.available()})"


# ── Pre-populated registries ─────────────────────────────────────────────

from marn.mappers.base import BaseMapper  # noqa: E402
from marn.mappers.mlp_mapper import MLPMapper, ResidualMLPMapper  # noqa: E402
from marn.modulation.base import BaseModulation  # noqa: E402
from marn.modulation.additive import AdditiveModulation  # noqa: E402
from marn.modulation.affine import AffineModulation  # noqa: E402
from marn.modulation.low_rank import LowRankModulation  # noqa: E402
from marn.losses.base import BaseLoss  # noqa: E402
from marn.losses.task import (  # noqa: E402
    ClassificationLoss,
    RegressionLoss,
    TaskLoss,
)
from marn.losses.stability import StabilityLoss  # noqa: E402
from marn.losses.smoothness import SmoothnessLoss  # noqa: E402
from marn.losses.alignment import AlignmentLoss  # noqa: E402
from marn.generators.base import ParameterGenerator  # noqa: E402
from marn.generators.single_vector import SingleVectorGenerator  # noqa: E402
from marn.generators.layerwise import LayerwiseGenerator  # noqa: E402
from marn.generators.grouped import GroupedGenerator  # noqa: E402
from marn.generators.finetuning import FineTuningGenerator  # noqa: E402
from marn.generators.lrd import LRDGenerator  # noqa: E402

MAPPER_REGISTRY: Registry[BaseMapper] = Registry("MAPPER_REGISTRY")
MAPPER_REGISTRY.register("mlp", MLPMapper)
MAPPER_REGISTRY.register("residual_mlp", ResidualMLPMapper)

MODULATION_REGISTRY: Registry[BaseModulation] = Registry("MODULATION_REGISTRY")
MODULATION_REGISTRY.register("additive", AdditiveModulation)
MODULATION_REGISTRY.register("affine", AffineModulation)
MODULATION_REGISTRY.register("low_rank", LowRankModulation)

LOSS_REGISTRY: Registry[BaseLoss] = Registry("LOSS_REGISTRY")
LOSS_REGISTRY.register("classification", ClassificationLoss)
LOSS_REGISTRY.register("regression", RegressionLoss)
LOSS_REGISTRY.register("task", TaskLoss)
LOSS_REGISTRY.register("stability", StabilityLoss)
LOSS_REGISTRY.register("smoothness", SmoothnessLoss)
LOSS_REGISTRY.register("alignment", AlignmentLoss)

GENERATOR_REGISTRY: Registry[ParameterGenerator] = Registry("GENERATOR_REGISTRY")
GENERATOR_REGISTRY.register("single_vector", SingleVectorGenerator)
GENERATOR_REGISTRY.register("layerwise", LayerwiseGenerator)
GENERATOR_REGISTRY.register("grouped", GroupedGenerator)
GENERATOR_REGISTRY.register("finetuning", FineTuningGenerator)
GENERATOR_REGISTRY.register("lrd", LRDGenerator)
