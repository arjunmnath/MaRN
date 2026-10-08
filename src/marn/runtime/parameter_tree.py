"""A named, validated container for generated model parameters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Callable

from torch import Tensor


class ParameterTree(Mapping[str, Tensor]):
    """An immutable mapping from fully qualified parameter names to tensors.

    The mapping itself is immutable, while its tensor values retain normal
    PyTorch autograd semantics. Keeping names attached to generated tensors
    avoids positional split/reshape contracts leaking through the package.
    """

    def __init__(self, parameters: Mapping[str, Tensor]) -> None:
        copied = dict(parameters)
        self._validate(copied)
        self._parameters = MappingProxyType(copied)

    @staticmethod
    def _validate(parameters: Mapping[str, Tensor]) -> None:
        for name, tensor in parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Parameter names must be non-empty strings")
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Parameter {name!r} must be a torch.Tensor")

    def __getitem__(self, name: str) -> Tensor:
        return self._parameters[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)

    def __repr__(self) -> str:
        shapes = {name: tuple(tensor.shape) for name, tensor in self.items()}
        return f"ParameterTree({shapes!r})"

    def to_dict(self) -> dict[str, Tensor]:
        """Return a shallow mutable copy suitable for ``functional_call``."""

        return dict(self._parameters)

    def map(self, function: Callable[[Tensor], Tensor]) -> ParameterTree:
        """Apply ``function`` to every tensor without discarding names."""

        return ParameterTree({name: function(tensor) for name, tensor in self.items()})
