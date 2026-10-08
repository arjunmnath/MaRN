"""A compiled description of a target model's parameter layout."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import overload

import torch
from torch import Tensor, nn

from marn.runtime.parameter_tree import ParameterTree


@dataclass(frozen=True, slots=True)
class ParameterEntry:
    """Shape and flat-vector location for one named parameter."""

    name: str
    shape: torch.Size
    numel: int
    start: int
    stop: int

    @property
    def slice_range(self) -> slice:
        return slice(self.start, self.stop)


class ParameterSpec(Sequence[ParameterEntry]):
    """Compile parameter shapes and offsets once, then reuse them cheaply."""

    def __init__(self, entries: Iterable[ParameterEntry]) -> None:
        self._entries = tuple(entries)
        self._by_name = {entry.name: entry for entry in self._entries}
        self._validate()

    @classmethod
    def from_module(cls, module: nn.Module) -> ParameterSpec:
        """Build a spec from trainable and frozen parameters in module order."""

        return cls.from_named_tensors(module.named_parameters())

    @classmethod
    def from_named_tensors(cls, named_tensors: Iterable[tuple[str, Tensor]]) -> ParameterSpec:
        entries: list[ParameterEntry] = []
        offset = 0
        for name, tensor in named_tensors:
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Parameter {name!r} must be a torch.Tensor")
            numel = tensor.numel()
            entries.append(
                ParameterEntry(
                    name=name,
                    shape=torch.Size(tensor.shape),
                    numel=numel,
                    start=offset,
                    stop=offset + numel,
                )
            )
            offset += numel
        return cls(entries)

    def _validate(self) -> None:
        if len(self._by_name) != len(self._entries):
            raise ValueError("ParameterSpec contains duplicate names")

        expected_start = 0
        for entry in self._entries:
            if not entry.name:
                raise ValueError("Parameter names must be non-empty")
            if entry.numel != prod(entry.shape):
                raise ValueError(f"Invalid numel for parameter {entry.name!r}")
            if entry.start != expected_start or entry.stop != entry.start + entry.numel:
                raise ValueError(f"Non-contiguous slice for parameter {entry.name!r}")
            expected_start = entry.stop

    @property
    def total_numel(self) -> int:
        return self._entries[-1].stop if self._entries else 0

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self._entries)

    def entry(self, name: str) -> ParameterEntry:
        return self._by_name[name]

    def unflatten(self, vector: Tensor) -> ParameterTree:
        """Create named tensor views over a one-dimensional generated vector."""

        if vector.ndim != 1:
            raise ValueError(f"Expected a 1D vector, got shape {tuple(vector.shape)}")
        if vector.numel() != self.total_numel:
            raise ValueError(f"Expected {self.total_numel} values, got {vector.numel()}")

        return ParameterTree(
            {entry.name: vector[entry.slice_range].view(entry.shape) for entry in self._entries}
        )

    def flatten(self, tree: Mapping[str, Tensor]) -> Tensor:
        """Validate and flatten a named parameter tree in compiled order."""

        actual_names = set(tree)
        expected_names = set(self.names)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise ValueError(
                f"Parameter names do not match spec; missing={missing}, unexpected={unexpected}"
            )

        flattened: list[Tensor] = []
        for entry in self._entries:
            tensor = tree[entry.name]
            if torch.Size(tensor.shape) != entry.shape:
                raise ValueError(
                    f"Expected shape {tuple(entry.shape)} for {entry.name!r}, "
                    f"got {tuple(tensor.shape)}"
                )
            flattened.append(tensor.reshape(-1))

        if not flattened:
            return torch.empty(0)
        return torch.cat(flattened)

    def validate_tree(self, tree: Mapping[str, Tensor]) -> None:
        """Raise ``ValueError`` if names or shapes differ from this spec."""

        actual_names = set(tree)
        expected_names = set(self.names)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise ValueError(
                f"Parameter names do not match spec; missing={missing}, unexpected={unexpected}"
            )
        for entry in self._entries:
            actual_shape = torch.Size(tree[entry.name].shape)
            if actual_shape != entry.shape:
                raise ValueError(
                    f"Expected shape {tuple(entry.shape)} for {entry.name!r}, "
                    f"got {tuple(actual_shape)}"
                )

    @overload
    def __getitem__(self, index: int) -> ParameterEntry: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ParameterEntry]: ...

    def __getitem__(self, index: int | slice) -> ParameterEntry | Sequence[ParameterEntry]:
        return self._entries[index]

    def __iter__(self) -> Iterator[ParameterEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
