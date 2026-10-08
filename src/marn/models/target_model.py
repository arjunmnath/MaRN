"""Frozen, stateless execution wrapper for an arbitrary PyTorch model."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn
from torch.nn.utils import parametrize

from marn.runtime.functional import functional_forward
from marn.runtime.parameter_spec import ParameterSpec
from marn.runtime.parameter_tree import ParameterTree


class UnsupportedTargetModelError(ValueError):
    """Raised when a target cannot yet be represented without changing semantics."""


def _shared_parameter_groups(module: nn.Module) -> tuple[tuple[str, ...], ...]:
    names_by_identity: dict[int, list[str]] = defaultdict(list)
    for name, parameter in module.named_parameters(remove_duplicate=False):
        names_by_identity[id(parameter)].append(name)
    return tuple(tuple(names) for names in names_by_identity.values() if len(names) > 1)


def _parametrized_module_names(module: nn.Module) -> tuple[str, ...]:
    names: list[str] = []
    for name, child in module.named_modules():
        if parametrize.is_parametrized(child):
            names.append(name or "<root>")
    return tuple(names)


class TargetModel(nn.Module):
    """Freeze and execute a target model with externally generated parameters.

    Target parameters are never assigned or optimized. Every forward uses
    ``torch.func.functional_call`` and cloned buffers, so stateful layers such
    as BatchNorm cannot update the wrapped target's persistent state.

    Tied parameters and PyTorch parametrizations are rejected for now because
    a generator must preserve their aliasing/transformation contracts. Failing
    explicitly is safer than silently generating semantically different models.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._validate_supported(module)
        module.requires_grad_(False)
        self.module = module
        self.parameter_spec = ParameterSpec.from_module(module)

    @staticmethod
    def _validate_supported(module: nn.Module) -> None:
        shared_groups = _shared_parameter_groups(module)
        if shared_groups:
            formatted = ", ".join(" = ".join(group) for group in shared_groups)
            raise UnsupportedTargetModelError(
                "Tied/shared target parameters are not supported yet: " + formatted
            )

        parametrized_names = _parametrized_module_names(module)
        if parametrized_names:
            raise UnsupportedTargetModelError(
                "Parametrized target modules are not supported yet: "
                + ", ".join(parametrized_names)
            )

    def forward(
        self,
        parameters: ParameterTree | Mapping[str, Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run a stateless forward using the supplied complete parameter tree."""

        self.parameter_spec.validate_tree(parameters)
        return functional_forward(self.module, parameters, args, kwargs)
