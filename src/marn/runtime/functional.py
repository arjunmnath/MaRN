"""Stateless execution helpers for generated parameter trees."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn
from torch.func import functional_call


def clone_module_buffers(module: nn.Module) -> dict[str, Tensor]:
    """Clone all target buffers so stateful forwards cannot mutate the target.

    BatchNorm and similar modules may update buffers in-place while in training
    mode. Passing per-call clones preserves their normal within-forward behavior
    and discards those updates after the call.
    """

    return {name: buffer.clone() for name, buffer in module.named_buffers()}


def functional_forward(
    module: nn.Module,
    parameters: Mapping[str, Tensor],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Run ``module`` with generated parameters and isolated target buffers."""

    parameter_dict = dict(parameters)
    buffer_dict = clone_module_buffers(module)
    keyword_args = dict(kwargs) if kwargs is not None else {}
    return functional_call(
        module,
        (parameter_dict, buffer_dict),
        args,
        keyword_args,
        tie_weights=True,
        strict=True,
    )
