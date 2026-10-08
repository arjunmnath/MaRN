"""Fixed orthogonal MLP mapper variants."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from marn.mappers.base import BaseMapper
from marn.modulation.base import BaseModulation


def _orthogonal_matrix(rows: int, columns: int, *, expand_output_variance: bool = False) -> Tensor:
    matrix = torch.empty(rows, columns)
    nn.init.orthogonal_(matrix)
    if expand_output_variance:
        matrix = matrix * math.sqrt(max(rows, columns))
    return matrix


class MLPMapper(BaseMapper):
    """A fixed orthogonal MLP whose matrix weights are registered buffers."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (),
        modulation: BaseModulation | None = None,
        activation: Callable[[Tensor], Tensor] = functional.gelu,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(latent_dim, output_dim)
        dimensions = (latent_dim, *hidden_dims, output_dim)
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("All mapper dimensions must be positive")
        self.layer_count = len(dimensions) - 1
        self.modulation = modulation
        self.activation = activation
        self.output_activation = output_activation
        for index, (input_dim, layer_output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            self.register_buffer(
                f"weight_{index}",
                _orthogonal_matrix(
                    layer_output_dim,
                    input_dim,
                    expand_output_variance=index == self.layer_count - 1,
                ),
            )
            self.register_buffer(f"bias_{index}", torch.zeros(layer_output_dim))

    def forward(self, latent: Tensor) -> Tensor:
        self._validate_latent(latent)
        value = latent
        for index in range(self.layer_count):
            weight = self.get_buffer(f"weight_{index}")
            bias = self.get_buffer(f"bias_{index}")
            if self.modulation is not None:
                weight = self.modulation(weight, value)
            if index == 0:
                self._last_modulated_weight = weight
            value = functional.linear(value, weight, bias)
            if index < self.layer_count - 1:
                value = self.activation(value)
        if self.output_activation is not None:
            value = self.output_activation(value)
        return value


class ResidualMLPMapper(BaseMapper):
    """A fixed MLP with same-width residual blocks and orthogonal projections."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        depth: int = 2,
        modulation: BaseModulation | None = None,
        activation: Callable[[Tensor], Tensor] = functional.gelu,
    ) -> None:
        super().__init__(latent_dim, output_dim)
        if hidden_dim <= 0 or depth <= 0:
            raise ValueError("hidden_dim and depth must be positive")
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.modulation = modulation
        self.activation = activation
        self.register_buffer("input_weight", _orthogonal_matrix(hidden_dim, latent_dim))
        self.register_buffer("input_bias", torch.zeros(hidden_dim))
        for index in range(depth):
            self.register_buffer(
                f"block_weight_{index}", _orthogonal_matrix(hidden_dim, hidden_dim)
            )
            self.register_buffer(f"block_bias_{index}", torch.zeros(hidden_dim))
        self.register_buffer(
            "output_weight",
            _orthogonal_matrix(output_dim, hidden_dim, expand_output_variance=True),
        )
        self.register_buffer("output_bias", torch.zeros(output_dim))

    def _linear(self, value: Tensor, weight_name: str, bias_name: str) -> Tensor:
        weight = self.get_buffer(weight_name)
        if self.modulation is not None:
            weight = self.modulation(weight, value)
        if weight_name == "input_weight":
            self._last_modulated_weight = weight
        return functional.linear(value, weight, self.get_buffer(bias_name))

    def forward(self, latent: Tensor) -> Tensor:
        self._validate_latent(latent)
        value = self.activation(self._linear(latent, "input_weight", "input_bias"))
        for index in range(self.depth):
            residual = self._linear(value, f"block_weight_{index}", f"block_bias_{index}")
            value = value + self.activation(residual)
        return self._linear(value, "output_weight", "output_bias")
