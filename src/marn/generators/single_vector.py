"""Paper-compatible single latent vector training (SLVT)."""

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn

from marn.generators.base import (
    MapperFactory,
    ParameterGenerator,
    default_mapper_factory,
)
from marn.generators.layer_generator import LayerGenerator
from marn.runtime import ParameterSpec, ParameterTree


class SingleVectorGenerator(ParameterGenerator):
    """Generate every target parameter from one latent vector."""

    def __init__(
        self,
        parameter_spec: ParameterSpec,
        latent_dim: int,
        *,
        mapper_factory: MapperFactory = default_mapper_factory,
        max_projection_elements: int = 100_000_000,
        allow_large: bool = False,
    ) -> None:
        super().__init__(parameter_spec)
        projection_elements = latent_dim * parameter_spec.total_numel
        if projection_elements > max_projection_elements and not allow_large:
            raise MemoryError(
                "SLVT fixed projection would contain "
                f"{projection_elements:,} elements; use layerwise/grouped generation or "
                "set allow_large=True explicitly"
            )
        mapper = mapper_factory(latent_dim, parameter_spec.total_numel)
        self.layer = LayerGenerator(latent_dim, parameter_spec.total_numel, mapper)

    def generate_parameters(self, latents: dict[str, Tensor] | None = None) -> ParameterTree:
        latent = latents.get("all") if latents is not None else None
        flat_params = self.layer(latent)
        if hasattr(self, "parameter_scale"):
            flat_params = flat_params * self.parameter_scale
        return self.parameter_spec.unflatten(flat_params)

    def initialize_parameter_scaling(self, target_model: nn.Module) -> None:
        scales = []
        for entry in self.parameter_spec:
            try:
                parts = entry.name.split(".")
                curr: Any = target_model
                for part in parts:
                    curr = getattr(curr, part)
                assert isinstance(curr, Tensor)
                std = curr.std().item()
                if std > 1e-5:
                    scale = std
                else:
                    if entry.name.endswith(".bias"):
                        weight_name = entry.name[:-5] + ".weight"
                        try:
                            w_parts = weight_name.split(".")
                            w_curr: Any = target_model
                            for w_part in w_parts:
                                w_curr = getattr(w_curr, w_part)
                            scale = w_curr.std().item()
                            if scale <= 1e-5:
                                scale = 1.0
                        except Exception:
                            scale = 1.0
                    else:
                        scale = 1.0
            except Exception:
                scale = 1.0
            scales.append(torch.full((entry.numel,), scale, dtype=self.layer.latent.dtype))

        scale_vector = torch.cat(scales)
        self.register_buffer("parameter_scale", scale_vector)

    def modulated_mapper_weights(self) -> dict[str, Tensor]:
        weights = {}
        if hasattr(self.layer.mapper, "_last_modulated_weight"):
            w = self.layer.mapper._last_modulated_weight
            if w is not None:
                weights["all"] = w
        return weights

    def named_latent_vectors(self) -> Iterator[tuple[str, Tensor]]:
        yield "all", self.layer.latent
