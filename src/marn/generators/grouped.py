"""Explicit grouped parameter generation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from marn.generators.base import (
    MapperFactory,
    ParameterGenerator,
    default_mapper_factory,
)
from marn.generators.layer_generator import LayerGenerator
from marn.runtime import ParameterEntry, ParameterSpec, ParameterTree


def _subspec(entries: Sequence[ParameterEntry]) -> ParameterSpec:
    offset = 0
    local_entries: list[ParameterEntry] = []
    for entry in entries:
        local_entries.append(
            ParameterEntry(entry.name, entry.shape, entry.numel, offset, offset + entry.numel)
        )
        offset += entry.numel
    return ParameterSpec(local_entries)


class GroupedGenerator(ParameterGenerator):
    """Generate target parameters from explicit, independently mapped groups."""

    def __init__(
        self,
        parameter_spec: ParameterSpec,
        groups: Mapping[str, Sequence[str]],
        latent_dim: int | Mapping[str, int],
        *,
        mapper_factory: MapperFactory = default_mapper_factory,
    ) -> None:
        super().__init__(parameter_spec)
        self.group_names = tuple(groups)
        if not self.group_names:
            raise ValueError("At least one parameter group is required")
        flattened_names = [name for names in groups.values() for name in names]
        if len(flattened_names) != len(set(flattened_names)):
            raise ValueError("Parameter groups contain duplicate names")
        if set(flattened_names) != set(parameter_spec.names):
            missing = sorted(set(parameter_spec.names) - set(flattened_names))
            unexpected = sorted(set(flattened_names) - set(parameter_spec.names))
            raise ValueError(
                f"Groups must cover the spec exactly; missing={missing}, unexpected={unexpected}"
            )

        self.group_specs: list[ParameterSpec] = []
        self.layers = nn.ModuleList()
        for group_name, parameter_names in groups.items():
            entries = [parameter_spec.entry(name) for name in parameter_names]
            group_spec = _subspec(entries)
            group_latent_dim = (
                latent_dim[group_name] if isinstance(latent_dim, Mapping) else latent_dim
            )
            self.group_specs.append(group_spec)
            self.layers.append(
                LayerGenerator(
                    group_latent_dim,
                    group_spec.total_numel,
                    mapper_factory(group_latent_dim, group_spec.total_numel),
                )
            )

    def generate_parameters(self, latents: dict[str, Tensor] | None = None) -> ParameterTree:
        generated: dict[str, Tensor] = {}
        for i, (name, layer, group_spec) in enumerate(
            zip(self.group_names, self.layers, self.group_specs, strict=True)
        ):
            latent = latents.get(name) if latents is not None else None
            flat_params = layer(latent)
            scale_name = f"parameter_scale_{i}"
            if hasattr(self, scale_name):
                flat_params = flat_params * getattr(self, scale_name)
            generated.update(group_spec.unflatten(flat_params).to_dict())
        tree = ParameterTree(generated)
        self.parameter_spec.validate_tree(tree)
        return tree

    def initialize_parameter_scaling(self, target_model: nn.Module) -> None:
        for i, group_spec in enumerate(self.group_specs):
            scales = []
            for entry in group_spec:
                try:
                    parts = entry.name.split(".")
                    curr: Any = target_model
                    for part in parts:
                        curr = getattr(curr, part)
                    assert isinstance(curr, Tensor)
                    std = curr.std(unbiased=False).item()
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
                layer = self.layers[i]
                assert isinstance(layer, LayerGenerator)
                dtype: torch.dtype | None = layer.latent.dtype
                scales.append(torch.full((entry.numel,), scale, dtype=dtype))

            scale_vector = torch.cat(scales)
            self.register_buffer(f"parameter_scale_{i}", scale_vector)

    def modulated_mapper_weights(self) -> dict[str, Tensor]:
        weights: dict[str, Tensor] = {}
        for name, layer in zip(self.group_names, self.layers, strict=True):
            if hasattr(layer.mapper, "_last_modulated_weight"):
                w = layer.mapper._last_modulated_weight
                if isinstance(w, Tensor):
                    weights[name] = w
        return weights

    def named_latent_vectors(self) -> Iterator[tuple[str, Tensor]]:
        for name, layer in zip(self.group_names, self.layers, strict=True):
            assert isinstance(layer, LayerGenerator)
            yield name, layer.latent
