"""Pre-trained model parameter fine-tuning generator."""

from __future__ import annotations

from collections.abc import Iterator
import math

import torch
from torch import Tensor, nn

from marn.generators.base import (
    MapperFactory,
    ParameterGenerator,
    default_mapper_factory,
)
from marn.generators.layer_generator import LayerGenerator
from marn.runtime import ParameterSpec, ParameterTree


class FineTuningGenerator(ParameterGenerator):
    r"""Generate fine-tuning modulation vectors added to pre-trained weights.

    This generator implements the paper's fine-tuning method (Section 2.2.6).
    Instead of generating the complete weight tensors from scratch, it generates
    smaller modulation vectors $o$ which are then tiled (block-wise repeated),
    scaled by $\alpha$, and added to the original pre-trained frozen weights:

        W_FT = W_pretrained + \alpha \cdot o_expanded

    This allows fine-tuning large pre-trained networks with a extremely low number
    of trainable parameter variables by dividing the target parameters into blocks
    of size $L$, each of which is modulated by a single generated element.

    Args:
        parameter_spec: The target model's parameter specification.
        latent_dim: Dimensionality of each trainable latent vector.
        pretrained_parameters: Dict mapping parameter names to their original
            pretrained PyTorch Tensors.
        L: Block size (default: 250). Each generated element modulates L consecutive
            flattened parameter values.
        alpha: Scaling factor for the applied modulation.
        mapper_factory: Callable ``(latent_dim, output_dim) -> BaseMapper``.
            Defaults to the orthogonal MLP mapper with additive modulation.
    """

    def __init__(
        self,
        parameter_spec: ParameterSpec,
        latent_dim: int,
        pretrained_parameters: dict[str, Tensor],
        L: int = 250,
        alpha: float = 0.01,
        *,
        mapper_factory: MapperFactory = default_mapper_factory,
    ) -> None:
        super().__init__(parameter_spec)
        if L <= 0:
            raise ValueError("Block size L must be positive")
        self.latent_dim = latent_dim
        self.L = L
        self.alpha = alpha

        # Register pre-trained weights as buffers to automatically manage device movements
        for name, tensor in pretrained_parameters.items():
            self.register_buffer(f"pretrained_{name.replace('.', '_')}", tensor.clone().detach())

        # Group layer-wise
        from marn.generators.layerwise import layerwise_groups

        self.groups = layerwise_groups(parameter_spec)
        self.group_names = tuple(self.groups)

        self.group_specs = []
        self.layers = nn.ModuleList()

        for group_name, parameter_names in self.groups.items():
            entries = [parameter_spec.entry(name) for name in parameter_names]
            from marn.generators.grouped import _subspec

            group_spec = _subspec(entries)
            self.group_specs.append(group_spec)

            # Determine the generated modulation vector dimension
            output_dim = math.ceil(group_spec.total_numel / L)
            self.layers.append(
                LayerGenerator(
                    latent_dim,
                    output_dim,
                    mapper_factory(latent_dim, output_dim),
                )
            )

    def generate_parameters(self, latents: dict[str, Tensor] | None = None) -> ParameterTree:
        generated: dict[str, Tensor] = {}
        for name, layer, group_spec in zip(
            self.group_names, self.layers, self.group_specs, strict=True
        ):
            latent = latents.get(name) if latents is not None else None
            o = layer(latent)
            o_expanded = torch.repeat_interleave(o, self.L)
            o_expanded = o_expanded[: group_spec.total_numel]

            pretrained_flat_parts = []
            for param_name in group_spec.names:
                pretrained_tensor = self.get_buffer(f"pretrained_{param_name.replace('.', '_')}")
                pretrained_flat_parts.append(pretrained_tensor.reshape(-1))
            pretrained_flat = torch.cat(pretrained_flat_parts)

            modulated_flat = pretrained_flat + self.alpha * o_expanded

            generated.update(group_spec.unflatten(modulated_flat).to_dict())

        tree = ParameterTree(generated)
        self.parameter_spec.validate_tree(tree)
        return tree

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
