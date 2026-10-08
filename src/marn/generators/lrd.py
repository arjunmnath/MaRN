"""Low-Rank Decomposition (LRD) target parameter generator."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from torch import Tensor, nn

from marn.generators.base import (
    MapperFactory,
    ParameterGenerator,
    default_mapper_factory,
)
from marn.generators.layer_generator import LayerGenerator
from marn.runtime import ParameterSpec, ParameterTree


class LRDGenerator(ParameterGenerator):
    r"""Generate target parameters using target-side Low-Rank Decomposition.

    This generator implements target-side Low-Rank Decomposition (LRD) (Section 2.2.5).
    For any target parameter that is a 2D weight matrix $W \\in \\mathbb{R}^{m \\times n}$
    where the decomposition rank satisfies $r < \\min(m, n)$, we represent $W$ as the
    product of two lower-rank factor matrices:

        W = U V^T

    where $U \\in \\mathbb{R}^{m \\times r}$ and $V \\in \\mathbb{R}^{n \\times r}$.
    Instead of generating $m \\cdot n$ elements, the mapper is only required to generate
    $r(m + n)$ values, reducing generator memory footprint and parameter count for large
    fully-connected layers.

    Biases and non-2D parameters are generated in full size without decomposition.

    Args:
        parameter_spec: The target model's parameter specification.
        latent_dim: Dimensionality of each trainable latent vector.
        rank: Decomposition rank. Used for factoring 2D weight parameters.
        mapper_factory: Callable ``(latent_dim, output_dim) -> BaseMapper``.
            Defaults to the orthogonal MLP mapper with additive modulation.
    """

    def __init__(
        self,
        parameter_spec: ParameterSpec,
        latent_dim: int,
        rank: int,
        *,
        mapper_factory: MapperFactory = default_mapper_factory,
    ) -> None:
        super().__init__(parameter_spec)
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.latent_dim = latent_dim
        self.rank = rank

        # Group layer-wise
        from marn.generators.layerwise import layerwise_groups

        self.groups = layerwise_groups(parameter_spec)
        self.group_names = tuple(self.groups)

        self.group_specs: list[ParameterSpec] = []
        self.layers = nn.ModuleList()
        self.group_lrd_info: dict[str, list[dict[str, Any]]] = {}

        for group_name, parameter_names in self.groups.items():
            total_elements = 0
            param_infos = []

            for name in parameter_names:
                entry = parameter_spec.entry(name)
                shape = entry.shape
                numel = entry.numel

                # Apply LRD for 2D weights where rank is strictly smaller than both dimensions
                is_lrd = len(shape) == 2 and rank < min(shape[0], shape[1])

                if is_lrd:
                    m, n = shape
                    u_numel = m * rank
                    v_numel = n * rank
                    gen_numel = u_numel + v_numel
                    info = {
                        "name": name,
                        "is_lrd": True,
                        "shape": shape,
                        "start": total_elements,
                        "end": total_elements + gen_numel,
                        "u_shape": (m, rank),
                        "v_shape": (n, rank),
                        "u_numel": u_numel,
                    }
                    total_elements += gen_numel
                else:
                    info = {
                        "name": name,
                        "is_lrd": False,
                        "shape": shape,
                        "start": total_elements,
                        "end": total_elements + numel,
                    }
                    total_elements += numel

                param_infos.append(info)

            self.group_lrd_info[group_name] = param_infos

            self.layers.append(
                LayerGenerator(
                    latent_dim,
                    total_elements,
                    mapper_factory(latent_dim, total_elements),
                )
            )

    def generate_parameters(self, latents: dict[str, Tensor] | None = None) -> ParameterTree:
        generated: dict[str, Tensor] = {}
        for name, layer, info_list in zip(
            self.group_names, self.layers, self.group_lrd_info.values(), strict=True
        ):
            latent = latents.get(name) if latents is not None else None
            flat_group = layer(latent)

            for info in info_list:
                param_name = info["name"]
                slice_tensor = flat_group[info["start"] : info["end"]]

                if info["is_lrd"]:
                    u_flat = slice_tensor[: info["u_numel"]]
                    v_flat = slice_tensor[info["u_numel"] :]

                    U = u_flat.view(info["u_shape"])
                    V = v_flat.view(info["v_shape"])

                    W = U @ V.T
                    generated[param_name] = W
                else:
                    generated[param_name] = slice_tensor.view(info["shape"])

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
