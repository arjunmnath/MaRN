"""Lazy (group-at-a-time) layerwise parameter generation.

``LazyLayerwiseGenerator`` is a memory-optimised variant of
:class:`~marn.generators.layerwise.LayerwiseGenerator`.

Compatibility constraints
-------------------------
True streaming (generate one layer → call target with that layer → discard →
move to next layer) is **not** compatible with
:func:`torch.func.functional_call`, which requires the *entire* parameter
dictionary to be present at call time.

``LazyLayerwiseGenerator`` therefore uses a *soft-lazy* strategy:

1. Generate each group's flat tensor.
2. Immediately unflatten it into named tensors and add them to the output dict.
3. Delete the intermediate flat buffer (``del flat_tensor``) so it can be
   reclaimed by the Python allocator before the next group is generated.
4. On CUDA, call :func:`torch.cuda.empty_cache` after each group to return
   freed memory to the pool immediately.

This reduces *peak* memory compared with the eager approach (which first
materialises every group's flat buffer and only then unflattens), at the cost
of one extra ``empty_cache`` call per layer group.

When to use
-----------
* Target models with many large layers where all flat buffers held
  simultaneously would exceed memory.
* CPU profiling/debugging where memory is limited.
* **Avoid** when ``torch.compile`` traces the entire forward as a single graph,
  because the ``empty_cache`` call inside the loop may cause graph breaks.
"""

from __future__ import annotations

import torch
from torch import Tensor

from marn.generators.layerwise import LayerwiseGenerator
from marn.generators.base import MapperFactory, default_mapper_factory
from marn.generators.layer_generator import LayerGenerator
from marn.runtime import ParameterSpec, ParameterTree


class LazyLayerwiseGenerator(LayerwiseGenerator):
    """Layerwise generator that frees each group's flat buffer as it goes.

    Produces the same :class:`~marn.runtime.ParameterTree` as
    :class:`~marn.generators.layerwise.LayerwiseGenerator` but
    with a lower peak memory footprint.

    See the module docstring for compatibility constraints.

    Args:
        parameter_spec: Target :class:`~marn.runtime.ParameterSpec`.
        latent_dim: Dimensionality of each layer's trainable latent vector.
        mapper_factory: Callable ``(latent_dim, output_dim) → BaseMapper``.
            Defaults to the orthogonal MLP mapper with additive modulation.
        empty_cache_on_cuda: When *True* (default), call
            :func:`torch.cuda.empty_cache` after freeing each group's buffer
            if CUDA is available.  Set to *False* to skip the cache flush (e.g.
            in micro-benchmarks where the overhead matters).
    """

    def __init__(
        self,
        parameter_spec: ParameterSpec,
        latent_dim: int,
        *,
        mapper_factory: MapperFactory = default_mapper_factory,
        empty_cache_on_cuda: bool = True,
    ) -> None:
        super().__init__(parameter_spec, latent_dim, mapper_factory=mapper_factory)
        self._empty_cache_on_cuda = empty_cache_on_cuda

    def generate_parameters(self, latents: dict[str, Tensor] | None = None) -> ParameterTree:
        """Generate target parameters one group at a time, freeing intermediates.

        Returns:
            A fully populated :class:`~marn.runtime.ParameterTree`
            with an entry for every parameter in the spec.
        """
        generated: dict[str, Tensor] = {}

        for name, layer, group_spec in zip(
            self.group_names, self.layers, self.group_specs, strict=True
        ):
            assert isinstance(layer, LayerGenerator)
            latent = latents.get(name) if latents is not None else None

            # Generate the flat tensor for this group
            flat_tensor: Tensor = layer(latent)

            # Unflatten into named tensors and accumulate
            generated.update(group_spec.unflatten(flat_tensor).to_dict())

            # Free the flat buffer immediately
            del flat_tensor
            if self._empty_cache_on_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

        tree = ParameterTree(generated)
        self.parameter_spec.validate_tree(tree)
        return tree
