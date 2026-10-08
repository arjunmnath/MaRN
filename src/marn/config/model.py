"""Pydantic configuration models for mapper, generator, and mapping setup."""

from __future__ import annotations


from pydantic import BaseModel, ConfigDict, Field


class MapperConfig(BaseModel):
    """Configuration for the latent-to-parameter mapper.

    Attributes:
        type: Registry key for the mapper class (``"mlp"`` or ``"residual_mlp"``).
        hidden_dims: Hidden layer dimensions for the MLP mapper.
        modulation: Registry key for the modulation strategy.
        alpha: Modulation strength for additive modulation.
        hidden_dim: Hidden dimension for residual MLP mapper (ignored for ``"mlp"``).
        depth: Number of residual blocks (ignored for ``"mlp"``).
    """

    model_config = ConfigDict(extra="forbid")

    type: str = "mlp"
    hidden_dims: tuple[int, ...] = ()
    modulation: str = "additive"
    alpha: float = 0.01
    hidden_dim: int = 128
    depth: int = 2


class GeneratorConfig(BaseModel):
    """Configuration for the parameter generation strategy.

    Attributes:
        strategy: Strategy name — ``"layerwise"`` (default), ``"slvt"``, or ``"grouped"``.
        latent_dim: Dimensionality of each trainable latent vector.
        groups: Explicit parameter groups for the ``"grouped"`` strategy.
            Maps group name to list of parameter names.
        max_projection_elements: Upper bound for SLVT projection matrix size.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = "layerwise"
    latent_dim: int = 256
    groups: dict[str, list[str]] | None = None
    max_projection_elements: int = 100_000_000


class MappingConfig(BaseModel):
    """Top-level configuration for a mapping network.

    Combines generator and mapper configuration into a single validated object.
    Load from a Python dict or YAML file using :func:`marn.config.load_config`.

    Example::

        config = MappingConfig(
            generator=GeneratorConfig(strategy="layerwise", latent_dim=64),
            mapper=MapperConfig(type="mlp", modulation="additive"),
        )
    """

    model_config = ConfigDict(extra="forbid")

    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    mapper: MapperConfig = Field(default_factory=MapperConfig)
