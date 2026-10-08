"""Typed configuration models and YAML/dict loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from marn.config.loss import LossConfig, TaskLossConfig
from marn.config.model import GeneratorConfig, MapperConfig, MappingConfig
from marn.config.trainer import TrainerConfig


def load_config(source: str | Path | dict[str, Any]) -> MappingConfig:
    """Load a ``MappingConfig`` from a YAML file path or Python dict.

    Args:
        source: A file path (str or Path) to a YAML file, or a Python dict.

    Returns:
        A validated ``MappingConfig`` instance.

    Raises:
        FileNotFoundError: If a file path is given and the file does not exist.
        pydantic.ValidationError: If the data fails validation (including
            unknown keys due to ``extra="forbid"``).

    Example::

        # From a Python dict
        config = load_config({"generator": {"latent_dim": 64}})

        # From a YAML file
        config = load_config("config.yaml")
    """
    if isinstance(source, dict):
        return MappingConfig(**source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at the top level, got {type(data).__name__}")

    return MappingConfig(**data)


__all__ = [
    "GeneratorConfig",
    "LossConfig",
    "MapperConfig",
    "MappingConfig",
    "TaskLossConfig",
    "TrainerConfig",
    "load_config",
]
