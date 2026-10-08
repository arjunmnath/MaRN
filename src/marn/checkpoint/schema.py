"""Versioned checkpoint schema for mapping network training state.

The checkpoint stores only what is needed to reconstruct and continue training:

- ``schema_version``: integer bumped on breaking format changes.
- ``package_version``: the marn package version string at save time.
- ``latent_state``: ``state_dict`` subset containing only the trainable latent
  tensors (keys produced by ``model.generator.named_latent_vectors``).
- ``mapper_buffers``: buffer state required for deterministic reconstruction
  (e.g. fixed orthogonal projection matrices).
- ``parameter_spec_names``: ordered parameter names and shapes from the target's
  ``ParameterSpec``.  Used to validate that the same architecture is loaded.
- ``config``: optional serialised ``MappingConfig`` dict saved for reference.
- ``trainer_state``: optional dict containing optimizer/scheduler/epoch state
  so training can be resumed at the same point.
- ``metadata``: free-form dict for user annotations (dataset, notes, …).

Generated target weights are **not** included.  Checkpoints are compact because
only latent vectors and small fixed buffers are saved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CURRENT_SCHEMA_VERSION: int = 1


@dataclass
class CheckpointSchema:
    """All fields written to / read from a checkpoint file.

    Attributes:
        schema_version: Integer incremented on breaking format changes.
        package_version: ``marn.__version__`` at save time.
        latent_state: Ordered ``state_dict`` of trainable latent tensors only.
        mapper_buffers: Buffer ``state_dict`` required for deterministic
            reconstruction (projection matrices, running statistics …).
        parameter_spec_names: Ordered list of ``(name, shape)`` tuples from
            the target ``ParameterSpec``.  Validated on load.
        config: Optional serialised ``MappingConfig`` dict (may be ``None``).
        trainer_state: Optional ``dict`` with optimizer/scheduler/epoch entries.
        metadata: User-supplied annotations, e.g. dataset or description.
    """

    schema_version: int
    package_version: str
    latent_state: dict[str, Any]
    mapper_buffers: dict[str, Any]
    parameter_spec_names: list[tuple[str, list[int]]]
    config: dict[str, Any] | None = None
    trainer_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
