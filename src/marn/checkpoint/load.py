"""Load a mapping network checkpoint from disk and restore model/trainer state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from marn.checkpoint.schema import CURRENT_SCHEMA_VERSION, CheckpointSchema

if TYPE_CHECKING:
    from marn.models.mapping_model import MappingModel
    from marn.trainers.trainer import MappingTrainer

logger = logging.getLogger("marn.checkpoint")


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint's architecture is incompatible with the model.

    Possible causes:

    * Schema version from a future release is not understood by this version.
    * Target ``ParameterSpec`` names or shapes differ (different architecture).
    * Latent or buffer tensors do not match the current generator's keys.
    """


def _load_payload(path: Path, map_location: Any) -> dict[str, Any]:
    raw = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(raw, dict):
        raise CheckpointCompatibilityError(
            f"Checkpoint at {path} is not a valid marn checkpoint "
            f"(expected a dict, got {type(raw).__name__})."
        )
    return raw


def _parse_schema(payload: dict[str, Any], path: Path) -> CheckpointSchema:
    schema_version = payload.get("schema_version")
    if schema_version is None:
        raise CheckpointCompatibilityError(
            f"Checkpoint at {path} is missing 'schema_version'. "
            "It may have been saved by an incompatible tool."
        )
    if schema_version > CURRENT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"Checkpoint schema version {schema_version} is newer than the "
            f"current package supports ({CURRENT_SCHEMA_VERSION}). "
            "Upgrade marn to load this checkpoint."
        )
    return CheckpointSchema(
        schema_version=schema_version,
        package_version=payload.get("package_version", "unknown"),
        latent_state=payload["latent_state"],
        mapper_buffers=payload["mapper_buffers"],
        parameter_spec_names=payload["parameter_spec_names"],
        config=payload.get("config"),
        trainer_state=payload.get("trainer_state"),
        metadata=payload.get("metadata", {}),
    )


def _validate_spec(schema: CheckpointSchema, model: MappingModel, path: Path) -> None:
    """Raise ``CheckpointCompatibilityError`` if the target spec has changed."""
    spec = model.target.parameter_spec
    current: list[tuple[str, list[int]]] = [(entry.name, list(entry.shape)) for entry in spec]
    saved = [(name, shape) for name, shape in schema.parameter_spec_names]

    if len(current) != len(saved):
        raise CheckpointCompatibilityError(
            f"Checkpoint at {path} has {len(saved)} target parameters but "
            f"the model has {len(current)}.  The target architecture differs."
        )

    mismatches: list[str] = []
    for (cur_name, cur_shape), (sav_name, sav_shape) in zip(current, saved):
        if cur_name != sav_name:
            mismatches.append(f"  name mismatch: checkpoint={sav_name!r} model={cur_name!r}")
        elif cur_shape != sav_shape:
            mismatches.append(
                f"  shape mismatch for {cur_name!r}: checkpoint={sav_shape} model={cur_shape}"
            )

    if mismatches:
        detail = "\n".join(mismatches)
        raise CheckpointCompatibilityError(
            f"Checkpoint at {path} is incompatible with this model:\n{detail}"
        )


def load_checkpoint(
    path: str | Path,
    model: MappingModel,
    *,
    trainer: MappingTrainer | None = None,
    map_location: Any = None,
    strict_latents: bool = True,
) -> CheckpointSchema:
    """Load a checkpoint and restore model (and optionally trainer) state.

    The target's ``ParameterSpec`` is validated against the checkpoint before
    any state is applied.  If the architecture differs a
    ``CheckpointCompatibilityError`` is raised and the model is left unchanged.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        model: A ``MappingModel`` whose generator state will be restored.
        trainer: Optional ``MappingTrainer`` whose optimizer, scheduler, epoch,
            and GradScaler state will be restored when present in the checkpoint.
        map_location: Passed to ``torch.load`` for device remapping.
        strict_latents: When *True*, ``load_state_dict`` is called in strict
            mode — every saved latent and buffer key must match the generator.
            Set to *False* to allow partial loading (e.g. subsets of layers).

    Returns:
        The parsed ``CheckpointSchema`` so callers can inspect ``metadata``,
        ``config``, or ``trainer_state`` independently.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        CheckpointCompatibilityError: If schema version or architecture is
            incompatible.

    Example::

        schema = load_checkpoint("checkpoints/epoch10.pt", model=model, trainer=trainer)
        print(f"Resumed from epoch {schema.trainer_state['epoch']}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = _load_payload(path, map_location)
    schema = _parse_schema(payload, path)

    # Validate architecture before touching any model state
    _validate_spec(schema, model, path)

    # Restore generator state (latents + buffers together)
    combined_state: dict[str, Any] = {**schema.latent_state, **schema.mapper_buffers}
    model.generator.load_state_dict(combined_state, strict=strict_latents)

    logger.info(
        "Checkpoint loaded from %s  (schema_version=%d  package=%s)",
        path,
        schema.schema_version,
        schema.package_version,
    )

    # Restore trainer state if both are available
    if trainer is not None and schema.trainer_state is not None:
        ts = schema.trainer_state
        trainer.current_epoch = ts.get("epoch", 0)
        trainer.should_stop = ts.get("should_stop", False)

        if "optimizer" in ts:
            trainer.optimizer.load_state_dict(ts["optimizer"])

        if "scheduler" in ts and trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(ts["scheduler"])

        if "scaler" in ts and trainer.scaler is not None:
            trainer.scaler.load_state_dict(ts["scaler"])

        logger.info(
            "Trainer state restored: epoch=%d  should_stop=%s",
            trainer.current_epoch,
            trainer.should_stop,
        )

    return schema
