"""Save a mapping network checkpoint to disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from marn.checkpoint.schema import CURRENT_SCHEMA_VERSION, CheckpointSchema

if TYPE_CHECKING:
    from marn.config.model import MappingConfig
    from marn.models.mapping_model import MappingModel
    from marn.trainers.trainer import MappingTrainer

logger = logging.getLogger("marn.checkpoint")


def save_checkpoint(
    path: str | Path,
    model: MappingModel,
    *,
    config: MappingConfig | None = None,
    trainer: MappingTrainer | None = None,
    save_trainer_state: bool = True,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a compact, versioned checkpoint for the mapping network.

    Only trainable latent vectors and required mapper buffers are written.
    Generated target weights are excluded — they are ephemeral and can be
    fully reconstructed at inference time.

    Args:
        path: Destination file path (e.g. ``"ckpt/epoch10.pt"``).
            Parent directories are created automatically.
        model: A ``MappingModel`` instance (may be on any device).
        config: Optional ``MappingConfig`` saved for reference.  Not used
            during load, but useful for reproducibility auditing.
        trainer: Optional ``MappingTrainer`` whose optimizer, scheduler, and
            epoch state are serialised when ``save_trainer_state`` is ``True``.
        save_trainer_state: When *True* and a *trainer* is supplied, include
            optimizer state dict, scheduler state dict (if any), current epoch,
            and ``should_stop`` flag so training can be resumed exactly.
        metadata: Free-form dict of user annotations (dataset name, notes …).

    Example::

        save_checkpoint(
            "checkpoints/epoch10.pt",
            model=model,
            config=mapping_config,
            trainer=trainer,
        )
    """
    import marn  # local import to avoid circular at module level

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # ── Latent state ──────────────────────────────────────────────────────
    # Only save trainable parameters (latent vectors); mapper buffers are
    # captured separately so the two sets are clearly delineated.
    latent_names = {name for name, _ in model.generator.named_latent_vectors()}
    full_state = model.generator.state_dict()

    latent_state: dict[str, Any] = {k: v for k, v in full_state.items() if k in latent_names}
    mapper_buffers: dict[str, Any] = {k: v for k, v in full_state.items() if k not in latent_names}

    # ── ParameterSpec fingerprint ─────────────────────────────────────────
    spec = model.target.parameter_spec
    spec_names: list[tuple[str, list[int]]] = [(entry.name, list(entry.shape)) for entry in spec]

    # ── Config ───────────────────────────────────────────────────────────
    config_dict: dict[str, Any] | None = None
    if config is not None:
        config_dict = config.model_dump()

    # ── Trainer state ────────────────────────────────────────────────────
    trainer_state: dict[str, Any] | None = None
    if trainer is not None and save_trainer_state:
        trainer_state = {
            "epoch": trainer.current_epoch,
            "should_stop": trainer.should_stop,
            "optimizer": trainer.optimizer.state_dict(),
        }
        if trainer.scheduler is not None:
            trainer_state["scheduler"] = trainer.scheduler.state_dict()
        if trainer.scaler is not None:
            trainer_state["scaler"] = trainer.scaler.state_dict()

    schema = CheckpointSchema(
        schema_version=CURRENT_SCHEMA_VERSION,
        package_version=marn.__version__,
        latent_state=latent_state,
        mapper_buffers=mapper_buffers,
        parameter_spec_names=spec_names,
        config=config_dict,
        trainer_state=trainer_state,
        metadata=metadata or {},
    )

    payload = {
        "schema_version": schema.schema_version,
        "package_version": schema.package_version,
        "latent_state": schema.latent_state,
        "mapper_buffers": schema.mapper_buffers,
        "parameter_spec_names": schema.parameter_spec_names,
        "config": schema.config,
        "trainer_state": schema.trainer_state,
        "metadata": schema.metadata,
    }

    torch.save(payload, path)
    logger.info(
        "Checkpoint saved to %s  (latents=%d  buffers=%d)",
        path,
        len(latent_state),
        len(mapper_buffers),
    )
