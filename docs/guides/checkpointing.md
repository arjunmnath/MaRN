# Checkpointing and Resumption

Mapping network checkpoints are intentionally **compact**: they exclude the
generated target weights (which are ephemeral and reproducible) and store only
the small, trainable state needed to continue training or perform inference.

## What is saved

| Field | Description |
|-------|-------------|
| `schema_version` | Integer bumped on breaking format changes |
| `package_version` | `marn` version string at save time |
| `latent_state` | `state_dict` of all trainable latent vectors |
| `mapper_buffers` | Fixed mapper tensors (e.g. orthogonal projection matrices) |
| `parameter_spec_names` | Ordered `(name, shape)` list — validated on load |
| `config` | Optional serialised `MappingConfig` dict for reference |
| `trainer_state` | Optional optimizer / scheduler / epoch state |
| `metadata` | Free-form dict for user annotations |

Generated target weights are **not** stored.  They are reconstructed on the
fly from latent vectors during inference.

## Quickstart

```python
from marn import (
    MappingModel,
    MappingTrainer,
    MappingConfig,
    save_checkpoint,
    load_checkpoint,
)

# ── Save ─────────────────────────────────────────────────────────────────
save_checkpoint(
    "checkpoints/epoch10.pt",
    model=model,
    config=mapping_config,   # optional — for reference only
    trainer=trainer,          # optional — saves optimizer / epoch state
    metadata={"dataset": "cifar10", "notes": "baseline run"},
)

# ── Load ─────────────────────────────────────────────────────────────────
schema = load_checkpoint(
    "checkpoints/epoch10.pt",
    model=model,
    trainer=trainer,   # optional — restores optimizer / epoch state
)
print(f"Resumed from epoch {schema.trainer_state['epoch']}")
print(f"Checkpoint metadata: {schema.metadata}")
```

## Saving a checkpoint

```python
from marn import save_checkpoint

save_checkpoint(
    path,
    model,
    *,
    config=None,
    trainer=None,
    save_trainer_state=True,
    metadata=None,
)
```

**Arguments**

- `path` — Destination file path.  Parent directories are created automatically.
- `model` — A `MappingModel` instance (may be on any device).
- `config` — Optional `MappingConfig`.  Saved for auditing; not used during load.
- `trainer` — Optional `MappingTrainer`.  When provided and `save_trainer_state`
  is `True`, the optimizer state dict, scheduler state dict, current epoch
  counter, and `should_stop` flag are serialised.
- `save_trainer_state` — Set to `False` to omit trainer state even when a
  trainer is supplied.
- `metadata` — Free-form `dict` of user annotations.

## Loading a checkpoint

```python
from marn import load_checkpoint

schema = load_checkpoint(
    path,
    model,
    *,
    trainer=None,
    map_location=None,
    strict_latents=True,
)
```

**Arguments**

- `path` — Path to the `.pt` checkpoint file.
- `model` — A `MappingModel` whose generator state will be restored.
- `trainer` — Optional `MappingTrainer`.  When provided and the checkpoint
  contains trainer state, optimizer, scheduler, epoch, and GradScaler are
  restored.
- `map_location` — Forwarded to `torch.load` for device remapping (e.g. loading
  a GPU checkpoint on CPU).
- `strict_latents` — When `True` (default), every saved key must match the
  generator exactly.  Set to `False` for partial loading.

**Returns** a `CheckpointSchema` dataclass with all checkpoint fields.

## Compatibility validation

`load_checkpoint` validates the saved `parameter_spec_names` against the
current model's `ParameterSpec` **before** touching any model state.  If there
is a mismatch, a `CheckpointCompatibilityError` is raised and the model is left
unchanged.

```python
from marn import CheckpointCompatibilityError

try:
    load_checkpoint("ckpt.pt", wrong_model)
except CheckpointCompatibilityError as e:
    print(f"Architecture mismatch: {e}")
```

Conditions that raise `CheckpointCompatibilityError`:

| Condition | Message contains |
|-----------|-----------------|
| Schema from a future release | `"newer"` |
| Missing `schema_version` key | `"schema_version"` |
| Non-dict payload | `"not a valid"` |
| Different number of target parameters | `"The target architecture differs"` |
| Parameter name or shape mismatch | `"shape mismatch"` / `"name mismatch"` |

## Resuming interrupted training

```python
# First run — train 10 epochs and save
trainer = MappingTrainer(model, train_loader, config=TrainerConfig(max_epochs=10))
trainer.fit()
save_checkpoint("mid.pt", model, trainer=trainer)

# Second run — resume from epoch 10
fresh_model = MappingModel(target_model, latent_dim=256)
fresh_trainer = MappingTrainer(fresh_model, train_loader, config=TrainerConfig(max_epochs=20))

schema = load_checkpoint("mid.pt", fresh_model, trainer=fresh_trainer)
# fresh_trainer.current_epoch == 9  (last completed epoch)

fresh_trainer.fit()   # continues for max_epochs more steps
```

> [!NOTE]
> `load_checkpoint` restores `current_epoch` and `should_stop` from the saved
> trainer state.  The training loop's `max_epochs` in the new `TrainerConfig`
> controls how many **additional** epochs are run.

## Device remapping

Load a checkpoint saved on GPU onto CPU:

```python
schema = load_checkpoint("gpu_ckpt.pt", model, map_location="cpu")
```

## Schema versioning

The `schema_version` field is an integer that will be incremented on breaking
format changes.  Forward-compatibility is not guaranteed: a checkpoint saved
by a newer version of the package that bumped the schema version will raise
`CheckpointCompatibilityError` when loaded by an older version.

The current schema version is available as:

```python
from marn.checkpoint.schema import CURRENT_SCHEMA_VERSION
```

## What is **not** saved

- **Generated target weights** — these are always reconstructed from latent
  vectors via `functional_call`; including them would make checkpoints as large
  as a full model save.
- **The target model itself** — the user must supply the same architecture when
  calling `load_checkpoint`.
- **DataLoader / dataset state** — shuffle state and sampler position are not
  captured.
