# Trainer and callbacks

> **Status**: Implemented in Milestone 8.

The `MappingTrainer` class manages the training loop, validation loop, device placement, precision scaling, and callback notifications. It keeps the high-level mapping and loss logic clean by managing only the optimization process.

## Quick start

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from marn import MappingModel, MappingTrainer, TrainerConfig
from marn.callbacks import EarlyStopping, MetricLogger

# Create model and loaders
model = MappingModel(target_model, latent_dim=128)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# Configure trainer
config = TrainerConfig(
    max_epochs=50,
    learning_rate=1e-3,
    optimizer="adamw",
    scheduler="cosine",
    device="auto",
    seed=42,
)

trainer = MappingTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5),
        MetricLogger(log_every_n_batches=10),
    ],
)

history = trainer.fit()
```

## Batch adapters

DataLoader output formats vary across datasets. To support arbitrary formats without modifying the training loop, `MappingTrainer` delegates batch unpacking to a `BatchAdapter`.

### Built-in adapters

#### TupleBatchAdapter

Unpacks standard PyTorch datasets returning `(inputs, targets)` or `(inputs, targets, *extra)`. Extra elements are ignored.

```python
from marn.trainers import TupleBatchAdapter

# Automatically used by default in MappingTrainer
adapter = TupleBatchAdapter()
```

#### MappingBatchAdapter

Unpacks dict-like batches (common in Hugging Face or text datasets) by key:

```python
from marn.trainers import MappingBatchAdapter

# For a batch returning {"pixel_values": X, "label": y}
adapter = MappingBatchAdapter(input_key="pixel_values", target_key="label")
```

### Custom batch adapter

If your data format is complex (e.g. structured keywords), subclass `BatchAdapter` and override `unpack`:

```python
from typing import Any
from marn.trainers import BatchAdapter

class CustomBatchAdapter(BatchAdapter):
    def unpack(self, batch: Any) -> tuple[tuple[Any, ...], Any]:
        # Return positional arguments to pass to target forward, and targets
        return (batch.features, batch.metadata), batch.label
```

## Training features

- **Device selection**: Resolves `"auto"` to CUDA when available, otherwise CPU. Automatically places the model, data, and PyTorch modules on the target device.
- **AMP (Automatic Mixed Precision)**: Enabled with `amp_enabled=True` in `TrainerConfig`. Uses `torch.amp.autocast` and `GradScaler` for training.
- **Gradient accumulation**: Splits a batch size logically across multiple forward steps to save memory. Set `accumulation_steps > 1`.
- **Gradient clipping**: Supports clipping gradients either by norm or absolute value:
  - `gradient_clip_norm: float`
  - `gradient_clip_value: float`

## Callbacks

Subclass `Callback` to observe or modify the training run at key life cycle hooks:

```python
from marn import Callback

class CustomCallback(Callback):
    def on_fit_start(self, trainer):
        print("Training starting!")
        
    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"Epoch {epoch} finished. Metrics: {metrics}")
```

### Hooks lifecycle

1. `on_fit_start(trainer)`
2. For each epoch:
   a. `on_epoch_start(trainer, epoch)`
   b. For each batch: `on_batch_start(trainer, batch_idx)` → `on_batch_end(trainer, batch_idx, loss_output)`
   c. `on_validation_start(trainer)` → `on_validation_end(trainer, metrics)`
   d. `on_epoch_end(trainer, epoch, metrics)`
3. `on_fit_end(trainer)`

Callbacks can alter trainer behavior (e.g., setting `trainer.should_stop = True` to terminate the training loop).

### Built-in callbacks

- **`EarlyStopping(monitor, patience, min_delta, mode)`**: Stops training early if a metric (e.g. `val_loss`) stops improving.
- **`MetricLogger(log_every_n_batches)`**: Prints epoch and step loss metrics using standard Python logging.

## Learning Rate Finder

Before training, selecting an appropriate learning rate is crucial. The `MappingTrainer` provides a production-quality Learning Rate Finder that implements Leslie Smith's Learning Rate Range Test. It exponentially increases the learning rate batch-by-batch, records the loss, stops early if the loss diverges, and suggests a recommended learning rate using a robust steepest-gradient heuristic.

Importantly, **the finder completely restores the trainer's state** (including model weights, optimizer/scheduler values, AMP scaler, and random number generator states) so that subsequent training starts from a clean slate.

### Manual Tuning

You can run the sweep manually to analyze and plot the loss versus learning rate:

```python
# Run the sweep (returns an LRFinderResult)
lr_finder = trainer.lr_find(
    min_lr=1e-7,
    max_lr=10.0,
    num_iterations=100,
)

# Plot the sweep (requires matplotlib)
lr_finder.plot()

# Print the recommended learning rate
print(lr_finder.suggestion())
```

Alternatively, you can assign the suggestion and start training:

```python
result = trainer.lr_find()
trainer.learning_rate = result.suggestion()
trainer.fit()
```

### Automatic Tuning

To run the finder and start training automatically in a single call, set `auto_lr_find=True` in `fit()`:

```python
trainer.fit(auto_lr_find=True)
```

