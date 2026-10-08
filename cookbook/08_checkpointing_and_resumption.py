#!/usr/bin/env python3
"""Recipe 8: Saving, Loading, and Resuming Training from Checkpoints.

In practical deep learning workflows, training can be interrupted or we may want
to save intermediate states for evaluation. This recipe demonstrates how to use the
built-in `save_checkpoint` and `load_checkpoint` functions to seamlessly persist
both the model weights (including trainable latents and projection matrices) and
the trainer states (optimizer parameters, current epoch).
"""

from typing import Any
import os
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from marn import (
    MappingLoss,
    MappingModel,
    MappingTrainer,
    RegressionLoss,
    TrainerConfig,
    load_checkpoint,
    save_checkpoint,
)


class TargetLinearModel(nn.Module):
    """Simple target model for regression."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> Any:
        return self.fc(x)


def create_pipeline(max_epochs: int = 2) -> tuple[MappingModel, MappingTrainer, DataLoader[Any]]:
    """Helper to create a fresh target, mapping model, loss, and trainer."""
    torch.manual_seed(0)
    target = TargetLinearModel()

    # Generate synthetic regression data
    x = torch.randn(100, 3)
    y = x.sum(dim=-1, keepdim=True)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    model = MappingModel(target, latent_dim=8, strategy="layerwise")
    loss_fn = MappingLoss(task_loss=RegressionLoss())
    trainer = MappingTrainer(
        model=model,
        train_loader=loader,
        loss_fn=loss_fn,
        config=TrainerConfig(learning_rate=0.01, max_epochs=max_epochs),
    )
    return model, trainer, loader


def main() -> None:
    # Use a temp directory for saving checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "training_checkpoint.pt")

        # 1. Start initial training phase (Epochs 0 to 2)
        print("=== Phase 1: Initial Training ===")
        model_1, trainer_1, _ = create_pipeline(max_epochs=2)
        print("Training for 2 epochs...")
        trainer_1.fit()
        print(
            f"Trainer epoch after initial fit: {trainer_1.current_epoch} (0-indexed, meaning epoch 1 completed)"
        )

        # Save dummy input prediction to compare later
        dummy_input = torch.tensor([[1.0, 2.0, 3.0]])
        res_1 = model_1(dummy_input)
        pred_before_save = res_1.predictions.detach().clone()
        print(f"Prediction before saving checkpoint: {pred_before_save.item():.6f}")

        # Save checkpoint (including trainer states: optimizer, epoch, metadata)
        print(f"Saving checkpoint to {checkpoint_path}...")
        save_checkpoint(
            path=checkpoint_path,
            model=model_1,
            trainer=trainer_1,
            metadata={"description": "Checkpoint from epoch 2 in cookbook Recipe 8"},
        )

        # 2. Re-initialize a fresh model & trainer (Phase 2: Resumption)
        print("\n=== Phase 2: Resuming Training ===")
        model_2, trainer_2, _ = create_pipeline(max_epochs=4)

        # Verify un-loaded fresh model prediction differs
        res_fresh = model_2(dummy_input)
        pred_before_load = res_fresh.predictions.detach()
        print(f"Prediction of fresh model (before loading): {pred_before_load.item():.6f}")

        # Restore checkpoint state
        print("Loading checkpoint and restoring model + trainer state...")
        schema = load_checkpoint(
            path=checkpoint_path,
            model=model_2,
            trainer=trainer_2,
        )
        print(f"Metadata stored in checkpoint: {schema.metadata}")
        print(
            f"Resumed trainer epoch state: {trainer_2.current_epoch} (0-indexed, loaded from checkpoint)"
        )

        # Verify predictions match exactly
        res_loaded = model_2(dummy_input)
        pred_after_load = res_loaded.predictions.detach()
        print(f"Prediction after loading checkpoint: {pred_after_load.item():.6f}")
        assert torch.allclose(pred_before_save, pred_after_load), "Restored model output mismatch!"
        print("Model prediction check passed (outputs match exactly).")

        # 3. Resume training from epoch 2 to 4
        print("\n=== Phase 3: Continuing Training ===")
        print("Training for 4 epochs...")
        trainer_2.fit()
        print(
            f"Final trainer epoch: {trainer_2.current_epoch} (0-indexed, meaning epoch 3 completed)"
        )
        print("Training resumption completed successfully.")


if __name__ == "__main__":
    main()
