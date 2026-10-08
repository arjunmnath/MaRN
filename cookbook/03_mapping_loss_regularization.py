#!/usr/bin/env python3
"""Recipe 3: Mapping Loss and Regularization Components.

This recipe demonstrates configuring the composite Mapping Loss with Stability,
Smoothness, and Alignment loss terms. It also illustrates how to configure
trainable regularization coefficients, allowing the model to learn the optimal
balance of loss terms dynamically during training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from marn import (
    MappingModel,
    MappingLoss,
    ClassificationLoss,
    StabilityLoss,
    SmoothnessLoss,
    AlignmentLoss,
    MappingTrainer,
    TrainerConfig,
    Callback,
)


from typing import Any


class TargetModel(nn.Module):
    """Simple linear target model for classification."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> Any:
        return self.fc2(self.relu(self.fc1(x)))


class RegularizerLoggingCallback(Callback):
    """Custom callback to print trainable loss coefficients at the end of each epoch."""

    def on_epoch_end(
        self, trainer: MappingTrainer, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        loss_fn = trainer.loss_fn
        # Check if loss function has trainable coefficients
        if isinstance(loss_fn, MappingLoss) and loss_fn.trainable_coefficients:
            # Under the hood, these values are PyTorch parameters mapped through softplus
            stab = loss_fn.lambda_stability
            sm = loss_fn.lambda_smoothness
            align = loss_fn.lambda_alignment

            # Since they are PyTorch tensors, extract their scalar values
            stab_val = stab.item() if isinstance(stab, torch.Tensor) else stab
            sm_val = sm.item() if isinstance(sm, torch.Tensor) else sm
            align_val = align.item() if isinstance(align, torch.Tensor) else align

            print(
                f"[{epoch + 1:02d}] Dynamic Lambdas -> Stability: {stab_val:.4f} | Smoothness: {sm_val:.4f} | Alignment: {align_val:.4f}"
            )


def main() -> None:
    torch.manual_seed(42)

    # 1. Setup synthetic data
    x_train = torch.randn(128, 10)
    y_train = torch.randint(0, 2, (128,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

    # 2. Build model in Layer-wise mode
    target = TargetModel()
    model = MappingModel(target, latent_dim=16, strategy="layerwise")

    print("--- Configuring Mapping Loss components ---")
    # Stability loss: penalizes large changes in predictions under input latent noise
    stability_loss = StabilityLoss(epsilon=0.01)

    # Smoothness loss: penalizes the Frobenius norm of the Jacobian to enforce C2 continuity
    smoothness_loss = SmoothnessLoss(method="stochastic", num_projections=1)

    # Alignment loss: aligns latent coordinates with target weight features using cosine similarity
    alignment_loss = AlignmentLoss()

    # Create the composite MappingLoss with trainable_coefficients=True
    loss_fn = MappingLoss(
        task_loss=ClassificationLoss(),
        stability_loss=stability_loss,
        smoothness_loss=smoothness_loss,
        alignment_loss=alignment_loss,
        lambda_stability=0.1,  # Initial value
        lambda_smoothness=0.01,  # Initial value
        lambda_alignment=0.01,  # Initial value
        trainable_coefficients=True,
    )

    # 3. Setup trainer
    config = TrainerConfig(max_epochs=5, learning_rate=0.01)

    # We include our custom callback to log the coefficient updates
    trainer = MappingTrainer(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        config=config,
        callbacks=[RegularizerLoggingCallback()],
    )

    print("--- Starting Training with Trainable Loss Coefficients ---")
    trainer.fit()
    print("Training finished.")


if __name__ == "__main__":
    main()
