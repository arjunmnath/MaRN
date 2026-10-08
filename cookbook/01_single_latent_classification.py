#!/usr/bin/env python3
"""Recipe 1: Single Latent Vector Training (SLVT) classification.

This example illustrates the fundamental workflow of Mapping Networks: wrapping
a standard PyTorch CNN, setting up classification task loss, and running the
fit loop using the Single Latent Vector Training (SLVT) strategy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from marn import (
    MappingModel,
    MappingLoss,
    ClassificationLoss,
    MappingTrainer,
    TrainerConfig,
    MetricLogger,
    EarlyStopping,
)


from typing import Any


# 1. Define target architecture
class TargetCNN(nn.Module):
    """Standard CNN classification module."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Any:
        x = self.features(x)
        return self.classifier(x)


def main() -> None:
    # Set random seed for reproducibility
    torch.manual_seed(42)

    print("--- Step 1: Preparing MNIST Dataset ---")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Download MNIST dataset (data/ directory is gitignored)
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Use a small subset of MNIST for a fast, responsive demo
    train_subset = torch.utils.data.Subset(train_dataset, range(512))
    val_subset = torch.utils.data.Subset(val_dataset, range(128))

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    print("--- Step 2: Instantiating Target CNN and MappingModel ---")
    target = TargetCNN(num_classes=10)

    # Wrap with MappingModel in SLVT mode.
    # The entire model weights will be generated from a single trainable latent vector of size 512.
    model = MappingModel(
        target_model=target,
        latent_dim=512,
        strategy="slvt",
    )

    print(f"Target parameter count: {sum(p.numel() for p in target.parameters()):,}")
    print(
        f"Trainable mapping parameters (latent vector): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    print("--- Step 3: Configuring Composite Loss and Trainer ---")
    # We combine standard classification cross-entropy with the mapping trainer
    loss_fn = MappingLoss(task_loss=ClassificationLoss())

    # Configure training settings
    config = TrainerConfig(
        max_epochs=10,
        learning_rate=0.01,
        device="cpu",
    )

    # Setup callbacks for metric tracking and early stopping
    callbacks = [
        MetricLogger(log_every_n_batches=2),
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
    ]

    trainer = MappingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        callbacks=callbacks,
    )

    print("--- Step 4: Training the Latent Space ---")
    history = trainer.fit()

    print("--- Step 5: Evaluating Final Performance ---")
    # Verify validation metrics in history
    final_val_loss = history["val_loss"][-1]
    print(f"Training completed. Final Validation Loss: {final_val_loss:.4f}")

    # Evaluate final validation accuracy manually
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            res = model(x)
            correct += (res.predictions.argmax(dim=-1) == y).sum().item()
            total += y.numel()

    print(f"Final Validation Accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()
