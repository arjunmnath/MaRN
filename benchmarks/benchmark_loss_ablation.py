#!/usr/bin/env python3
"""Benchmark performing an ablation study on the components of MappingLoss using real MNIST data."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from accelerate import Accelerator
from marn import (
    MappingModel,
    MappingLoss,
    ClassificationLoss,
    StabilityLoss,
    SmoothnessLoss,
    AlignmentLoss,
    MappingTrainer,
    TrainerConfig,
)

from typing import Any


# ---------------------------------------------------------------------------
# Dataset helpers — reuse MNIST (same as benchmark_cnn)
# ---------------------------------------------------------------------------

_MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

_DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def _get_mnist_loaders(
    train_size: int = 512,
    val_size: int = 256,
    batch_size: int = 64,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_ds = datasets.MNIST(_DATA_ROOT, train=True, download=True, transform=_MNIST_TRANSFORM)
    val_ds = datasets.MNIST(_DATA_ROOT, train=False, download=True, transform=_MNIST_TRANSFORM)
    train_ds = Subset(train_ds, list(range(train_size)))
    val_ds = Subset(val_ds, list(range(val_size)))
    train_loader: DataLoader[Any] = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """Small CNN target for the loss ablation study (MNIST, 10-class)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Any:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_ablation(
    use_stab: bool,
    use_smooth: bool,
    use_align: bool,
    epochs: int = 3,
    accelerator: Accelerator | None = None,
) -> float:
    torch.manual_seed(42)

    if accelerator is None:
        accelerator = Accelerator()

    train_loader, val_loader = _get_mnist_loaders()
    device_str = str(accelerator.device)

    target_model = SimpleCNN()
    model = MappingModel(target_model, latent_dim=256, strategy="layerwise")

    stab_fn = StabilityLoss(epsilon=0.01) if use_stab else None
    smooth_fn = SmoothnessLoss(method="stochastic", num_projections=1) if use_smooth else None
    align_fn = AlignmentLoss() if use_align else None

    loss_fn = MappingLoss(
        task_loss=ClassificationLoss(),
        stability_loss=stab_fn,
        smoothness_loss=smooth_fn,
        alignment_loss=align_fn,
        lambda_stability=0.1,
        lambda_smoothness=0.01,
        lambda_alignment=0.01,
    )

    config = TrainerConfig(max_epochs=epochs, learning_rate=0.01, device=device_str)
    trainer = MappingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        loss_fn=loss_fn,
        config=config,
    )

    trainer.fit()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device_str), y.to(device_str)
            res = model(x)
            correct += (res.predictions.argmax(dim=-1) == y).sum().item()
            total += y.numel()

    return correct / total


def run_benchmarks(accelerator: Accelerator | None = None) -> dict[str, float]:
    if accelerator is None:
        accelerator = Accelerator()

    results: dict[str, float] = {}

    configs = [
        ("Task Loss Only", False, False, False),
        ("+ Stability", True, False, False),
        ("+ Smoothness", False, True, False),
        ("+ Alignment", False, False, True),
        ("+ Smoothness + Align", False, True, True),
        ("+ Stab + Smooth", True, True, False),
        ("Full Mapping Loss", True, True, True),
    ]

    for name, use_stab, use_smooth, use_align in configs:
        print(f"  [Ablation] {name}")
        results[name] = run_ablation(use_stab, use_smooth, use_align, accelerator=accelerator)

    return results


if __name__ == "__main__":
    acc = Accelerator()
    print(f"Running Loss Ablation benchmarks on: {acc.device}")
    res = run_benchmarks(accelerator=acc)
    for config_name, accuracy in res.items():
        print(f"  {config_name}: Accuracy = {accuracy * 100:.2f}%")
