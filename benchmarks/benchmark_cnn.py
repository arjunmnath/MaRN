#!/usr/bin/env python3
"""Benchmark comparing Direct training, SLVT, and LWT on CNN1 and CNN2 targets using MNIST."""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from accelerate import Accelerator
from marn import (
    MappingModel,
    MappingLoss,
    ClassificationLoss,
    MappingTrainer,
    TrainerConfig,
    profile_peak_memory,
)

from typing import Any


# ---------------------------------------------------------------------------
# Dataset helpers
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
# Model definitions
# ---------------------------------------------------------------------------


class CNN1(nn.Module):
    """AlexNet-like CNN variant (~538k parameters) for MNIST."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 862),
            nn.ReLU(),
            nn.Linear(862, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Any:
        x = self.features(x)
        return self.classifier(x)


class CNN2(nn.Module):
    """LeNet-like CNN variant (~108k parameters) for MNIST."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 60),
            nn.ReLU(),
            nn.Linear(60, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Any:
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_cnn_experiment(
    target_cls: type[nn.Module],
    strategy_name: str,
    latent_dim: int,
    epochs: int = 3,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    torch.manual_seed(42)

    if accelerator is None:
        accelerator = Accelerator()

    train_loader, val_loader = _get_mnist_loaders()

    target_model = target_cls()
    total_params = sum(p.numel() for p in target_model.parameters())

    if strategy_name == "direct":
        trainable_params = total_params
        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        model, optimizer, train_loader, val_loader = accelerator.prepare(
            target_model, optimizer, train_loader, val_loader
        )

        def train_one_epoch() -> None:
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                accelerator.backward(loss)
                optimizer.step()

        # Warm-up epoch before timing
        train_one_epoch()

        t0 = time.perf_counter()
        _, peak_mem = profile_peak_memory(train_one_epoch, device=str(accelerator.device))
        t1 = time.perf_counter()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                correct += (out.argmax(dim=-1) == y).sum().item()
                total += y.numel()

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "peak_mem_mb": max(peak_mem / 1e6, 0.5),
            "epoch_time_ms": (t1 - t0) * 1000.0,
            "accuracy": correct / total,
        }

    else:
        # Mapping model path
        num_groups = 5 if target_cls == CNN1 else 4
        trainable_params = latent_dim if strategy_name == "slvt" else latent_dim * num_groups

        mapping_model = MappingModel(
            target_model=target_model,
            latent_dim=latent_dim,
            strategy=strategy_name,
        )
        mapping_loss_fn = MappingLoss(task_loss=ClassificationLoss())
        device_str = str(accelerator.device)
        config = TrainerConfig(max_epochs=epochs, learning_rate=0.01, device=device_str)

        # Prepare raw loaders for MappingTrainer (it manages its own device placement via config)
        raw_train, raw_val = _get_mnist_loaders()

        trainer = MappingTrainer(
            model=mapping_model,
            train_loader=raw_train,
            val_loader=raw_val,
            loss_fn=mapping_loss_fn,
            config=config,
        )

        # Warm-up
        trainer.fit()

        t0 = time.perf_counter()
        _, peak_mem = profile_peak_memory(lambda: trainer.fit(), device=device_str)
        t1 = time.perf_counter()

        mapping_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in raw_val:
                x, y = x.to(device_str), y.to(device_str)
                res = mapping_model(x)
                correct += (res.predictions.argmax(dim=-1) == y).sum().item()
                total += y.numel()

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "peak_mem_mb": max(peak_mem / 1e6, 0.5),
            "epoch_time_ms": (t1 - t0) * 1000.0 / epochs,
            "accuracy": correct / total,
        }


def run_benchmarks(accelerator: Accelerator | None = None) -> dict[str, Any]:
    if accelerator is None:
        accelerator = Accelerator()

    results: dict[str, Any] = {}

    print("  [CNN1] direct")
    results["cnn1_direct"] = run_cnn_experiment(CNN1, "direct", 0, accelerator=accelerator)

    print("  [CNN1] layerwise latent=816 (5 groups → 4080 total)")
    results["cnn1_lwt_4080"] = run_cnn_experiment(CNN1, "layerwise", 816, accelerator=accelerator)

    print("  [CNN2] direct")
    results["cnn2_direct"] = run_cnn_experiment(CNN2, "direct", 0, accelerator=accelerator)

    print("  [CNN2] layerwise latent=468 (4 groups → 1872 total)")
    results["cnn2_lwt_1872"] = run_cnn_experiment(CNN2, "layerwise", 468, accelerator=accelerator)

    print("  [CNN2] layerwise latent=672 (4 groups → 2688 total)")
    results["cnn2_lwt_2688"] = run_cnn_experiment(CNN2, "layerwise", 672, accelerator=accelerator)

    return results


if __name__ == "__main__":
    acc = Accelerator()
    print(f"Running CNN benchmarks on: {acc.device}")
    res = run_benchmarks(accelerator=acc)
    for name, stats in res.items():
        print(
            f"{name}: params={stats['total_params']:,}, trainable={stats['trainable_params']:,}, "
            f"mem={stats['peak_mem_mb']:.2f}MB, time={stats['epoch_time_ms']:.1f}ms, "
            f"acc={stats['accuracy'] * 100:.2f}%"
        )
