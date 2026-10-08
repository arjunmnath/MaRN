#!/usr/bin/env python3
"""Benchmark demonstrating the impact of LRD and Pruning on Mapping Networks using MNIST."""

import os
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from accelerate import Accelerator
from marn import (
    MappingModel,
    MappingLoss,
    ClassificationLoss,
    MappingTrainer,
    TrainerConfig,
    LRDStrategy,
    SLVTStrategy,
    profile_peak_memory,
)

from typing import Any


# ---------------------------------------------------------------------------
# Dataset helpers — reuse MNIST
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


class CNN2(nn.Module):
    """LeNet-like CNN for MNIST (10-class)."""

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
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_addons_experiment(
    strategy_type: str,
    prune_amount: float = 0.0,
    epochs: int = 3,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    torch.manual_seed(42)

    if accelerator is None:
        accelerator = Accelerator()

    target_model = CNN2()
    total_params = sum(p.numel() for p in target_model.parameters())

    if strategy_type == "direct":
        # Apply structured pruning before training if requested
        if prune_amount > 0.0:
            for _name, module in target_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name="weight", amount=prune_amount)  # type: ignore[no-untyped-call]

        train_loader, val_loader = _get_mnist_loaders()
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

        for _ in range(epochs - 1):
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

        # Effective trainable params after pruning
        trainable = int(total_params * (1.0 - prune_amount)) if prune_amount > 0.0 else total_params

        return {
            "total_params": total_params,
            "trainable_params": trainable,
            "peak_mem_mb": max(peak_mem / 1e6, 0.5),
            "epoch_time_ms": (t1 - t0) * 1000.0,
            "accuracy": correct / total,
        }

    # ---- Mapping model path -----------------------------------------------
    strategy: Any
    if strategy_type == "lrd":
        strategy = LRDStrategy(rank=8)
    else:
        strategy = SLVTStrategy(allow_large=True)

    mapping_model = MappingModel(
        target_model=target_model,
        latent_dim=2048,
        strategy=strategy,
    )

    # Apply stochastic weight masking to latent parameters if pruning requested
    if prune_amount > 0.0:
        for _name, param in mapping_model.named_parameters():
            if param.requires_grad:
                with torch.no_grad():
                    mask = torch.rand_like(param) > prune_amount
                    param.mul_(mask.float())

    mapping_loss_fn = MappingLoss(task_loss=ClassificationLoss())
    device_str = str(accelerator.device)
    config = TrainerConfig(max_epochs=epochs, learning_rate=0.01, device=device_str)

    raw_train, raw_val = _get_mnist_loaders()

    trainer = MappingTrainer(
        model=mapping_model,
        train_loader=raw_train,
        val_loader=raw_val,
        loss_fn=mapping_loss_fn,
        config=config,
    )

    trainer.fit()  # warm-up epoch

    t0 = time.perf_counter()
    _, peak_mem = profile_peak_memory(lambda: trainer.fit(), device=device_str)
    t1 = time.perf_counter()

    trainable_params = sum(p.numel() for p in mapping_model.parameters() if p.requires_grad)
    if prune_amount > 0.0:
        trainable_params = int(trainable_params * (1.0 - prune_amount))

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

    print("  [Addons] CNN2 baseline (direct)")
    results["CNN2 Baseline"] = run_addons_experiment("direct", accelerator=accelerator)

    print("  [Addons] CNN2 + Prune 90% (direct)")
    results["CNN2 + Prune 90%"] = run_addons_experiment(
        "direct", prune_amount=0.90, accelerator=accelerator
    )

    print("  [Addons] Ours* (SLVT latent=2048)")
    results["Ours* (SLVT)"] = run_addons_experiment("slvt", accelerator=accelerator)

    print("  [Addons] Ours* + LRD (rank=8)")
    results["Ours* + LRD (rank 8)"] = run_addons_experiment("lrd", accelerator=accelerator)

    print("  [Addons] Ours* + Prune 90%")
    results["Ours* + Prune 90%"] = run_addons_experiment(
        "slvt", prune_amount=0.90, accelerator=accelerator
    )

    return results


if __name__ == "__main__":
    acc = Accelerator()
    print(f"Running Add-ons benchmarks on: {acc.device}")
    res = run_benchmarks(accelerator=acc)
    for name, stats in res.items():
        print(
            f"{name}: total={stats['total_params']:,}, trainable={stats['trainable_params']:,}, "
            f"mem={stats['peak_mem_mb']:.2f}MB, time={stats['epoch_time_ms']:.1f}ms, "
            f"acc={stats['accuracy'] * 100:.2f}%"
        )
