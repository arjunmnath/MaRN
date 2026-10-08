#!/usr/bin/env python3
"""Benchmark comparing Direct training and LWT on Deepfake detection (binary classification).

Dataset: torchvision.datasets.FakeData simulating 3-channel 64×64 video frames with
binary real/fake labels — mirroring the Celeb-DF / FaceForensics++ benchmark structure.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

_DEEPFAKE_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def _get_deepfake_loaders(
    train_size: int = 512,
    val_size: int = 256,
    batch_size: int = 32,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Return dataloaders backed by FakeData (3×64×64, 2-class binary labels)."""
    train_ds = datasets.FakeData(
        size=train_size,
        image_size=(3, 64, 64),
        num_classes=2,
        transform=_DEEPFAKE_TRANSFORM,
        random_offset=0,
    )
    val_ds = datasets.FakeData(
        size=val_size,
        image_size=(3, 64, 64),
        num_classes=2,
        transform=_DEEPFAKE_TRANSFORM,
        random_offset=train_size,  # distinct split
    )
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


class CNN1Deepfake(nn.Module):
    """AlexNet-like CNN for 3-channel 64×64 binary deepfake detection."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 123),
            nn.ReLU(),
            nn.Linear(123, 2),
        )

    def forward(self, x: torch.Tensor) -> Any:
        return self.classifier(self.features(x))


class CNN2Deepfake(nn.Module):
    """LeNet-like CNN for 3-channel 64×64 binary deepfake detection."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 11),
            nn.ReLU(),
            nn.Linear(11, 2),
        )

    def forward(self, x: torch.Tensor) -> Any:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_deepfake_experiment(
    target_cls: type[nn.Module],
    strategy_name: str,
    latent_dim: int,
    epochs: int = 3,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    torch.manual_seed(42)

    if accelerator is None:
        accelerator = Accelerator()

    target_model = target_cls()
    total_params = sum(p.numel() for p in target_model.parameters())
    num_groups = 5 if target_cls == CNN1Deepfake else 4

    if strategy_name == "direct":
        trainable_params = total_params
        train_loader, val_loader = _get_deepfake_loaders()
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

        train_one_epoch()  # warm-up

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
        trainable_params = latent_dim if strategy_name == "slvt" else latent_dim * num_groups

        mapping_model = MappingModel(
            target_model=target_model,
            latent_dim=latent_dim,
            strategy=strategy_name,
        )
        mapping_loss_fn = MappingLoss(task_loss=ClassificationLoss())
        device_str = str(accelerator.device)
        config = TrainerConfig(max_epochs=epochs, learning_rate=0.01, device=device_str)

        raw_train, raw_val = _get_deepfake_loaders()

        trainer = MappingTrainer(
            model=mapping_model,
            train_loader=raw_train,
            val_loader=raw_val,
            loss_fn=mapping_loss_fn,
            config=config,
        )

        trainer.fit()  # warm-up

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

    print("  [CNN1-Deepfake] direct")
    results["cnn1_direct"] = run_deepfake_experiment(
        CNN1Deepfake, "direct", 0, accelerator=accelerator
    )

    print("  [CNN1-Deepfake] layerwise latent=391 (5 groups → 1955 total)")
    results["cnn1_lwt_1955"] = run_deepfake_experiment(
        CNN1Deepfake, "layerwise", 391, accelerator=accelerator
    )

    print("  [CNN1-Deepfake] layerwise latent=558 (5 groups → 2790 total)")
    results["cnn1_lwt_2790"] = run_deepfake_experiment(
        CNN1Deepfake, "layerwise", 558, accelerator=accelerator
    )

    print("  [CNN2-Deepfake] direct")
    results["cnn2_direct"] = run_deepfake_experiment(
        CNN2Deepfake, "direct", 0, accelerator=accelerator
    )

    print("  [CNN2-Deepfake] layerwise latent=468 (4 groups → 1872 total)")
    results["cnn2_lwt_1872"] = run_deepfake_experiment(
        CNN2Deepfake, "layerwise", 468, accelerator=accelerator
    )

    print("  [CNN2-Deepfake] layerwise latent=672 (4 groups → 2688 total)")
    results["cnn2_lwt_2688"] = run_deepfake_experiment(
        CNN2Deepfake, "layerwise", 672, accelerator=accelerator
    )

    return results


if __name__ == "__main__":
    acc = Accelerator()
    print(f"Running Deepfake benchmarks on: {acc.device}")
    res = run_benchmarks(accelerator=acc)
    for name, stats in res.items():
        print(
            f"{name}: params={stats['total_params']:,}, trainable={stats['trainable_params']:,}, "
            f"mem={stats['peak_mem_mb']:.2f}MB, time={stats['epoch_time_ms']:.1f}ms, "
            f"acc={stats['accuracy'] * 100:.2f}%"
        )
