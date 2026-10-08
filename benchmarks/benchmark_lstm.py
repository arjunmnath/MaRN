#!/usr/bin/env python3
"""Benchmark comparing Direct training vs. SLVT on LSTM for time-series forecasting.

Dataset: Deterministic multivariate sine-wave signal with additive noise.
Each input is a window of 24 timesteps × 8 features; target is the next-step value
of the first feature. This produces a real, learnable forecasting task (not pure noise).
"""

import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from marn import (
    MappingModel,
    MappingLoss,
    RegressionLoss,
    MappingTrainer,
    TrainerConfig,
    profile_peak_memory,
    SLVTStrategy,
)

from typing import Any


# ---------------------------------------------------------------------------
# Dataset helpers — deterministic sine-wave multivariate time series
# ---------------------------------------------------------------------------

_SEQ_LEN = 24  # look-back window
_N_FEATURES = 8  # number of input channels
_TRAIN_STEPS = 1024
_VAL_STEPS = 256


def _make_sine_series(n_steps: int, n_features: int, offset: int = 0) -> torch.Tensor:
    """Generate a deterministic multi-channel sine-wave series of shape (n_steps, n_features)."""
    t = torch.arange(n_steps + offset, dtype=torch.float32)
    channels = []
    for i in range(n_features):
        freq = 0.05 * (i + 1)
        phase = math.pi * i / n_features
        noise = 0.05 * torch.sin(7.3 * freq * t + phase)  # deterministic "noise"
        channels.append(torch.sin(2 * math.pi * freq * t + phase) + noise)
    return torch.stack(channels, dim=1)[offset:]  # (n_steps, n_features)


def _build_windows(series: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Slide a window over the series to build (X, y) pairs."""
    xs, ys = [], []
    for i in range(len(series) - seq_len):
        xs.append(series[i : i + seq_len])  # (seq_len, n_features)
        ys.append(series[i + seq_len, :1])  # next value of feature-0 only
    return torch.stack(xs), torch.stack(ys)


def _get_lstm_loaders(batch_size: int = 32) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_series = _make_sine_series(_TRAIN_STEPS + _SEQ_LEN, _N_FEATURES, offset=0)
    val_series = _make_sine_series(_VAL_STEPS + _SEQ_LEN, _N_FEATURES, offset=_TRAIN_STEPS)

    x_train, y_train = _build_windows(train_series, _SEQ_LEN)
    x_val, y_val = _build_windows(val_series, _SEQ_LEN)

    train_loader: DataLoader[Any] = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader: DataLoader[Any] = DataLoader(
        TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class TargetLSTM(nn.Module):
    """LSTM model for multivariate time-series next-step forecasting."""

    def __init__(
        self,
        input_size: int = _N_FEATURES,
        hidden_size: int = 50,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> Any:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last time step


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_lstm_experiment(
    strategy_name: str,
    latent_dim: int | None = None,
    epochs: int = 5,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    torch.manual_seed(42)

    if accelerator is None:
        accelerator = Accelerator()

    target_model = TargetLSTM()
    total_params = sum(p.numel() for p in target_model.parameters())

    if strategy_name == "direct":
        train_loader, val_loader = _get_lstm_loaders()
        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()

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

        # Multi-epoch training then time one more epoch
        for _ in range(epochs - 1):
            train_one_epoch()

        t0 = time.perf_counter()
        _, peak_mem = profile_peak_memory(train_one_epoch, device=str(accelerator.device))
        t1 = time.perf_counter()

        model.eval()
        val_loss_sum, total = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss_sum += loss_fn(model(x), y).item()
                total += 1

        return {
            "total_params": total_params,
            "trainable_params": total_params,
            "peak_mem_mb": max(peak_mem / 1e6, 0.5),
            "epoch_time_ms": (t1 - t0) * 1000.0,
            "val_loss": val_loss_sum / total,
        }

    assert latent_dim is not None
    strategy = SLVTStrategy(allow_large=True) if strategy_name == "slvt" else strategy_name
    mapping_model = MappingModel(
        target_model=target_model,
        latent_dim=latent_dim,
        strategy=strategy,
    )
    mapping_loss_fn = MappingLoss(task_loss=RegressionLoss())
    device_str = str(accelerator.device)
    config = TrainerConfig(max_epochs=epochs, learning_rate=0.005, device=device_str)

    raw_train, raw_val = _get_lstm_loaders()

    trainer = MappingTrainer(
        model=mapping_model,
        train_loader=raw_train,
        val_loader=raw_val,
        loss_fn=mapping_loss_fn,
        config=config,
    )

    t0 = time.perf_counter()
    _, peak_mem = profile_peak_memory(lambda: trainer.fit(), device=device_str)
    t1 = time.perf_counter()

    trainable_params = sum(p.numel() for p in mapping_model.parameters() if p.requires_grad)

    mse = nn.MSELoss()
    mapping_model.eval()
    val_loss_sum, total = 0.0, 0
    with torch.no_grad():
        for x, y in raw_val:
            x, y = x.to(device_str), y.to(device_str)
            res = mapping_model(x)
            val_loss_sum += mse(res.predictions, y).item()
            total += 1

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "peak_mem_mb": max(peak_mem / 1e6, 0.5),
        "epoch_time_ms": (t1 - t0) * 1000.0 / epochs,
        "val_loss": val_loss_sum / total,
    }


def run_benchmarks(accelerator: Accelerator | None = None) -> dict[str, Any]:
    if accelerator is None:
        accelerator = Accelerator()

    results: dict[str, Any] = {}

    print("  [LSTM] direct")
    results["lstm_direct"] = run_lstm_experiment("direct", accelerator=accelerator)

    print("  [LSTM] slvt latent=64")
    results["lstm_slvt_64"] = run_lstm_experiment("slvt", 64, accelerator=accelerator)

    print("  [LSTM] slvt latent=2048")
    results["lstm_slvt_2048"] = run_lstm_experiment("slvt", 2048, accelerator=accelerator)

    return results


if __name__ == "__main__":
    acc = Accelerator()
    print(f"Running LSTM benchmarks on: {acc.device}")
    res = run_benchmarks(accelerator=acc)
    for name, stats in res.items():
        print(
            f"{name}: total={stats['total_params']:,}, trainable={stats['trainable_params']:,}, "
            f"mem={stats['peak_mem_mb']:.2f}MB, time={stats['epoch_time_ms']:.1f}ms, "
            f"mse={stats['val_loss']:.5f}"
        )
