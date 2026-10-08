#!/usr/bin/env python3
"""Recipe 2: Transitioning to Layer-wise Training (LWT) for deeper models.

For shallow models, SLVT is highly effective. However, as neural networks grow
deeper, generating all target weights from a single global latent vector
leads to a memory bottleneck in the fixed mapper projection buffers.

Layer-wise Training (LWT) resolves this by using independent, smaller latent
vectors for each target layer, dramatically reducing peak memory requirements
while preserving target-model expressivity.
"""

import torch
import torch.nn as nn
from marn import (
    MappingModel,
    SLVTStrategy,
)

from typing import Any


class DeepTargetCNN(nn.Module):
    """A deeper convolutional neural network with 4 conv layers and linear head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Any:
        x = self.conv_block(x)
        return self.classifier(x)


def run_experiment(strategy_name: str, latent_dim: int) -> tuple[int, int]:
    target = DeepTargetCNN()

    # Wrap using selected strategy
    strategy = SLVTStrategy(allow_large=True) if strategy_name == "slvt" else strategy_name
    model = MappingModel(
        target_model=target,
        latent_dim=latent_dim,
        strategy=strategy,
    )

    # Calculate parameter statistics
    total_target_params = sum(p.numel() for p in target.parameters())
    trainable_mapping_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_target_params, trainable_mapping_params


def main() -> None:
    torch.manual_seed(42)

    print("=== Strategy 1: Single Latent Vector Training (SLVT) ===")
    slvt_target_cnt, slvt_mapping_cnt = run_experiment("slvt", latent_dim=1024)
    print(f"SLVT Model Target Parameters: {slvt_target_cnt:,}")
    print(f"SLVT Model Trainable Latents: {slvt_mapping_cnt:,}")
    print(f"Reduction Ratio: {slvt_target_cnt / slvt_mapping_cnt:.2f}x\n")

    print("=== Strategy 2: Layer-wise Training (LWT) ===")
    lwt_target_cnt, lwt_mapping_cnt = run_experiment("layerwise", latent_dim=256)
    print(f"LWT Model Target Parameters: {lwt_target_cnt:,}")
    print(f"LWT Model Trainable Latents: {lwt_mapping_cnt:,}")
    print(f"Reduction Ratio: {lwt_target_cnt / lwt_mapping_cnt:.2f}x\n")

    print("=== Analysis & Comparison ===")
    print("In SLVT, a single mapper network maps a 1024-dim vector to ALL target parameters.")
    print("This requires a large linear transformation buffer of size [1024, total_target_params].")
    print("For deeper models, this buffer takes massive memory and is computationally expensive.")
    print("\nIn LWT, each layer has its own independent 256-dim latent vector.")
    print(
        "This distributes the mapping task across multiple smaller mappers, where each maps to a single layer's parameters."
    )
    print(
        "This significantly reduces peak memory requirements during backpropagation and forward passes."
    )


if __name__ == "__main__":
    main()
