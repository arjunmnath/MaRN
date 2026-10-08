#!/usr/bin/env python3
"""Recipe 5: Pre-trained parameter fine-tuning via modulation.

This recipe demonstrates fine-tuning pre-trained networks using Mapping Networks
via the `FineTuningStrategy`.

Instead of generating the entire high-dimensional parameter tensor from scratch,
the mapping network generates small modulation vector coordinates that adapt the
frozen, pre-trained weights. To be highly efficient, a block size parameter L
is used: each generated modulation coordinate modulates L contiguous flattened
parameters (i.e. w_ij <- w_ij + alpha * o_i).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from marn import (
    MappingModel,
    MappingLoss,
    ClassificationLoss,
    MappingTrainer,
    TrainerConfig,
    FineTuningStrategy,
)


from typing import Any


class PretrainedTarget(nn.Module):
    """Simulated pretrained model to fine-tune on a downstream task."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> Any:
        return self.head(self.features(x))


def main() -> None:
    torch.manual_seed(42)

    # 1. Setup synthetic data representing a downstream task
    x_train = torch.randn(128, 16)
    y_train = torch.randint(0, 2, (128,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

    # 2. Instantiate and load pretrained weights (simulated)
    target = PretrainedTarget()
    # Save a snapshot of the pretrained parameters
    pretrained_params = {name: param.clone() for name, param in target.named_parameters()}

    print("--- Step 1: Instantiating FineTuningStrategy ---")
    # L=4 means each generated element modulates 4 parameters.
    # alpha=0.05 is the modulation scaling factor.
    strategy = FineTuningStrategy(
        pretrained_parameters=pretrained_params,
        L=4,
        alpha=0.05,
    )

    # Wrap target with FineTuningStrategy
    model = MappingModel(
        target_model=target,
        latent_dim=16,
        strategy=strategy,
    )

    total_target_params = sum(p.numel() for p in target.parameters())
    trainable_mapping_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Target Model Parameters: {total_target_params:,}")
    print(f"Trainable Modulation parameters (latents): {trainable_mapping_params:,}")
    print(f"PEFT Compression Ratio: {total_target_params / trainable_mapping_params:.2f}x")

    print("\n--- Step 2: Training the Fine-Tuning Module ---")
    loss_fn = MappingLoss(task_loss=ClassificationLoss())
    config = TrainerConfig(max_epochs=10, learning_rate=0.001)

    trainer = MappingTrainer(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        config=config,
    )
    trainer.fit()

    print("\n--- Step 3: Verifying Weight Modulation ---")
    # Verify that the generated weights have been adapted from original weights
    inputs = torch.randn(1, 16)
    result = model(inputs)
    generated_params = result.generated_parameters.to_dict()

    for name, pretrained_tensor in pretrained_params.items():
        generated_tensor = generated_params[name]

        # Verify that the weights are NOT identical anymore
        diff = generated_tensor - pretrained_tensor
        max_diff = diff.abs().max().item()
        print(f"Parameter '{name}' -> Max Adaptation: {max_diff:.6f}")

        assert not torch.allclose(generated_tensor, pretrained_tensor), (
            f"Weights for '{name}' were not updated!"
        )

    print("\nPEFT modulation successfully adapted the pre-trained weights.")


if __name__ == "__main__":
    main()
