#!/usr/bin/env python3
"""Recipe 7: Advanced Custom Components (Mappers, Modulations, and Callbacks).

This recipe demonstrates the extensible architecture of MaRN.
We will:
1. Implement a custom training callback (`WeightSnapshotCallback`) that monitors
   the generated weights of a specific layer across epochs.
2. Train the target model using these custom extensions.
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
    Callback,
)


from typing import Any


# 1. Define a Custom Callback
class WeightSnapshotCallback(Callback):
    """Callback to monitor parameter generation dynamics by snapshotting weights."""

    def __init__(self, param_name: str) -> None:
        super().__init__()
        self.param_name = param_name
        self.snapshots: list[float] = []

    def on_epoch_end(
        self, trainer: MappingTrainer, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        # Generate parameters using a sample forward pass
        sample_input = torch.randn(1, 10).to(trainer.device)
        with torch.no_grad():
            res = trainer.model(sample_input)
            generated_dict = res.generated_parameters.to_dict()
            if self.param_name in generated_dict:
                weight_norm = torch.norm(generated_dict[self.param_name]).item()
                self.snapshots.append(weight_norm)
                print(
                    f"[{epoch + 1:02d}] Weight norm snapshot of '{self.param_name}': {weight_norm:.4f}"
                )


class TargetNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> Any:
        return self.fc2(self.relu(self.fc1(x)))


def main() -> None:
    torch.manual_seed(42)

    # 1. Setup synthetic dataset
    x_train = torch.randn(128, 10)
    y_train = torch.randint(0, 2, (128,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

    # 2. Instantiate model
    target = TargetNetwork()
    model = MappingModel(
        target_model=target,
        latent_dim=16,
        strategy="layerwise",
    )

    from marn.generators.grouped import GroupedGenerator
    generator = model.generator
    assert isinstance(generator, GroupedGenerator)

    print("\n--- Training model using custom callback ---")
    loss_fn = MappingLoss(task_loss=ClassificationLoss())
    config = TrainerConfig(max_epochs=4, learning_rate=0.01)

    # Snapshot fc1.weight parameter evolution
    snapshot_callback = WeightSnapshotCallback(param_name="fc1.weight")

    trainer = MappingTrainer(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        config=config,
        callbacks=[snapshot_callback],
    )
    trainer.fit()

    print(
        "\nCustom components successfully registered, initialized, and evaluated during training!"
    )


if __name__ == "__main__":
    main()
