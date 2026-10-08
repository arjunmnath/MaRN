# Low Rank Decomposition

[View full source code on GitHub](https://github.com/arjunmnath/MaRN/blob/main/cookbook/04_low_rank_decomposition.py)

## Explanation
Recipe 4: Low-Rank Decomposition (LRD) strategy.

This recipe demonstrates the Low-Rank Decomposition (LRD) strategy. LRD
factors fully connected layers in the target model as W ≈ U @ V^T. The mapping
network generates U and V rather than the full weight matrix W. This significantly
reduces parameter requirements for wide layers.

We will train a model with LRD, extract the generated parameter weights,
and perform Singular Value Decomposition (SVD) to verify they are low-rank.

## Key Parts
```python
class LargeFCTarget(nn.Module):
    """Target model with wide fully connected layers."""

    def __init__(self) -> None:
        super().__init__()
        # 128 * 128 = 16,384 weight parameters
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> Any:
        return self.fc2(self.relu(self.fc1(x)))


def main() -> None:
    torch.manual_seed(42)

    # 1. Setup synthetic data
    x_train = torch.randn(128, 64)
    y_train = torch.randint(0, 2, (128,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

    print("--- Step 1: Comparing strategies parameter sizes ---")
    target_base = LargeFCTarget()

    # Strategy 1: Standard Layer-wise
    model_lwt = MappingModel(target_base, latent_dim=16, strategy="layerwise")
    # Strategy 2: LRD with rank 8
    # Decomposes fc1 weight [128, 64] -> U [128, 8] + V [64, 8]. Output features to generate: 8*(128+64) = 1536 (vs 8192).
    strategy_lrd = LRDStrategy(rank=8)
    model_lrd = MappingModel(LargeFCTarget(), latent_dim=16, strategy=strategy_lrd)

    target_params = sum(p.numel() for p in target_base.parameters())
    lwt_trainable = sum(p.numel() for p in model_lwt.parameters() if p.requires_grad)
    lrd_trainable = sum(p.numel() for p in model_lrd.parameters() if p.requires_grad)

    print(f"Target Parameters: {target_params:,}")
    print(f"Layer-wise (LWT) Trainable parameters: {lwt_trainable:,}")
    print(f"Low-Rank (LRD) Trainable parameters: {lrd_trainable:,}")

    print("\n--- Step 2: Training the LRD Model ---")
    loss_fn = MappingLoss(task_loss=ClassificationLoss())
    config = TrainerConfig(max_epochs=5, learning_rate=0.01)

    trainer = MappingTrainer(
        model=model_lrd,
        train_loader=train_loader,
        loss_fn=loss_fn,
        config=config,
    )
    trainer.fit()

    print("\n--- Step 3: Verifying Weight Low-Rank Properties ---")
    # Perform forward pass to generate weights
    # We inspect the generated weights for 'fc1.weight'
    inputs = torch.randn(1, 64)
    result = model_lrd(inputs)

    # Retrieve the generated parameters dict
    generated_params = result.generated_parameters.to_dict()
    fc1_weight = generated_params["fc1.weight"]

    print(f"Generated fc1.weight shape: {fc1_weight.shape}")

    # Compute Singular Value Decomposition (SVD) of the generated matrix
    # The rank is at most 8, meaning only the first 8 singular values should be significant,
    # and all subsequent singular values must be 0 (or close to numerical precision).
    U, S, V = torch.linalg.svd(fc1_weight)

    print("Singular Values of the generated weight matrix:")
    for idx, val in enumerate(S[:12]):
        print(f"  sigma_{idx + 1:02d}: {val.item():.6f}")

    # Count non-zero singular values (using threshold)
    rank = (S > 1e-4).sum().item()
    print(f"\nEmpirical Rank of generated weight matrix: {rank} (Expected: <= 8)")

    assert rank <= 8, f"Expected weight matrix rank to be <= 8, got {rank}"
    print("LRD successfully enforced the low-rank parameter constraint!")
```
