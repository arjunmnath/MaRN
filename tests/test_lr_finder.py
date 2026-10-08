"""Tests for the Learning Rate Finder."""

from __future__ import annotations

from typing import Any
import math
from unittest.mock import MagicMock, patch
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from marn import (
    MappingLoss,
    MappingModel,
    MappingTrainer,
    RegressionLoss,
    TrainerConfig,
)
from marn.trainers.lr_finder import LRFinderResult, _suggest_lr


def _make_tiny_setup(
    n_samples: int = 16,
    n_features: int = 4,
    n_outputs: int = 2,
    latent_dim: int = 8,
    batch_size: int = 4,
) -> tuple[MappingModel, DataLoader[Any]]:
    """Create a tiny target/mapping model and a training dataloader."""
    target = nn.Linear(n_features, n_outputs)
    model = MappingModel(target, latent_dim=latent_dim)
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, n_outputs)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return model, loader


def test_suggest_lr_heuristic() -> None:
    # 1. Sweep with decreasing loss, then increasing loss
    # Steepest descent is at index 2 (between 1.0 and 0.5)
    lrs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    losses = [2.0, 1.8, 1.0, 0.5, 3.0]
    smoothed_losses = [2.0, 1.8, 1.0, 0.5, 3.0]

    suggestion = _suggest_lr(lrs, losses, smoothed_losses)
    assert suggestion is not None
    # We expect suggestion to be around 1e-2 (index 2) where the gradient is steepest
    assert abs(math.log10(suggestion) - (-2)) < 1e-5

    # 2. Too short sweep (len < 5) should return first lr
    assert _suggest_lr([1e-3], [1.0], [1.0]) == 1e-3


def test_lr_finder_run_and_restore() -> None:
    model, loader = _make_tiny_setup()
    config = TrainerConfig(max_epochs=2, learning_rate=1e-3)
    trainer = MappingTrainer(
        model=model,
        train_loader=loader,
        loss_fn=MappingLoss(RegressionLoss()),
        config=config,
    )

    # Record initial state
    initial_weights = {
        k: v.clone() for k, v in model.generator.state_dict().items() if "latent_vectors" in k
    }
    initial_lr = trainer.learning_rate

    # Run LR finder
    result = trainer.lr_find(min_lr=1e-6, max_lr=1.0, num_iterations=10)

    # Check results
    assert isinstance(result, LRFinderResult)
    assert len(result.lrs) == 10
    assert len(result.losses) == 10
    assert len(result.smoothed_losses) == 10
    assert result.lrs[0] == 1e-6
    assert abs(result.lrs[-1] - 1.0) < 1e-5

    # Verify state is completely restored
    restored_weights = {
        k: v.clone() for k, v in model.generator.state_dict().items() if "latent_vectors" in k
    }
    for k in initial_weights:
        assert torch.allclose(initial_weights[k], restored_weights[k])

    assert trainer.learning_rate == initial_lr
    assert trainer.config.learning_rate == initial_lr


def test_lr_finder_divergence() -> None:
    model, loader = _make_tiny_setup()
    trainer = MappingTrainer(
        model=model,
        train_loader=loader,
        loss_fn=MappingLoss(RegressionLoss()),
        config=TrainerConfig(learning_rate=1e-3),
    )

    # Mock the train step to return exploding losses after a few steps
    original_train_step = trainer._train_step

    def mock_train_step(batch: Any, batch_idx: int) -> Any:
        loss_out = original_train_step(batch, batch_idx)
        if batch_idx >= 3:
            loss_out.total = torch.tensor(1000.0)
        return loss_out

    with patch.object(trainer, "_train_step", mock_train_step):
        result = trainer.lr_find(
            min_lr=1e-5, max_lr=10.0, num_iterations=20, divergence_threshold=2.0
        )
        # Should stop before completing all 20 iterations
        assert len(result.lrs) < 20


def test_auto_lr_find_in_fit() -> None:
    model, loader = _make_tiny_setup()
    trainer = MappingTrainer(
        model=model,
        train_loader=loader,
        loss_fn=MappingLoss(RegressionLoss()),
        config=TrainerConfig(max_epochs=1, learning_rate=1e-3),
    )

    # Set up mock lr_find that returns a suggestion
    dummy_result = LRFinderResult(
        lrs=[1e-5, 1e-4, 1e-3],
        losses=[1.0, 0.5, 2.0],
        smoothed_losses=[1.0, 0.5, 2.0],
        suggested_lr=1e-4,
    )

    with patch.object(trainer, "lr_find", return_value=dummy_result) as mock_lr_find:
        history = trainer.fit(auto_lr_find=True)
        # Check that lr_find was called
        mock_lr_find.assert_called_once()
        # Check that learning rate was updated
        assert trainer.learning_rate == 1e-4
        assert "train_loss" in history


def test_lr_finder_plot() -> None:
    result = LRFinderResult(
        lrs=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        losses=[1.0, 0.8, 0.5, 0.3, 2.0],
        smoothed_losses=[1.0, 0.8, 0.5, 0.3, 2.0],
        suggested_lr=1e-3,
    )

    # Test plot when matplotlib is mock-patched
    mock_matplotlib = MagicMock()
    mock_plt = MagicMock()
    mock_matplotlib.pyplot = mock_plt

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    with patch.dict("sys.modules", {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}):
        fig = result.plot(show=False)
        assert fig is not None


@pytest.mark.skipif(not torch.distributed.is_available(), reason="Distributed not available")
def test_lr_finder_distributed_reduction() -> None:
    import os
    from marn import setup_ddp, cleanup_ddp

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29505"
    setup_ddp(rank=0, world_size=1, backend="gloo")

    try:
        model, loader = _make_tiny_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(learning_rate=1e-3),
        )
        # Run 1 step finder
        result = trainer.lr_find(min_lr=1e-4, max_lr=1e-3, num_iterations=1)
        assert len(result.lrs) == 1
    finally:
        cleanup_ddp()


def test_lr_finder_suggestion_methods() -> None:
    result = LRFinderResult(
        lrs=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        losses=[2.0, 1.8, 1.0, 0.5, 3.0],
        smoothed_losses=[2.0, 1.8, 1.0, 0.5, 3.0],
        suggested_lr=None,
    )

    assert abs(math.log10(result.suggestion("steep")) - (-2)) < 1e-5
    assert abs(math.log10(result.suggestion("valley")) - (-3)) < 1e-5
    assert abs(math.log10(result.suggestion("minimum")) - (-2)) < 1e-5

    with pytest.raises(ValueError, match="Unknown suggestion method"):
        result.suggestion("invalid")
