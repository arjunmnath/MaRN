"""Tests for MappingTrainer, callbacks, and batch adapters."""

from __future__ import annotations

import logging
from typing import Any
import pytest
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

from marn import (
    Callback,
    EarlyStopping,
    MappingLoss,
    MappingModel,
    MappingTrainer,
    MetricLogger,
    RegressionLoss,
    TrainerConfig,
    TupleBatchAdapter,
)
from marn.trainers.batch_adapter import MappingBatchAdapter


loader: DataLoader[tuple[Tensor, Tensor]]

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_regression_setup(
    n_samples: int = 16,
    n_features: int = 4,
    n_outputs: int = 2,
    latent_dim: int = 8,
    batch_size: int = 4,
) -> tuple[
    MappingModel,
    DataLoader[tuple[Tensor, ...]],
    DataLoader[tuple[Tensor, ...]],
]:
    """Create a tiny regression model, train, and val loaders."""
    target = nn.Linear(n_features, n_outputs)
    model = MappingModel(target, latent_dim=latent_dim)

    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, n_outputs)

    half = n_samples // 2
    train_ds = TensorDataset(X[:half], y[:half])
    val_ds = TensorDataset(X[half:], y[half:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return model, train_loader, val_loader


def _make_classification_setup(
    n_samples: int = 16,
    n_features: int = 4,
    n_classes: int = 3,
    latent_dim: int = 8,
    batch_size: int = 4,
) -> tuple[MappingModel, DataLoader[tuple[Tensor, ...]]]:
    target = nn.Linear(n_features, n_classes)
    model = MappingModel(target, latent_dim=latent_dim)

    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))

    train_ds = TensorDataset(X, y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    return model, train_loader


# ── Overfit tests ────────────────────────────────────────────────────────


class TestOverfit:
    def test_overfit_tiny_linear(self) -> None:
        """Loss decreases over epochs on a tiny 4-sample linear regression."""
        model, train_loader, _ = _make_regression_setup(n_samples=8, batch_size=4)
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=30, learning_rate=0.01),
        )
        history = trainer.fit()

        losses = history["train_loss"]
        assert len(losses) == 30
        # Loss should decrease over training
        assert losses[-1] < losses[0], (
            f"Expected loss to decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_overfit_tiny_cnn(self) -> None:
        """Loss decreases over epochs on a tiny CNN."""
        target = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8, 2))
        model = MappingModel(target, latent_dim=8)

        X = torch.randn(8, 1, 4, 4)
        y = torch.randn(8, 2)
        train_ds = TensorDataset(X, y)
        train_loader = DataLoader(train_ds, batch_size=4)

        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=30, learning_rate=0.01),
        )
        history = trainer.fit()

        assert history["train_loss"][-1] < history["train_loss"][0]


# ── Validation ───────────────────────────────────────────────────────────


class TestValidation:
    def test_validation_metrics_returned(self) -> None:
        model, train_loader, val_loader = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=3),
        )
        history = trainer.fit()

        assert "val_loss" in history
        assert len(history["val_loss"]) == 3
        assert all(v >= 0 for v in history["val_loss"])


# ── Early stopping ───────────────────────────────────────────────────────


class TestEarlyStopping:
    def test_early_stopping_triggers(self) -> None:
        """EarlyStopping stops training before max_epochs."""
        model, train_loader, val_loader = _make_regression_setup(n_samples=8, batch_size=4)
        # High LR so training overshoots and val loss stops improving quickly
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=200, learning_rate=0.1),
            callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
        )
        history = trainer.fit()

        # Should stop well before 200 epochs
        assert len(history["train_loss"]) < 200

    def test_early_stopping_mode_max(self) -> None:
        callback = EarlyStopping(mode="max", patience=2)
        assert callback.mode == "max"

    def test_early_stopping_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="min.*max"):
            EarlyStopping(mode="invalid")


# ── MetricLogger ─────────────────────────────────────────────────────────


class TestMetricLogger:
    def test_logger_produces_output(self, caplog: pytest.LogCaptureFixture) -> None:
        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=2),
            callbacks=[MetricLogger(log_every_n_batches=1)],
        )

        with caplog.at_level(logging.INFO, logger="marn.trainer"):
            trainer.fit()

        # Should have logged epoch start + batch + epoch end messages
        assert len(caplog.records) > 0
        messages = [r.message for r in caplog.records]
        assert any("Epoch" in m for m in messages)


# ── Gradient clipping ────────────────────────────────────────────────────


class TestGradientClipping:
    def test_gradient_clip_norm(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=2, gradient_clip_norm=0.5),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 2

    def test_gradient_clip_value(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=2, gradient_clip_value=0.1),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 2


# ── Gradient accumulation ────────────────────────────────────────────────


class TestGradientAccumulation:
    def test_accumulation_steps(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=3, accumulation_steps=2),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 3


# ── Batch adapters ───────────────────────────────────────────────────────


class TestBatchAdapters:
    def test_tuple_adapter(self) -> None:
        adapter = TupleBatchAdapter()
        batch = (torch.randn(4, 3), torch.randint(0, 2, (4,)))
        inputs, targets = adapter.unpack(batch)
        assert len(inputs) == 1
        assert inputs[0].shape == (4, 3)
        assert targets.shape == (4,)

    def test_tuple_adapter_extra_elements(self) -> None:
        adapter = TupleBatchAdapter()
        batch = (torch.randn(4, 3), torch.randint(0, 2, (4,)), "extra")
        inputs, targets = adapter.unpack(batch)
        assert len(inputs) == 1

    def test_tuple_adapter_invalid_raises(self) -> None:
        adapter = TupleBatchAdapter()
        with pytest.raises(ValueError, match="length >= 2"):
            adapter.unpack((torch.randn(4, 3),))

    def test_mapping_adapter(self) -> None:
        adapter = MappingBatchAdapter(input_key="image", target_key="label")
        batch = {"image": torch.randn(4, 3), "label": torch.randint(0, 2, (4,))}
        inputs, targets = adapter.unpack(batch)
        assert len(inputs) == 1
        assert inputs[0].shape == (4, 3)

    def test_mapping_adapter_missing_key_raises(self) -> None:
        adapter = MappingBatchAdapter(input_key="image", target_key="label")
        with pytest.raises(KeyError, match="image"):
            adapter.unpack({"data": torch.randn(4, 3)})

    def test_mapping_adapter_not_dict_raises(self) -> None:
        adapter = MappingBatchAdapter()
        with pytest.raises(ValueError, match="dict"):
            adapter.unpack((torch.randn(4, 3), torch.randn(4)))


# ── Custom optimizer ─────────────────────────────────────────────────────


class TestCustomOptimizer:
    def test_sgd_works(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=3, optimizer="sgd", learning_rate=0.01),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 3

    def test_adamw_works(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=3, optimizer="adamw", weight_decay=0.01),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 3

    def test_unknown_optimizer_raises(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        with pytest.raises(ValueError, match="Unknown optimizer"):
            MappingTrainer(
                model=model,
                train_loader=train_loader,
                config=TrainerConfig(optimizer="nonexistent"),
            )


# ── Scheduler ────────────────────────────────────────────────────────────


class TestScheduler:
    def test_cosine_scheduler(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=5, scheduler="cosine"),
        )
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        history = trainer.fit()
        final_lr = trainer.optimizer.param_groups[0]["lr"]

        assert len(history["train_loss"]) == 5
        # Cosine should reduce LR over epochs
        assert final_lr < initial_lr

    def test_step_scheduler(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=5, scheduler="step"),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 5

    def test_plateau_scheduler(self) -> None:
        model, train_loader, val_loader = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=MappingLoss(RegressionLoss()),
            config=TrainerConfig(max_epochs=5, scheduler="plateau"),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 5

    def test_unknown_scheduler_raises(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        with pytest.raises(ValueError, match="Unknown scheduler"):
            MappingTrainer(
                model=model,
                train_loader=train_loader,
                config=TrainerConfig(scheduler="nonexistent"),
            )


# ── Callback error handling ──────────────────────────────────────────────


class TestCallbackErrors:
    def test_failing_callback_includes_name(self) -> None:
        class BrokenCallback(Callback):
            def on_epoch_start(self, _trainer: Any, epoch: int) -> None:
                raise ValueError("intentional error")

        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            config=TrainerConfig(max_epochs=1),
            callbacks=[BrokenCallback()],
        )
        with pytest.raises(RuntimeError, match="BrokenCallback.*on_epoch_start"):
            trainer.fit()


# ── Interruption ─────────────────────────────────────────────────────────


class TestInterruption:
    def test_keyboard_interrupt_returns_partial_history(self) -> None:
        class InterruptCallback(Callback):
            def on_epoch_end(self, _trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
                if epoch >= 2:
                    raise KeyboardInterrupt

        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            config=TrainerConfig(max_epochs=100),
            callbacks=[InterruptCallback()],
        )
        history = trainer.fit()

        # Should have partial history (3 epochs: 0, 1, 2)
        assert 0 < len(history["train_loss"]) < 100


# ── Reproducibility ──────────────────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_history(self) -> None:
        def _run(seed: int) -> list[float]:
            torch.manual_seed(seed)
            target = nn.Linear(4, 2)
            model = MappingModel(target, latent_dim=8)
            X = torch.randn(8, 4)
            y = torch.randn(8, 2)
            train_ds = TensorDataset(X, y)
            loader = DataLoader(train_ds, batch_size=4, shuffle=False)
            trainer = MappingTrainer(
                model=model,
                train_loader=loader,
                loss_fn=MappingLoss(RegressionLoss()),
                config=TrainerConfig(max_epochs=5, seed=seed),
            )
            loss_list = trainer.fit()["train_loss"]
            assert isinstance(loss_list, list)
            return loss_list

        run1 = _run(42)
        run2 = _run(42)
        assert run1 == pytest.approx(run2, abs=1e-5)


# ── Default loss ─────────────────────────────────────────────────────────


class TestDefaultLoss:
    def test_trainer_works_without_explicit_loss(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            config=TrainerConfig(max_epochs=2),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 2

    def test_base_loss_wrapped_in_mapping_loss(self) -> None:
        model, train_loader, _ = _make_regression_setup()
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=RegressionLoss(),
            config=TrainerConfig(max_epochs=2),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 2


# ── Loss System Integration ──────────────────────────────────────────────


class TestTrainerLossSystemIntegration:
    def test_stability_loss_integration(self) -> None:
        from marn import StabilityLoss

        model, train_loader, _ = _make_regression_setup(n_samples=8, batch_size=4)
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=StabilityLoss(epsilon=0.01, num_samples=2),
            lambda_stability=0.1,
        )
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            config=TrainerConfig(max_epochs=3, learning_rate=0.01),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 3

    def test_alignment_loss_integration(self) -> None:
        from marn import AlignmentLoss

        model, train_loader, _ = _make_regression_setup(n_samples=8, batch_size=4)
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            alignment_loss=AlignmentLoss(),
            lambda_alignment=0.1,
        )
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            config=TrainerConfig(max_epochs=3, learning_rate=0.01),
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 3

    def test_trainable_loss_coefficients_integration(self) -> None:
        from marn import StabilityLoss, AlignmentLoss

        model, train_loader, _ = _make_regression_setup(n_samples=8, batch_size=4)
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=StabilityLoss(epsilon=0.01),
            alignment_loss=AlignmentLoss(),
            lambda_stability=0.1,
            lambda_alignment=0.1,
            trainable_coefficients=True,
        )
        trainer = MappingTrainer(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            config=TrainerConfig(max_epochs=3, learning_rate=0.1),
        )

        # Store original parameter values
        original_stability = loss_fn._raw_lambda_stability.item()
        original_alignment = loss_fn._raw_lambda_alignment.item()

        trainer.fit()

        # Verify they have changed after optimization
        assert loss_fn._raw_lambda_stability.item() != original_stability
        assert loss_fn._raw_lambda_alignment.item() != original_alignment
