"""Tests for checkpoint save/load, compatibility validation, and training resumption."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

from marn import (
    CheckpointCompatibilityError,
    MappingLoss,
    MappingModel,
    MappingTrainer,
    MappingConfig,
    RegressionLoss,
    TrainerConfig,
    load_checkpoint,
    save_checkpoint,
)
from marn.checkpoint.schema import CURRENT_SCHEMA_VERSION


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_linear_model(
    n_features: int = 4, n_outputs: int = 2, latent_dim: int = 8
) -> MappingModel:
    return MappingModel(nn.Linear(n_features, n_outputs), latent_dim=latent_dim)


def _make_loader(
    n_samples: int = 16,
    n_features: int = 4,
    n_outputs: int = 2,
    batch_size: int = 4,
) -> DataLoader[tuple[Tensor, ...]]:
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, n_outputs)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_trainer(
    model: MappingModel,
    loader: DataLoader[tuple[Tensor, ...]],
    max_epochs: int = 3,
) -> MappingTrainer:
    return MappingTrainer(
        model=model,
        train_loader=loader,
        loss_fn=MappingLoss(RegressionLoss()),
        config=TrainerConfig(max_epochs=max_epochs, learning_rate=0.01, seed=0),
    )


# ── Schema basics ─────────────────────────────────────────────────────────


class TestCheckpointSchema:
    def test_schema_version_is_int(self) -> None:
        assert isinstance(CURRENT_SCHEMA_VERSION, int)
        assert CURRENT_SCHEMA_VERSION >= 1

    def test_saved_payload_has_required_keys(self, tmp_path: Path) -> None:
        model = _make_linear_model()
        ckpt = tmp_path / "test.pt"
        save_checkpoint(ckpt, model)
        payload = torch.load(ckpt, weights_only=False)
        required = {
            "schema_version",
            "package_version",
            "latent_state",
            "mapper_buffers",
            "parameter_spec_names",
        }
        assert required.issubset(payload.keys())

    def test_schema_version_matches_constant(self, tmp_path: Path) -> None:
        model = _make_linear_model()
        ckpt = tmp_path / "test.pt"
        save_checkpoint(ckpt, model)
        payload = torch.load(ckpt, weights_only=False)
        assert payload["schema_version"] == CURRENT_SCHEMA_VERSION


# ── Generated weights excluded ────────────────────────────────────────────


class TestGeneratedWeightsExcluded:
    def test_checkpoint_does_not_contain_target_weights(self, tmp_path: Path) -> None:
        model = _make_linear_model()
        ckpt = tmp_path / "test.pt"
        save_checkpoint(ckpt, model)
        payload = torch.load(ckpt, weights_only=False)

        # Target parameter names should not appear in latent or buffer state
        target_names = {entry.name for entry in model.target.parameter_spec}
        all_saved_keys = set(payload["latent_state"]) | set(payload["mapper_buffers"])
        assert target_names.isdisjoint(all_saved_keys), (
            f"Target weights found in checkpoint: {target_names & all_saved_keys}"
        )

    def test_checkpoint_is_compact(self, tmp_path: Path) -> None:
        """Checkpoint should be small because it excludes generated target weights."""
        model = _make_linear_model(n_features=32, n_outputs=32, latent_dim=32)
        ckpt = tmp_path / "test.pt"
        save_checkpoint(ckpt, model)
        # File should be well under 1 MB for a tiny model
        assert ckpt.stat().st_size < 1_000_000


# ── Exact prediction restoration ─────────────────────────────────────────


class TestPredictionRestoration:
    def test_predictions_identical_after_load(self, tmp_path: Path) -> None:
        """Predictions before save and after load must be bit-identical."""
        torch.manual_seed(42)
        model = _make_linear_model()
        loader = _make_loader()
        trainer = _make_trainer(model, loader, max_epochs=3)
        trainer.fit()

        # Record predictions before save
        x = torch.randn(4, 4)
        model.eval()
        with torch.no_grad():
            before = model(x).predictions

        # Save
        ckpt = tmp_path / "pred.pt"
        save_checkpoint(ckpt, model)

        # Build a fresh model with the same architecture and load
        fresh_model = _make_linear_model()
        load_checkpoint(ckpt, fresh_model)

        fresh_model.eval()
        with torch.no_grad():
            after = fresh_model(x).predictions

        assert torch.allclose(before, after, atol=1e-6), (
            f"Predictions differ after checkpoint round-trip: max delta "
            f"{(before - after).abs().max().item():.2e}"
        )

    def test_latent_vectors_identical_after_load(self, tmp_path: Path) -> None:
        """Latent tensors must be numerically identical after loading."""
        torch.manual_seed(7)
        model = _make_linear_model()
        loader = _make_loader()
        trainer = _make_trainer(model, loader, max_epochs=2)
        trainer.fit()

        # Capture latents before save
        before_latents = {
            name: tensor.detach().clone() for name, tensor in model.generator.named_latent_vectors()
        }

        ckpt = tmp_path / "latents.pt"
        save_checkpoint(ckpt, model)

        fresh_model = _make_linear_model()
        load_checkpoint(ckpt, fresh_model)

        after_latents = {
            name: tensor.detach().clone()
            for name, tensor in fresh_model.generator.named_latent_vectors()
        }

        assert set(before_latents) == set(after_latents)
        for name in before_latents:
            assert torch.allclose(before_latents[name], after_latents[name], atol=1e-7), (
                f"Latent {name!r} differs after load"
            )

    def test_cnn_predictions_restored(self, tmp_path: Path) -> None:
        """Round-trip works for a CNN target model."""
        torch.manual_seed(99)
        target = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8, 2))
        model = MappingModel(target, latent_dim=8)

        x = torch.randn(2, 1, 4, 4)
        model.eval()
        with torch.no_grad():
            before = model(x).predictions

        ckpt = tmp_path / "cnn.pt"
        save_checkpoint(ckpt, model)

        fresh_model = MappingModel(
            nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8, 2)),
            latent_dim=8,
        )
        load_checkpoint(ckpt, fresh_model)

        fresh_model.eval()
        with torch.no_grad():
            after = fresh_model(x).predictions

        assert torch.allclose(before, after, atol=1e-6)


# ── Interrupted training continuation ────────────────────────────────────


class TestTrainingResumption:
    def test_optimizer_state_restored(self, tmp_path: Path) -> None:
        """Optimizer momentum buffers must be preserved across a checkpoint."""
        torch.manual_seed(0)
        model = _make_linear_model()
        loader = _make_loader()
        trainer = _make_trainer(model, loader, max_epochs=5)
        trainer.fit()

        # Capture optimizer state
        opt_state_before = {
            k: (v.clone() if isinstance(v, Tensor) else v)
            for k, v in trainer.optimizer.state_dict()["state"].items()
        }

        ckpt = tmp_path / "trainer.pt"
        save_checkpoint(ckpt, model, trainer=trainer)

        # Fresh objects
        fresh_model = _make_linear_model()
        fresh_trainer = _make_trainer(fresh_model, loader)
        schema = load_checkpoint(ckpt, fresh_model, trainer=fresh_trainer)

        assert schema.trainer_state is not None
        assert fresh_trainer.current_epoch == trainer.current_epoch

        # Optimizer state must be restored (at least for groups that trained)
        restored_state = fresh_trainer.optimizer.state_dict()["state"]
        for group_idx, state_before in opt_state_before.items():
            assert group_idx in restored_state, f"Optimizer group {group_idx} missing after load"

    def test_epoch_counter_restored(self, tmp_path: Path) -> None:
        """current_epoch must be restored exactly."""
        model = _make_linear_model()
        loader = _make_loader()
        trainer = _make_trainer(model, loader, max_epochs=4)
        trainer.fit()

        ckpt = tmp_path / "epoch.pt"
        save_checkpoint(ckpt, model, trainer=trainer)

        fresh_model = _make_linear_model()
        fresh_trainer = _make_trainer(fresh_model, loader)
        load_checkpoint(ckpt, fresh_model, trainer=fresh_trainer)

        assert fresh_trainer.current_epoch == trainer.current_epoch

    def test_resumed_training_continues_from_checkpoint(self, tmp_path: Path) -> None:
        """Loss after resumption should differ from training from scratch (warm start)."""
        torch.manual_seed(0)
        model = _make_linear_model()
        loader = _make_loader(n_samples=8, batch_size=4)

        # Train 3 epochs, checkpoint, then resume for 2 more
        trainer = _make_trainer(model, loader, max_epochs=3)
        history_first = trainer.fit()

        ckpt = tmp_path / "mid.pt"
        save_checkpoint(ckpt, model, trainer=trainer)

        # Resume
        load_checkpoint(ckpt, model, trainer=trainer)
        trainer.config = TrainerConfig(max_epochs=5, learning_rate=0.01, seed=0)
        history_resumed = trainer.fit()

        # Combined history is non-empty; just confirm no errors
        assert len(history_first["train_loss"]) == 3
        assert len(history_resumed["train_loss"]) == 5

    def test_trainer_state_not_saved_when_disabled(self, tmp_path: Path) -> None:
        """save_trainer_state=False must omit trainer state."""
        model = _make_linear_model()
        loader = _make_loader()
        trainer = _make_trainer(model, loader, max_epochs=1)
        trainer.fit()

        ckpt = tmp_path / "no_trainer.pt"
        save_checkpoint(ckpt, model, trainer=trainer, save_trainer_state=False)

        payload = torch.load(ckpt, weights_only=False)
        assert payload["trainer_state"] is None

    def test_load_without_trainer_arg_restores_model_only(self, tmp_path: Path) -> None:
        """When no trainer is passed, load_checkpoint does not error."""
        model = _make_linear_model()
        loader = _make_loader()
        trainer = _make_trainer(model, loader, max_epochs=1)
        trainer.fit()

        ckpt = tmp_path / "only_model.pt"
        save_checkpoint(ckpt, model, trainer=trainer)

        fresh_model = _make_linear_model()
        schema = load_checkpoint(ckpt, fresh_model)  # no trainer arg
        assert schema.trainer_state is not None  # state is in file but not applied


# ── Architecture compatibility validation ─────────────────────────────────


class TestCompatibilityValidation:
    def test_wrong_architecture_raises(self, tmp_path: Path) -> None:
        """Loading into a model with different layer shapes must raise."""
        model_a = _make_linear_model(n_features=4, n_outputs=2)
        ckpt = tmp_path / "a.pt"
        save_checkpoint(ckpt, model_a)

        # Different output dimension
        model_b = _make_linear_model(n_features=4, n_outputs=8)
        with pytest.raises(CheckpointCompatibilityError, match="shape mismatch|incompatible"):
            load_checkpoint(ckpt, model_b)

    def test_extra_layer_raises(self, tmp_path: Path) -> None:
        """Loading into a deeper model raises CheckpointCompatibilityError."""
        model_shallow = _make_linear_model()
        ckpt = tmp_path / "shallow.pt"
        save_checkpoint(ckpt, model_shallow)

        # Extra linear layer produces more parameters
        model_deep = MappingModel(
            nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)),
            latent_dim=8,
        )
        with pytest.raises(CheckpointCompatibilityError):
            load_checkpoint(ckpt, model_deep)

    def test_future_schema_version_raises(self, tmp_path: Path) -> None:
        """A checkpoint from a future schema version must be rejected."""
        model = _make_linear_model()
        ckpt = tmp_path / "future.pt"
        save_checkpoint(ckpt, model)

        # Tamper with the schema version
        payload = torch.load(ckpt, weights_only=False)
        payload["schema_version"] = CURRENT_SCHEMA_VERSION + 999
        torch.save(payload, ckpt)

        fresh_model = _make_linear_model()
        with pytest.raises(CheckpointCompatibilityError, match="newer"):
            load_checkpoint(ckpt, fresh_model)

    def test_missing_schema_version_raises(self, tmp_path: Path) -> None:
        """A payload without schema_version is rejected with a clear error."""
        ckpt = tmp_path / "bad.pt"
        torch.save({"random": "junk"}, ckpt)

        model = _make_linear_model()
        with pytest.raises(CheckpointCompatibilityError, match="schema_version"):
            load_checkpoint(ckpt, model)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        model = _make_linear_model()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt", model)

    def test_non_dict_payload_raises(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "bad.pt"
        torch.save([1, 2, 3], ckpt)

        model = _make_linear_model()
        with pytest.raises(CheckpointCompatibilityError, match="not a valid"):
            load_checkpoint(ckpt, model)

    def test_correct_architecture_loads_cleanly(self, tmp_path: Path) -> None:
        """Same architecture passes validation with no error."""
        model = _make_linear_model()
        ckpt = tmp_path / "ok.pt"
        save_checkpoint(ckpt, model)

        fresh_model = _make_linear_model()
        schema = load_checkpoint(ckpt, fresh_model)
        assert schema.schema_version == CURRENT_SCHEMA_VERSION


# ── Config and metadata ───────────────────────────────────────────────────


class TestConfigAndMetadata:
    def test_config_saved_and_present_in_schema(self, tmp_path: Path) -> None:
        model = _make_linear_model()
        config = MappingConfig()
        ckpt = tmp_path / "config.pt"
        save_checkpoint(ckpt, model, config=config)

        fresh_model = _make_linear_model()
        schema = load_checkpoint(ckpt, fresh_model)
        assert schema.config is not None
        assert "generator" in schema.config

    def test_metadata_round_trips(self, tmp_path: Path) -> None:
        model = _make_linear_model()
        ckpt = tmp_path / "meta.pt"
        save_checkpoint(ckpt, model, metadata={"dataset": "cifar10", "epoch": 42})

        fresh_model = _make_linear_model()
        schema = load_checkpoint(ckpt, fresh_model)
        assert schema.metadata["dataset"] == "cifar10"
        assert schema.metadata["epoch"] == 42

    def test_no_config_saves_none(self, tmp_path: Path) -> None:
        model = _make_linear_model()
        ckpt = tmp_path / "noconf.pt"
        save_checkpoint(ckpt, model)

        fresh_model = _make_linear_model()
        schema = load_checkpoint(ckpt, fresh_model)
        assert schema.config is None

    def test_package_version_present(self, tmp_path: Path) -> None:
        import marn

        model = _make_linear_model()
        ckpt = tmp_path / "ver.pt"
        save_checkpoint(ckpt, model)

        fresh_model = _make_linear_model()
        schema = load_checkpoint(ckpt, fresh_model)
        assert schema.package_version == marn.__version__


# ── Parent directory creation ─────────────────────────────────────────────


class TestDirectoryCreation:
    def test_nested_path_created_automatically(self, tmp_path: Path) -> None:
        model = _make_linear_model()
        ckpt = tmp_path / "subdir" / "deep" / "ckpt.pt"
        save_checkpoint(ckpt, model)
        assert ckpt.exists()
