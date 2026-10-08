"""Tests for configuration models, registries, and YAML loading."""

from __future__ import annotations

import tempfile
from pathlib import Path
from pydantic import ValidationError

import pytest
import yaml

from marn import TaskLossConfig
from marn.config import (
    GeneratorConfig,
    LossConfig,
    MapperConfig,
    MappingConfig,
    TrainerConfig,
    load_config,
)
from marn.registry import (
    GENERATOR_REGISTRY,
    LOSS_REGISTRY,
    MAPPER_REGISTRY,
    MODULATION_REGISTRY,
    Registry,
)


# ── Default configs ──────────────────────────────────────────────────────


class TestDefaultConfigs:
    def test_mapping_config_defaults(self) -> None:
        config = MappingConfig()
        assert config.generator.strategy == "layerwise"
        assert config.generator.latent_dim == 256
        assert config.mapper.type == "mlp"
        assert config.mapper.modulation == "additive"

    def test_loss_config_defaults(self) -> None:
        config = LossConfig()
        assert config.task.type == "classification"
        assert config.enable_stability is False
        assert config.lambda_stability == 0.1

    def test_trainer_config_defaults(self) -> None:
        config = TrainerConfig()
        assert config.max_epochs == 100
        assert config.optimizer == "adam"
        assert config.learning_rate == 1e-3
        assert config.amp_enabled is False


# ── Construction from dict ───────────────────────────────────────────────


class TestConfigFromDict:
    def test_mapping_config_from_dict(self) -> None:
        config = MappingConfig(
            generator=GeneratorConfig(strategy="slvt", latent_dim=64),
            mapper=MapperConfig(type="residual_mlp", hidden_dim=32),
        )
        assert config.generator.strategy == "slvt"
        assert config.generator.latent_dim == 64
        assert config.mapper.type == "residual_mlp"
        assert config.mapper.hidden_dim == 32

    def test_loss_config_from_dict(self) -> None:
        config = LossConfig(
            enable_stability=True,
            lambda_stability=0.5,
            task=TaskLossConfig(type="regression"),
        )
        assert config.enable_stability is True
        assert config.lambda_stability == 0.5
        assert config.task.type == "regression"

    def test_trainer_config_from_dict(self) -> None:
        config = TrainerConfig(
            max_epochs=50,
            optimizer="sgd",
            learning_rate=0.01,
            gradient_clip_norm=1.0,
        )
        assert config.max_epochs == 50
        assert config.optimizer == "sgd"
        assert config.gradient_clip_norm == 1.0


# ── Validation errors ────────────────────────────────────────────────────


class TestConfigValidation:
    def test_unknown_keys_rejected(self) -> None:
        with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
            MappingConfig(unknown_field="bad")  # type: ignore[call-arg]

    def test_nested_unknown_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MappingConfig.model_validate(
                {
                    "generator": {"bad_key": 42},
                }
            )

    def test_bad_type_rejected(self) -> None:
        with pytest.raises(Exception):
            TrainerConfig(max_epochs="not_an_int")  # type: ignore[arg-type]


# ── YAML round-trip ──────────────────────────────────────────────────────


class TestYAMLRoundTrip:
    def test_serialize_and_load(self) -> None:
        original = MappingConfig(
            generator=GeneratorConfig(strategy="slvt", latent_dim=128),
            mapper=MapperConfig(type="residual_mlp", hidden_dim=64),
        )
        data = original.model_dump(mode="json")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name

        loaded = load_config(path)
        assert loaded == original
        Path(path).unlink()

    def test_load_from_dict(self) -> None:
        config = load_config({"generator": {"latent_dim": 32}})
        assert config.generator.latent_dim == 32

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config("/nonexistent/path.yaml")


# ── Registries ───────────────────────────────────────────────────────────


class TestRegistryBuiltins:
    def test_mapper_registry_has_builtins(self) -> None:
        assert "mlp" in MAPPER_REGISTRY
        assert "residual_mlp" in MAPPER_REGISTRY

    def test_modulation_registry_has_builtins(self) -> None:
        assert "additive" in MODULATION_REGISTRY
        assert "affine" in MODULATION_REGISTRY
        assert "low_rank" in MODULATION_REGISTRY

    def test_loss_registry_has_builtins(self) -> None:
        assert "classification" in LOSS_REGISTRY
        assert "regression" in LOSS_REGISTRY
        assert "stability" in LOSS_REGISTRY
        assert "smoothness" in LOSS_REGISTRY
        assert "alignment" in LOSS_REGISTRY

    def test_generator_registry_has_builtins(self) -> None:
        assert "single_vector" in GENERATOR_REGISTRY
        assert "layerwise" in GENERATOR_REGISTRY
        assert "grouped" in GENERATOR_REGISTRY


class TestRegistryCustom:
    def test_custom_registration(self) -> None:
        registry: Registry[object] = Registry("test")
        registry.register("custom", object)
        assert "custom" in registry
        assert registry.get("custom") is object

    def test_duplicate_raises(self) -> None:
        registry: Registry[object] = Registry("test")
        registry.register("item", object)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("item", object)

    def test_unknown_raises(self) -> None:
        registry: Registry[object] = Registry("test")
        with pytest.raises(KeyError, match="not registered"):
            registry.get("missing")

    def test_available_returns_sorted_keys(self) -> None:
        registry: Registry[object] = Registry("test")
        registry.register("beta", object)
        registry.register("alpha", object)
        assert registry.available() == ("alpha", "beta")


# ── Factory builds from config ───────────────────────────────────────────


class TestFactoryBuilds:
    def test_mapper_registry_returns_correct_class(self) -> None:
        from marn.mappers import MLPMapper, ResidualMLPMapper

        assert MAPPER_REGISTRY.get("mlp") is MLPMapper
        assert MAPPER_REGISTRY.get("residual_mlp") is ResidualMLPMapper

    def test_loss_registry_returns_correct_class(self) -> None:
        from marn.losses import ClassificationLoss, RegressionLoss

        assert LOSS_REGISTRY.get("classification") is ClassificationLoss
        assert LOSS_REGISTRY.get("regression") is RegressionLoss
