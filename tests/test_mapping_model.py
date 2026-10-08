"""Integration tests for MappingModel with various target architectures."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from marn import (
    ForwardResult,
    LayerwiseStrategy,
    MappingModel,
)


def state_snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def assert_state_matches(module: nn.Module, snapshot: dict[str, Tensor]) -> None:
    current = module.state_dict()
    assert current.keys() == snapshot.keys()
    for name, expected in snapshot.items():
        assert torch.equal(current[name], expected), name


# ── Tiny target architectures ───────────────────────────────────────────


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU())
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs)
        result: Tensor = self.head(features.flatten(1))
        return result


class TinyResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Linear(4, 4)
        self.block2 = nn.Linear(4, 4)

    def forward(self, inputs: Tensor) -> Tensor:
        h = torch.relu(self.block1(inputs))
        return inputs + torch.relu(self.block2(h))


class KeywordTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)

    def forward(self, inputs: Tensor, *, scale: float, residual: Tensor) -> Tensor:
        result: Tensor = self.projection(inputs) * scale + residual
        return result


def _trainable_params(model: MappingModel) -> list[nn.Parameter]:
    """Return only the parameters that require gradients (latent vectors)."""
    return [p for p in model.parameters() if p.requires_grad]


# ── Forward and gradient tests ───────────────────────────────────────────


def test_linear_forward_and_gradient() -> None:
    """MappingModel forward with a Linear target returns correct result type and routes gradients."""
    target = nn.Linear(4, 2)
    model = MappingModel(target, latent_dim=8)
    inputs = torch.randn(3, 4)

    result = model(inputs)

    assert isinstance(result, ForwardResult)
    assert result.predictions.shape == (3, 2)
    assert set(result.generated_parameters) == set(name for name, _ in target.named_parameters())
    assert len(result.latent_vectors) > 0

    # Gradient flows to latents only
    result.predictions.sum().backward()
    trainable = _trainable_params(model)
    assert len(trainable) > 0
    assert all(p.grad is not None for p in trainable)
    # Target parameters must not have gradients
    assert all(p.grad is None for p in model.target.module.parameters())


def test_cnn_forward() -> None:
    """MappingModel handles Conv2d + Linear targets."""
    model = MappingModel(TinyConvNet(), latent_dim=8)
    inputs = torch.randn(2, 1, 4, 4)

    result = model(inputs)

    assert result.predictions.shape == (2, 2)
    result.predictions.sum().backward()
    assert all(p.grad is not None for p in _trainable_params(model))


def test_residual_forward() -> None:
    """MappingModel handles residual skip connections."""
    model = MappingModel(TinyResidual(), latent_dim=8)
    inputs = torch.randn(2, 4)

    result = model(inputs)

    assert result.predictions.shape == (2, 4)
    result.predictions.sum().backward()
    assert all(p.grad is not None for p in _trainable_params(model))


def test_lstm_forward() -> None:
    """MappingModel handles recurrent targets."""
    target = nn.LSTM(input_size=4, hidden_size=4, batch_first=True)
    model = MappingModel(target, latent_dim=8)
    inputs = torch.randn(2, 3, 4)  # batch=2, seq=3, features=4

    result = model(inputs)

    # LSTM returns (output, (h_n, c_n))
    output, (h_n, c_n) = result.predictions
    assert output.shape == (2, 3, 4)
    assert h_n.shape == (1, 2, 4)
    output.sum().backward()
    assert all(p.grad is not None for p in _trainable_params(model))


# ── Strategy tests ───────────────────────────────────────────────────────


def test_slvt_strategy_string() -> None:
    """MappingModel accepts 'slvt' string strategy."""
    target = nn.Linear(4, 2)
    model = MappingModel(target, latent_dim=8, strategy="slvt")
    result = model(torch.randn(2, 4))
    assert result.predictions.shape == (2, 2)


def test_custom_strategy_object() -> None:
    """MappingModel accepts a GenerationStrategy instance."""
    target = nn.Linear(4, 2)
    model = MappingModel(target, latent_dim=8, strategy=LayerwiseStrategy())
    result = model(torch.randn(2, 4))
    assert result.predictions.shape == (2, 2)


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        MappingModel(nn.Linear(2, 1), strategy="nonexistent")


def test_invalid_strategy_type_raises() -> None:
    with pytest.raises(TypeError, match="string or GenerationStrategy"):
        MappingModel(nn.Linear(2, 1), strategy=42)  # type: ignore[arg-type]


# ── Keyword arguments ───────────────────────────────────────────────────


def test_keyword_arguments_forwarded() -> None:
    """MappingModel forwards kwargs to the target model."""
    model = MappingModel(KeywordTarget(), latent_dim=8)
    inputs = torch.randn(2, 3)
    residual = torch.randn(2, 3)

    result = model(inputs, scale=0.5, residual=residual)

    assert result.predictions.shape == (2, 3)


# ── Diagnostics ──────────────────────────────────────────────────────────


def test_diagnostics() -> None:
    """Trainable count and compression ratio are consistent."""
    target = nn.Linear(10, 5)
    model = MappingModel(target, latent_dim=4)

    assert model.target_parameter_count == 55  # 10*5 + 5
    assert model.trainable_parameter_count > 0
    assert model.trainable_parameter_count < model.target_parameter_count
    assert model.compression_ratio == (
        model.target_parameter_count / model.trainable_parameter_count
    )


def test_zero_latent_dim_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        MappingModel(nn.Linear(2, 1), latent_dim=0)


# ── State immutability ───────────────────────────────────────────────────


def test_target_never_mutated() -> None:
    """Target module state is unchanged after forward and backward."""
    target = nn.Linear(4, 2)
    snapshot = state_snapshot(target)
    model = MappingModel(target, latent_dim=8)

    result = model(torch.randn(3, 4))
    result.predictions.sum().backward()

    assert_state_matches(model.target.module, snapshot)
