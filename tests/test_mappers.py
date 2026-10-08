"""Tests for fixed mapper implementations."""

import pytest
import torch

from marn import AdditiveModulation, MLPMapper, ResidualMLPMapper


def test_mlp_mapper_has_only_buffers_and_latent_receives_gradient() -> None:
    mapper = MLPMapper(4, 6, modulation=AdditiveModulation(alpha=0.1))
    latent = torch.randn(4, requires_grad=True)

    output = mapper(latent)
    output.square().sum().backward()

    assert output.shape == (6,)
    assert list(mapper.parameters()) == []
    assert {name for name, _ in mapper.named_buffers()} == {"weight_0", "bias_0"}
    assert latent.grad is not None


def test_deep_mlp_mapper_dimensions_and_determinism() -> None:
    torch.manual_seed(12)
    mapper = MLPMapper(3, 2, hidden_dims=(5, 4), modulation=AdditiveModulation())
    latent = torch.randn(3)

    first = mapper(latent)
    second = mapper(latent)

    assert first.shape == (2,)
    assert torch.equal(first, second)


def test_mlp_expands_only_final_projection_variance() -> None:
    mapper = MLPMapper(3, 13, hidden_dims=(7,))

    assert torch.linalg.matrix_norm(mapper.weight_0).square().item() == pytest.approx(3)
    assert torch.linalg.matrix_norm(mapper.weight_1).square().item() == pytest.approx(7 * 13)


def test_residual_mapper_has_fixed_buffers_and_correct_output() -> None:
    mapper = ResidualMLPMapper(3, 7, hidden_dim=5, depth=3, modulation=AdditiveModulation())
    latent = torch.randn(3, requires_grad=True)

    output = mapper(latent)
    output.sum().backward()

    assert output.shape == (7,)
    assert list(mapper.parameters()) == []
    assert len(list(mapper.buffers())) == 10
    assert latent.grad is not None


def test_residual_mapper_expands_final_projection_variance() -> None:
    mapper = ResidualMLPMapper(3, 13, hidden_dim=7)

    assert torch.linalg.matrix_norm(mapper.input_weight).square().item() == pytest.approx(3)
    assert torch.linalg.matrix_norm(mapper.output_weight).square().item() == pytest.approx(7 * 13)


def test_mapper_rejects_incompatible_latent() -> None:
    mapper = MLPMapper(3, 2)
    with pytest.raises(ValueError, match="3 values"):
        mapper(torch.randn(2))


def test_mapper_state_dict_round_trip_preserves_output() -> None:
    source = MLPMapper(3, 2, hidden_dims=(4,))
    destination = MLPMapper(3, 2, hidden_dims=(4,))
    destination.load_state_dict(source.state_dict())
    latent = torch.randn(3)

    assert torch.equal(source(latent), destination(latent))
