"""Tests for latent-conditioned modulation strategies."""

import pytest
import torch

from marn import AdditiveModulation, AffineModulation, LowRankModulation


def test_additive_matches_paper_equation_without_mutation() -> None:
    weights = torch.ones(2, 3)
    original = weights.clone()
    latent = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    result = AdditiveModulation(alpha=0.5)(weights, latent)

    assert torch.equal(result, torch.tensor([[1.5, 2.0, 2.5], [1.5, 2.0, 2.5]]))
    assert torch.equal(weights, original)
    result.sum().backward()
    assert latent.grad is not None
    assert torch.equal(latent.grad, torch.ones(3))


def test_affine_is_identity_at_zero_and_routes_gradients() -> None:
    weights = torch.randn(4, 3)
    latent = torch.zeros(3, requires_grad=True)
    result = AffineModulation(scale=0.2, shift=0.3)(weights, latent)

    assert torch.equal(result, weights)
    result.sum().backward()
    assert latent.grad is not None


def test_low_rank_decodes_factors() -> None:
    weights = torch.zeros(2, 3)
    modulation = LowRankModulation(rank=1, alpha=1.0)
    latent = torch.tensor([2.0, 3.0, 1.0, 2.0, 4.0], requires_grad=True)

    result = modulation(weights, latent)

    assert modulation.latent_dim_for(weights) == 5
    assert torch.equal(result, torch.tensor([[2.0, 4.0, 8.0], [3.0, 6.0, 12.0]]))
    result.sum().backward()
    assert latent.grad is not None


@pytest.mark.parametrize(
    "modulation",
    [AdditiveModulation(), AffineModulation()],
)
def test_column_modulation_validates_shapes(modulation: object) -> None:
    with pytest.raises(ValueError, match="input width"):
        modulation(torch.ones(2, 3), torch.ones(2))  # type: ignore[operator]
