"""Tests for frozen, stateless target execution."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from marn import ParameterTree, TargetModel, UnsupportedTargetModelError


def generated_parameters(module: nn.Module) -> ParameterTree:
    """Create leaf generated tensors matching a module for gradient assertions."""

    return ParameterTree(
        {
            name: parameter.detach().clone().requires_grad_()
            for name, parameter in module.named_parameters()
        }
    )


def state_snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def assert_state_matches(module: nn.Module, snapshot: dict[str, Tensor]) -> None:
    current = module.state_dict()
    assert current.keys() == snapshot.keys()
    for name, expected in snapshot.items():
        assert torch.equal(current[name], expected), name


def test_matches_direct_linear_execution_and_routes_generated_gradients() -> None:
    module = nn.Linear(3, 2)
    direct = copy.deepcopy(module)
    target = TargetModel(module)
    parameters = generated_parameters(module)
    inputs = torch.randn(4, 3)
    original_state = state_snapshot(module)

    output = target(parameters, inputs)
    expected = direct(inputs)
    output.sum().backward()

    assert torch.allclose(output, expected)
    assert all(not parameter.requires_grad for parameter in module.parameters())
    assert all(parameter.grad is None for parameter in module.parameters())
    assert all(parameter.grad is not None for parameter in parameters.values())
    assert_state_matches(module, original_state)


class KeywordTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)

    def forward(self, inputs: Tensor, *, scale: float, residual: Tensor) -> Tensor:
        result: Tensor = self.projection(inputs) * scale + residual
        return result


def test_supports_nested_modules_and_keyword_only_arguments() -> None:
    module = KeywordTarget()
    target = TargetModel(module)
    parameters = generated_parameters(module)
    inputs = torch.randn(2, 3)
    residual = torch.randn(2, 3)

    actual = target(parameters, inputs, scale=0.25, residual=residual)
    expected = module(inputs, scale=0.25, residual=residual)

    assert torch.allclose(actual, expected)


def test_batch_norm_training_discards_buffer_updates() -> None:
    module = nn.BatchNorm1d(3)
    module.train()
    target = TargetModel(module)
    parameters = generated_parameters(module)
    original_state = state_snapshot(module)

    output = target(parameters, torch.randn(8, 3) + 4.0)
    output.square().mean().backward()

    assert_state_matches(module, original_state)
    assert parameters["weight"].grad is not None
    assert parameters["bias"].grad is not None


def test_eval_mode_uses_frozen_running_statistics() -> None:
    module = nn.BatchNorm1d(2)

    assert module.running_mean is not None
    assert module.running_var is not None

    module.running_mean.copy_(torch.tensor([1.0, -1.0]))
    module.running_var.copy_(torch.tensor([4.0, 9.0]))
    target = TargetModel(module).eval()
    parameters = generated_parameters(module)
    inputs = torch.tensor([[3.0, 2.0]])

    actual = target(parameters, inputs)
    expected = module(inputs)

    assert torch.allclose(actual, expected)


class TiedTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 3, bias=False)
        self.decoder = nn.Linear(3, 3, bias=False)
        self.decoder.weight = self.encoder.weight


def test_rejects_tied_parameters_with_alias_names() -> None:
    with pytest.raises(
        UnsupportedTargetModelError,
        match=r"encoder\.weight = decoder\.weight",
    ):
        TargetModel(TiedTarget())


class IdentityParametrization(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value


def test_rejects_parametrized_modules_with_module_name() -> None:
    module = nn.Sequential(nn.Linear(2, 2))
    parametrize.register_parametrization(module[0], "weight", IdentityParametrization())

    with pytest.raises(UnsupportedTargetModelError, match="0"):
        TargetModel(module)


def test_rejects_incomplete_or_misshaped_parameter_trees() -> None:
    target = TargetModel(nn.Linear(2, 1))

    with pytest.raises(ValueError, match="missing=.*bias"):
        target({"weight": torch.randn(1, 2)}, torch.randn(1, 2))
    with pytest.raises(ValueError, match="Expected shape"):
        target(
            {"weight": torch.randn(2), "bias": torch.randn(1)},
            torch.randn(1, 2),
        )


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU())
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs)
        result: Tensor = self.head(features.flatten(1))
        return result


def test_torch_compile_smoke_for_nested_cnn() -> None:
    module = TinyConvNet()
    target = TargetModel(module)
    parameters = generated_parameters(module)
    inputs = torch.randn(2, 1, 4, 4)
    expected = target(parameters, inputs)

    try:
        compiled = torch.compile(target, backend="eager", fullgraph=True)
        actual = compiled(parameters, inputs)
    except RuntimeError as error:
        pytest.skip(f"torch.compile is unavailable in this runtime: {error}")

    assert torch.allclose(actual, expected)
