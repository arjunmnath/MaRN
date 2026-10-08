"""Tests for named generated-parameter storage."""

import pytest
import torch

from marn.runtime import ParameterTree


def test_parameter_tree_preserves_order_names_and_autograd() -> None:
    weight = torch.randn(2, 3, requires_grad=True)
    tree = ParameterTree({"layer.weight": weight, "layer.bias": torch.zeros(2)})

    assert list(tree) == ["layer.weight", "layer.bias"]
    assert tree["layer.weight"] is weight
    tree["layer.weight"].sum().backward()  # type: ignore[no-untyped-call]
    assert weight.grad is not None


def test_parameter_tree_copies_input_mapping_and_exposes_copy() -> None:
    source = {"weight": torch.ones(1)}
    tree = ParameterTree(source)
    source["other"] = torch.zeros(1)

    assert list(tree) == ["weight"]
    mutable = tree.to_dict()
    mutable["other"] = torch.zeros(1)
    assert list(tree) == ["weight"]


@pytest.mark.parametrize("name", ["", 1])
def test_parameter_tree_rejects_invalid_names(name: object) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        ParameterTree({name: torch.ones(1)})  # type: ignore[dict-item]


def test_parameter_tree_rejects_non_tensor_values() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        ParameterTree({"weight": object()})  # type: ignore[dict-item]
