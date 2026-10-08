"""Tests for compiled parameter metadata and reconstruction."""

import pytest
import torch
from torch import nn

from marn.runtime import ParameterEntry, ParameterSpec, ParameterTree


def test_spec_from_module_records_shapes_offsets_and_total() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1, bias=False))
    spec = ParameterSpec.from_module(model)

    assert spec.names == ("0.weight", "0.bias", "1.weight")
    assert [entry.shape for entry in spec] == [
        torch.Size([2, 3]),
        torch.Size([2]),
        torch.Size([1, 2]),
    ]
    assert [(entry.start, entry.stop) for entry in spec] == [(0, 6), (6, 8), (8, 10)]
    assert spec.total_numel == 10
    assert spec.entry("0.bias").slice_range == slice(6, 8)


def test_unflatten_returns_shaped_views_and_preserves_gradients() -> None:
    model = nn.Linear(3, 2)
    spec = ParameterSpec.from_module(model)
    vector = torch.arange(spec.total_numel, dtype=torch.float32, requires_grad=True)

    tree = spec.unflatten(vector)

    assert isinstance(tree, ParameterTree)
    assert tree["weight"].shape == (2, 3)
    assert tree["bias"].shape == (2,)
    assert tree["weight"].untyped_storage().data_ptr() == vector.untyped_storage().data_ptr()
    loss = torch.stack([t.sum() for t in tree.values()]).sum()
    loss.backward()  # type: ignore[no-untyped-call]
    assert vector.grad is not None
    assert torch.equal(vector.grad, torch.ones_like(vector))


def test_flatten_round_trip_uses_spec_order_not_mapping_order() -> None:
    spec = ParameterSpec.from_named_tensors(
        [("weight", torch.empty(2, 2)), ("bias", torch.empty(2))]
    )
    tree = ParameterTree(
        {"bias": torch.tensor([5.0, 6.0]), "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    )

    vector = spec.flatten(tree)

    assert torch.equal(vector, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    rebuilt = spec.unflatten(vector)
    assert torch.equal(rebuilt["weight"], tree["weight"])
    assert torch.equal(rebuilt["bias"], tree["bias"])


@pytest.mark.parametrize(
    ("vector", "message"),
    [(torch.zeros(2, 2), "1D vector"), (torch.zeros(3), "Expected 4 values")],
)
def test_unflatten_rejects_incompatible_vectors(vector: torch.Tensor, message: str) -> None:
    spec = ParameterSpec.from_named_tensors([("weight", torch.empty(4))])
    with pytest.raises(ValueError, match=message):
        spec.unflatten(vector)


def test_flatten_rejects_name_and_shape_mismatches() -> None:
    spec = ParameterSpec.from_named_tensors([("weight", torch.empty(2, 2))])

    with pytest.raises(ValueError, match="missing=.*weight"):
        spec.flatten({"bias": torch.empty(2)})
    with pytest.raises(ValueError, match="Expected shape"):
        spec.flatten({"weight": torch.empty(4)})


def test_spec_rejects_duplicate_or_non_contiguous_entries() -> None:
    duplicate = [
        ParameterEntry("weight", torch.Size([1]), 1, 0, 1),
        ParameterEntry("weight", torch.Size([1]), 1, 1, 2),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        ParameterSpec(duplicate)

    with pytest.raises(ValueError, match="Non-contiguous"):
        ParameterSpec([ParameterEntry("weight", torch.Size([1]), 1, 1, 2)])


def test_empty_spec_round_trips() -> None:
    spec = ParameterSpec([])
    assert spec.total_numel == 0
    assert len(spec.unflatten(torch.empty(0))) == 0
    assert spec.flatten({}).numel() == 0
