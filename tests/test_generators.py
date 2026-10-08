"""Tests for single-vector, layer-wise, and grouped generators."""

import pytest
from torch import nn

from marn import (
    GroupedGenerator,
    GroupedStrategy,
    LayerGenerator,
    LayerwiseGenerator,
    LayerwiseStrategy,
    MLPMapper,
    ParameterSpec,
    SLVTStrategy,
    SingleVectorGenerator,
)
from marn.generators.layerwise import layerwise_groups


def linear_stack_spec() -> ParameterSpec:
    return ParameterSpec.from_module(nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1)))


def test_single_vector_generates_complete_tree_and_only_latent_is_parameter() -> None:
    spec = linear_stack_spec()
    generator = SingleVectorGenerator(spec, latent_dim=4)

    tree = generator.generate_parameters()
    tree["0.weight"].sum().backward()  # type: ignore[no-untyped-call]

    assert tuple(tree) == spec.names
    assert [name for name, _ in generator.named_latent_vectors()] == ["all"]
    assert [name for name, _ in generator.named_parameters()] == ["layer.latent"]
    assert generator.layer.latent.grad is not None


def test_single_vector_memory_guard_runs_before_projection_allocation() -> None:
    spec = ParameterSpec.from_module(nn.Linear(10, 10))
    with pytest.raises(MemoryError, match="layerwise/grouped"):
        SingleVectorGenerator(spec, latent_dim=10, max_projection_elements=100)


def test_layerwise_groups_parameters_by_owner_in_spec_order() -> None:
    spec = linear_stack_spec()
    assert layerwise_groups(spec) == {
        "0": ("0.weight", "0.bias"),
        "1": ("1.weight", "1.bias"),
    }


def test_layerwise_generator_owns_one_latent_per_parameterized_module() -> None:
    spec = linear_stack_spec()
    generator = LayerwiseGenerator(spec, latent_dim=3)
    tree = generator()
    sum(tensor.sum() for tensor in tree.values()).backward()

    assert tuple(tree) == spec.names
    assert [name for name, _ in generator.named_latent_vectors()] == ["0", "1"]
    assert len(list(generator.parameters())) == 2
    assert all(latent.grad is not None for _, latent in generator.named_latent_vectors())


def test_grouped_generator_supports_custom_groups_and_dimensions() -> None:
    spec = linear_stack_spec()
    groups = {
        "weights": ("0.weight", "1.weight"),
        "biases": ("0.bias", "1.bias"),
    }
    generator = GroupedGenerator(spec, groups, {"weights": 5, "biases": 2})

    tree = generator()

    assert tuple(tree) == ("0.weight", "1.weight", "0.bias", "1.bias")
    assert dict(generator.named_latent_vectors())["weights"].shape == (5,)
    spec.validate_tree(tree)


@pytest.mark.parametrize(
    "groups, message",
    [
        ({"a": ("0.weight",), "b": ("0.weight",)}, "duplicate"),
        ({"a": ("0.weight",)}, "cover the spec"),
        ({"a": ("unknown",)}, "cover the spec"),
    ],
)
def test_grouped_generator_rejects_invalid_coverage(
    groups: dict[str, tuple[str, ...]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GroupedGenerator(linear_stack_spec(), groups, latent_dim=2)


def test_layer_generator_validates_mapper_contract() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        LayerGenerator(3, 4, MLPMapper(3, 5))


def test_declarative_strategies_build_expected_generators() -> None:
    spec = linear_stack_spec()

    def factory(latent: int, output: int) -> MLPMapper:
        return MLPMapper(latent, output)

    assert isinstance(LayerwiseStrategy().build(spec, 2, factory), LayerwiseGenerator)
    assert isinstance(SLVTStrategy().build(spec, 2, factory), SingleVectorGenerator)
    grouped = GroupedStrategy({"all": spec.names}).build(spec, 2, factory)
    assert isinstance(grouped, GroupedGenerator)
