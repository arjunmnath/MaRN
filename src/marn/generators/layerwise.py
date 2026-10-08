"""Default layer-wise parameter generation."""

from collections import OrderedDict

from marn.generators.base import MapperFactory, default_mapper_factory
from marn.generators.grouped import GroupedGenerator
from marn.runtime import ParameterSpec


def layerwise_groups(parameter_spec: ParameterSpec) -> dict[str, tuple[str, ...]]:
    """Group weight and bias parameters by their owning module path."""

    groups: OrderedDict[str, list[str]] = OrderedDict()
    for name in parameter_spec.names:
        owner, separator, _ = name.rpartition(".")
        group_name = owner if separator else "<root>"
        groups.setdefault(group_name, []).append(name)
    return {name: tuple(parameter_names) for name, parameter_names in groups.items()}


class LayerwiseGenerator(GroupedGenerator):
    """Assign one latent mapper to each target module that owns parameters."""

    def __init__(
        self,
        parameter_spec: ParameterSpec,
        latent_dim: int,
        *,
        mapper_factory: MapperFactory = default_mapper_factory,
    ) -> None:
        super().__init__(
            parameter_spec,
            layerwise_groups(parameter_spec),
            latent_dim,
            mapper_factory=mapper_factory,
        )
