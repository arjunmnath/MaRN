# Parameter runtime

The parameter runtime is the named boundary between future generators and stateless target
execution. It consists of `ParameterEntry`, `ParameterSpec`, and `ParameterTree`.

## `ParameterEntry`

Each immutable entry records:

- `name`: fully qualified target parameter name.
- `shape`: original `torch.Size`.
- `numel`: number of scalar elements.
- `start` and `stop`: offsets into a compiled flat descriptor.
- `slice_range`: the corresponding Python `slice`.

Applications normally obtain entries from a `ParameterSpec` rather than constructing them.

## `ParameterSpec`

Create a specification once from any supported `nn.Module`:

```python
from torch import nn
from marn import ParameterSpec

model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
spec = ParameterSpec.from_module(model)

print(spec.names)
print(spec.total_numel)
print(spec.entry("0.weight").shape)
```

The module's parameter order defines the flat layout. `unflatten()` validates the descriptor length
and returns shaped views over its storage:

```python
import torch

vector = torch.randn(spec.total_numel, requires_grad=True)
tree = spec.unflatten(vector)

assert tree["0.weight"].shape == model[0].weight.shape
tree["0.weight"].square().sum().backward()
assert vector.grad is not None
```

Because the tensors are views, reconstruction does not copy the descriptor. Consumers must not
modify those tensors in-place while autograd needs their previous values.

`flatten(tree)` validates names and shapes, then concatenates tensors in compiled order. Mapping
insertion order does not affect the result. `validate_tree(tree)` performs name and shape checks
without allocating a flattened tensor.

## `ParameterTree`

`ParameterTree` is an immutable mapping from parameter names to tensors:

```python
import torch

from marn import ParameterTree

tree = ParameterTree({
    "projection.weight": torch.randn(2, 3, requires_grad=True),
    "projection.bias": torch.zeros(2, requires_grad=True),
})
```

The mapping cannot be structurally changed after construction, but tensor values retain ordinary
PyTorch behavior and autograd history. `to_dict()` returns a shallow mutable mapping for APIs such
as `functional_call`; it does not clone tensor storage. `map(function)` produces a new named tree
after applying a tensor transformation to every value.

Invalid empty names, non-tensor values, missing parameters, unexpected parameters, incompatible
shapes, and incompatible flat vector lengths fail with descriptive exceptions.
