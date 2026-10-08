# Stateless target execution

`TargetModel` wraps an arbitrary supported `torch.nn.Module`, freezes its parameters, and executes
it with a complete externally generated parameter tree.

## Basic use

```python
import torch
from torch import nn

from marn import ParameterTree, TargetModel

network = nn.Linear(3, 2)
target = TargetModel(network)
generated = ParameterTree({
    name: parameter.detach().clone().requires_grad_()
    for name, parameter in network.named_parameters()
})

inputs = torch.randn(5, 3)
outputs = target(generated, inputs)
outputs.sum().backward()
```

The generated tensors receive gradients. The wrapped target parameters have `requires_grad=False`,
receive no gradients, and are never assigned generated values.

## Arguments and return values

Arguments after the parameter tree are forwarded to the wrapped module. Positional, keyword, and
keyword-only arguments are supported:

```python
outputs = target(generated, inputs, attention_mask=mask, scale=0.5)
```

The wrapper returns the target's result unchanged, so tensor, tuple, mapping, and custom structured
outputs remain available to later model and loss layers.

## Buffer policy

Every forward clones all target buffers and passes those clones to `torch.func.functional_call`.
This matters in training mode: BatchNorm normally updates `running_mean`, `running_var`, and
`num_batches_tracked` in-place. The updates apply to per-call clones and are discarded afterward.

In evaluation mode, cloned buffers begin with the target's stored values, so inference uses the
expected frozen running statistics. This policy prioritizes target immutability and deterministic
stateless execution. A future explicit adaptive-buffer policy may retain generated or updated
buffers when a use case requires it.

## Validation and unsupported models

The supplied parameter mapping must contain exactly the target's parameter names and shapes.
`TargetModel` rejects:

- Tied or shared parameter objects, reporting their alias names.
- Modules registered through `torch.nn.utils.parametrize`, reporting affected module names.

These are deliberate semantic guards. A tied parameter must remain one tensor at every alias, and a
parametrized value must pass through its registered transformation. Generator-aware support will
require explicit metadata for those contracts.

## `torch.compile`

The implemented path supports `torch.compile` full-graph capture with ordinary tensor inputs and a
stable parameter tree:

```python
compiled_target = torch.compile(target, fullgraph=True)
outputs = compiled_target(generated, inputs)
```

As with other compiled PyTorch code, changing the target structure, parameter names, shapes, input
dtypes, or devices can cause recompilation or an error.
