# MappingModel guide

> **Status**: Implemented in Milestone 5. Requires generators (Milestone 4) and
> the parameter runtime (Milestone 1).

`MappingModel` is the user-facing entry point for training a PyTorch model
through a low-dimensional parameter manifold. It composes a frozen target, a
parameter generation strategy, and exposes a clean forward interface whose
output carries everything downstream losses need.

## Quick start

```python
import torch
from torch import nn
from marn import MappingModel

target = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
model = MappingModel(target, latent_dim=64, strategy="layerwise")

inputs = torch.randn(32, 784)
result = model(inputs)

# result.predictions        — target output, shape (32, 10)
# result.generated_parameters — ParameterTree used for this forward
# result.latent_vectors     — dict[str, Tensor] of named latent vectors
```

## Constructor

```python
MappingModel(
    target_model: nn.Module,
    latent_dim: int = 256,
    strategy: str | GenerationStrategy = "layerwise",
    *,
    mapper_factory: MapperFactory | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `target_model` | Any `nn.Module`. Tied parameters and parametrized modules are rejected. |
| `latent_dim` | Dimensionality of each trainable latent vector. Must be positive. |
| `strategy` | `"layerwise"` (default), `"slvt"`, or a `GenerationStrategy` instance. |
| `mapper_factory` | Optional `(latent_dim, output_dim) -> BaseMapper` callable. Defaults to orthogonal MLP with additive modulation. |

## Strategies

| String | Strategy | Description |
|--------|----------|-------------|
| `"layerwise"` | `LayerwiseStrategy` | One latent per target module that owns parameters. Memory-efficient default. |
| `"slvt"` | `SLVTStrategy` | Single latent vector for the entire target. Paper baseline, memory-heavy. |

Pass a `GenerationStrategy` instance for custom grouping:

```python
from marn import GroupedStrategy

groups = {"encoder": ("enc.weight", "enc.bias"), "decoder": ("dec.weight", "dec.bias")}
model = MappingModel(target, latent_dim=32, strategy=GroupedStrategy(groups))
```

## ForwardResult

`model(inputs)` returns a `ForwardResult` dataclass:

```python
@dataclass
class ForwardResult:
    predictions: Any                     # target model output
    generated_parameters: ParameterTree  # for loss computation
    latent_vectors: dict[str, Tensor]    # for stability/alignment losses
```

The result carries autograd-live tensors. Call `.predictions.sum().backward()`
and gradients flow to the latent vectors only — the target is never modified.

## Diagnostics

```python
model.trainable_parameter_count   # number of trainable latent parameters
model.target_parameter_count      # total target parameters
model.compression_ratio           # target / trainable ratio
```

## Keyword arguments

All positional and keyword arguments to `model(...)` are forwarded to the
target model:

```python
class CustomTarget(nn.Module):
    def forward(self, x, *, temperature=1.0):
        ...

model = MappingModel(CustomTarget(), latent_dim=32)
result = model(inputs, temperature=0.5)
```

## Supported targets

Any `nn.Module` with positional and keyword inputs is supported, including:

- `nn.Linear`, `nn.Conv2d`, `nn.LSTM`
- `nn.Sequential`, nested modules
- Residual architectures with skip connections

**Not supported** (will raise `UnsupportedTargetModelError`):

- Tied/shared parameters (e.g., weight tying between encoder/decoder)
- Parametrized modules (`torch.nn.utils.parametrize`)

## API reference

| Method / Property | Returns | Description |
|---|---|---|
| `forward(*args, **kwargs)` | `ForwardResult` | Generate parameters and run target |
| `trainable_parameter_count` | `int` | Count of requires_grad parameters |
| `target_parameter_count` | `int` | Total target parameter elements |
| `compression_ratio` | `float` | Target / trainable ratio |
| `generator` | `ParameterGenerator` | The underlying generator module |
| `target` | `TargetModel` | The frozen target wrapper |
