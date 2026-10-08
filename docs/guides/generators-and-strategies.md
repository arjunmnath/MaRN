# Generators and strategies

Generators own trainable latent vectors and use mappers to produce complete `ParameterTree`
instances. Strategies are small declarative builders that select a generator arrangement.

## Layer-wise generation

`LayerwiseGenerator` is the default architecture. Parameters are grouped by the module that owns
them, so a layer's weight and bias share one latent mapper:

```python
from torch import nn

from marn import LayerwiseGenerator, ParameterSpec

target = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
spec = ParameterSpec.from_module(target)
generator = LayerwiseGenerator(spec, latent_dim=16)
parameters = generator.generate_parameters()

assert tuple(name for name, _ in generator.named_latent_vectors()) == ("0", "2")
spec.validate_tree(parameters)
```

Only `LayerGenerator.latent` tensors are optimizer-visible parameters. Default MLP mapper weights
remain buffers. Each group is reconstructed independently, reducing peak fixed-projection size
relative to one giant mapper.

## Single-vector generation

`SingleVectorGenerator` implements the paper's SLVT baseline. One latent and one projection generate
all target parameters:

```python
from marn import SingleVectorGenerator

generator = SingleVectorGenerator(spec, latent_dim=16)
parameters = generator()
```

The projection contains approximately `latent_dim * target_parameter_count` elements. Construction
raises `MemoryError` when that exceeds `max_projection_elements` (100 million by default). Passing
`allow_large=True` is an explicit acknowledgement, not a memory optimization.

## Explicit groups

`GroupedGenerator` accepts a complete, non-overlapping mapping of group names to parameter names:

```python
from marn import GroupedGenerator

groups = {
    "input": ("0.weight", "0.bias"),
    "output": ("2.weight", "2.bias"),
}
generator = GroupedGenerator(
    spec,
    groups,
    latent_dim={"input": 12, "output": 8},
)
```

Every spec name must occur exactly once. A single integer applies one latent dimension to all
groups; a mapping assigns dimensions individually.

## Fine-tuning extension

`FineTuningGenerator` implements the paper's pretrained parameter-efficient fine-tuning strategy (Section 2.2.6). Instead of generating target weights from scratch, it projects trainable latent variables into a low-dimensional modulation vector $o$ of size $\lceil |W_{pretrained}| / L \rceil$. This vector is tiled (block-wise repeated) $L$ times and added to the original pre-trained frozen weight tensor:

$$W_{FT} = W_{pretrained} + \alpha \cdot o_{expanded}$$

This is constructed declaratively using `FineTuningStrategy`:

```python
from marn import FineTuningStrategy, MappingModel

# Save original model weights
pretrained = dict(target.named_parameters())

strategy = FineTuningStrategy(pretrained_parameters=pretrained, L=250, alpha=0.01)
model = MappingModel(target, latent_dim=64, strategy=strategy)
```

## Low-Rank Decomposition (LRD)

`LRDGenerator` implements target-side Low-Rank Decomposition (Section 2.2.5). For large 2D weight matrices (e.g. Linear layers) of shape $[m, n]$, LRD represents the weight as $W = U V^T$, where $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{n \times r}$ with rank $r < \min(m, n)$.

Instead of generating $m \cdot n$ elements, the generator only generates $r(m + n)$ elements, reducing projection/buffer sizes on the mappers.

This is constructed declaratively using `LRDStrategy`:

```python
from marn import LRDStrategy, MappingModel

strategy = LRDStrategy(rank=16)
model = MappingModel(target, latent_dim=128, strategy=strategy)
```

## Strategies and extension

`LayerwiseStrategy`, `SLVTStrategy`, `GroupedStrategy`, `FineTuningStrategy`, and `LRDStrategy` construct their corresponding generators through a shared `GenerationStrategy.build()` contract. `ParameterGenerator` subclasses implement `generate_parameters()` and `named_latent_vectors()`.

Custom mapper creation is supported with `mapper_factory(latent_dim, output_dim)`. The returned mapper dimensions must match the group contract. Custom generator outputs are always validated against their target `ParameterSpec` before stateless execution.

Lazy generation (via `LazyLayerwiseGenerator`) is also supported to reduce peak memory usage by materializing and unflattening parameter tensors one group at a time.
