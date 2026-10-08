# Mappers and modulation

Mappers turn a trainable latent vector into a flat generated descriptor. Modulation strategies
change fixed mapping tensors as a function of that latent vector.

## Fixed MLP mapper

`MLPMapper` uses orthogonally initialized matrices registered as buffers. They move with the module,
appear in its state dict, and are never returned by `parameters()`:

```python
import torch

from marn import AdditiveModulation, MLPMapper

mapper = MLPMapper(
    latent_dim=16,
    output_dim=128,
    hidden_dims=(32,),
    modulation=AdditiveModulation(alpha=0.01),
)
latent = torch.randn(16, requires_grad=True)
descriptor = mapper(latent)
descriptor.sum().backward()

assert descriptor.shape == (128,)
assert list(mapper.parameters()) == []
assert latent.grad is not None
```

The optional hidden layers use GELU by default. Both hidden and output activations are replaceable
with tensor callables. A mapper accepts one one-dimensional latent vector; batched latent mappings
should be expressed explicitly by a generator so parameter ownership remains unambiguous.

`ResidualMLPMapper` adds same-width residual blocks between fixed input and output projections. It
is useful when a deeper mapping is desired without discarding the fixed-buffer invariant.

## Modulation strategies

All strategies implement `BaseModulation.modulate(weights, latent)` and avoid in-place mutation.

### Additive

`AdditiveModulation(alpha)` implements the paper's mapping-weight rule across matrix rows:

```text
W' = W + alpha * z
```

The latent length must equal the matrix input width. `alpha` must be non-negative.

### Affine

`AffineModulation(scale, shift)` applies a parameter-free per-column transformation:

```text
W' = W * (1 + scale * tanh(z)) + shift * z
```

It is identity modulation at `z = 0` and introduces no optimizer-visible parameters.

### Low rank

`LowRankModulation(rank, alpha)` decodes the latent vector into factors `A` and `B`:

```text
W' = W + (alpha / rank) * A @ B
```

For a matrix with shape `(rows, columns)`, the required latent length is
`rank * (rows + columns)`, available through `latent_dim_for(weights)`. This strategy is intended
for generators that allocate a dedicated modulation latent of that size; it is not automatically
compatible with an MLP input latent.

## Extension contracts

Custom mappers subclass `BaseMapper`, validate a one-dimensional latent, and return exactly
`output_dim` values. Custom modulation subclasses `BaseModulation` and returns a new tensor with the
same shape as its input weights.

Transformer and explicitly trainable hypernetwork mappers are deferred. Their eventual
implementations will use the same `BaseMapper` contract and will be clearly identified when they
introduce trainable parameters beyond latent vectors.
