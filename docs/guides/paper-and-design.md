# Paper and design correspondence

The package is based on *Mapping Networks* by Lord Sen and Shyamapada Mukherjee. The paper proposes
replacing direct optimization in a large target parameter space with optimization of a compact
latent vector.

## Paper mechanism

The paper's basic path is:

```text
trainable latent vector
        -> fixed, modulated mapping network
        -> flattened target parameter descriptor
        -> reshaped target weights
        -> target network prediction
```

The mapping weights are fixed after orthogonal initialization. For the paper's additive
modulation, a mapping weight associated with latent coordinate `z_i` is adjusted by
`w_ij <- w_ij + alpha * z_i`. Gradients optimize the latent vector rather than the target model.

The paper presents two training arrangements:

- Single Latent Vector Training (SLVT) maps one latent vector to all target parameters. It is simple
  but its fixed projection can become prohibitively large.
- Layer-wise Training (LWT) assigns smaller latent vectors and mappings to separate layers. This is
  the package default (`strategy="layerwise"`) because it has a more practical memory profile.

The Mapping Loss combines task, stability, smoothness, and alignment terms. All four are implemented
in `MappingLoss` and configured through [Loss system guide](loss-system.md) and `LossConfig`.

## Package interpretation

The paper describes flattening, partitioning, and reshaping a generated vector each iteration. The
package compiles that positional contract once as a `ParameterSpec`, then exposes named tensors as a
`ParameterTree`. Names such as `features.0.weight` remain attached to tensors across subsystem
boundaries.

Target execution is stateless. Generated parameters are supplied to `torch.func.functional_call`;
the target module's parameter objects are never replaced or assigned. Persistent buffers are cloned
for each call so training-mode BatchNorm cannot mutate the wrapped target.

The complete design adds several engineering extensions around the paper:

- Layer-wise generation is the default, with SLVT retained as a baseline.
- Grouped generation covers custom parameter partitions; fine-tuning and LRD strategies extend the
  same generator/strategy pattern.
- Mapper, modulation, generator, and loss implementations use explicit interfaces and registries.
- Checkpoints store latent and reconstruction state, not ephemeral generated target weights.
- Scientific claims are not treated as software guarantees. Tests establish tensor semantics,
  gradient flow, immutability, and reproducibility rather than universal model quality.

## Current limitations

The implemented runtime rejects tied/shared target parameters and PyTorch parametrizations. Both
features carry semantic constraints that a generator must preserve; silently expanding aliases or
bypassing a parametrization would produce a different model. Support can be added later through an
explicit alias/transformation-aware parameter specification.

The full training stack—`MappingModel`, `MappingLoss`, `MappingTrainer`, configs, registries, and
checkpointing—is available. See the [Cookbook](../cookbook/index.md) for end-to-end scripts and
[Extension recipes](extension-recipes.md) for custom components.
