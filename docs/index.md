# marn documentation

`marn` is a PyTorch package for training target models through low-dimensional,
generated parameter spaces. It is inspired by the paper *Mapping Networks* and is being built as a
model-agnostic, stateless toolkit rather than a reproduction tied to one experiment.

## Current Status

The package is under active development. The following layers are fully implemented:

- **Parameter runtime**: Named parameter metadata and storage (`ParameterSpec` and `ParameterTree`).
- **Stateless target execution**: Frozen target wrapper and stateless execution using `torch.func.functional_call`.
- **Mappers & modulation**: Orthogonal and residual MLP mappers, plus additive, affine, and low-rank weight modulation.
- **Generators & strategies**: Single-latent vector (SLVT), layerwise, and custom grouped strategy/generator mappings.
- **MappingModel**: High-level `MappingModel` interface that orchestrates stateless generation and execution.
- **Loss system**: Task loss plus stability, Jacobian smoothness, and cosine alignment regularizations.
- **Configuration & registries**: Pydantic configurations and registries for custom components.
- **Trainer & callbacks**: Training loops, customizable batch adapters, and lifecycle callback observers.
- **Checkpointing**: Versioned compact checkpoint save/load with architecture validation and training resumption.
- **Scaling & distributed**: DDP wrapping, lazy generation, memory benchmarking, and `torch.compile` compatibility.


See [`progress.md`](../progress.md) for the complete specification and implementation checklist.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guides

guides/paper-and-design
guides/parameter-runtime
guides/stateless-target
guides/mappers-and-modulation
guides/generators-and-strategies
guides/mapping-model
guides/loss-system
guides/config-and-registries
guides/trainer-guide
guides/checkpointing
guides/scaling
guides/development
guides/extension-recipes
```


```{toctree}
:maxdepth: 2
:hidden:
:caption: API Reference

api/models
api/losses
api/trainers
api/mappers
api/modulation
api/generators
api/strategies
api/runtime
api/config
api/registry
api/callbacks
api/checkpoint
api/distributed
```


## Getting Started

- [Installation](installation.md)
- [Quickstart](quickstart.md)

## User Guides

- [Paper and Design Correspondence](guides/paper-and-design.md)
- [Parameter Runtime](guides/parameter-runtime.md)
- [Stateless Target Execution](guides/stateless-target.md)
- [Mappers and Modulation](guides/mappers-and-modulation.md)
- [Generators and Strategies](guides/generators-and-strategies.md)
- [MappingModel Guide](guides/mapping-model.md)
- [Loss System Guide](guides/loss-system.md)
- [Configuration and Registries](guides/config-and-registries.md)
- [Trainer and Callbacks Guide](guides/trainer-guide.md)
- [Checkpointing and Resumption](guides/checkpointing.md)
- [Scaling and Distributed Execution](guides/scaling.md)
- [Development and Compatibility](guides/development.md)
- [Extension Recipes](guides/extension-recipes.md)


## API Reference

- [Models](api/models.md)
- [Losses](api/losses.md)
- [Trainers](api/trainers.md)
- [Mappers](api/mappers.md)
- [Modulation](api/modulation.md)
- [Generators](api/generators.md)
- [Strategies](api/strategies.md)
- [Runtime](api/runtime.md)
- [Configuration](api/config.md)
- [Registries](api/registry.md)
- [Callbacks](api/callbacks.md)
- [Checkpointing](api/checkpoint.md)
- [Distributed & Memory](api/distributed.md)


