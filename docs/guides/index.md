# User guide

These pages explain how `marn` is designed and how to use it day to day. They complement the
[API reference](../api/index.md) (signatures and types) and the [Cookbook](../cookbook/index.md)
(runnable scripts).

## Recommended reading order

If you are new to mapping-based training, follow this path:

1. [Paper and design correspondence](paper-and-design.md) — paper mechanism vs. package layout.
2. [Parameter runtime](parameter-runtime.md) — named tensors and the parameter tree.
3. [Stateless target execution](stateless-target.md) — `functional_call` and immutability.
4. [Mappers and modulation](mappers-and-modulation.md) — latent projections and weight updates.
5. [Generators and strategies](generators-and-strategies.md) — SLVT, layerwise, grouped, LRD, PEFT.
6. [MappingModel guide](mapping-model.md) — high-level orchestration API.
7. [Loss system guide](loss-system.md) — composite `MappingLoss` and regularizers.
8. [Configuration and registries](config-and-registries.md) — Pydantic configs and plugins.
9. [Trainer and callbacks guide](trainer-guide.md) — loops, adapters, and observers.
10. [Checkpointing and resumption](checkpointing.md) — save/load and compatibility checks.
11. [Scaling and distributed execution](scaling.md) — DDP, memory, and compilation.
12. [Development and compatibility](development.md) — supported stack and invariants.
13. [Extension recipes](extension-recipes.md) — custom mappers, modulation, and registration.

```{toctree}
:maxdepth: 1
:hidden:

paper-and-design
parameter-runtime
stateless-target
mappers-and-modulation
generators-and-strategies
mapping-model
loss-system
config-and-registries
trainer-guide
checkpointing
scaling
development
extension-recipes
```

## Guide index

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Guide
     - Topics
   * - :doc:`paper-and-design`
     - SLVT vs. layerwise training, mapping loss, engineering extensions
   * - :doc:`parameter-runtime`
     - ``ParameterSpec``, ``ParameterTree``, naming invariants
   * - :doc:`stateless-target`
     - Frozen targets, buffers, unsupported tied/parametrized weights
   * - :doc:`mappers-and-modulation`
     - ``MLPMapper``, additive/affine/low-rank modulation
   * - :doc:`generators-and-strategies`
     - Strategy/generator pairing and memory trade-offs
   * - :doc:`mapping-model`
     - End-user ``MappingModel`` workflow
   * - :doc:`loss-system`
     - Task, stability, smoothness, alignment terms
   * - :doc:`config-and-registries`
     - YAML configs and component registries
   * - :doc:`trainer-guide`
     - ``MappingTrainer``, batch adapters, callbacks
   * - :doc:`checkpointing`
     - Schema, validation, resuming training
   * - :doc:`scaling`
     - Distributed training and performance tooling
   * - :doc:`development`
     - Tests, typing, documentation expectations
   * - :doc:`extension-recipes`
     - Custom components without forking the library
```
