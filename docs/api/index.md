# API reference

Reference pages are generated from docstrings via `autosummary`. Each symbol includes a
[view source](https://github.com/arjunmnath/MaRN) link when built on Read the Docs.

Import the stable surface from the package root when possible:

```python
from marn import MappingModel, MappingLoss, ClassificationLoss, MappingTrainer
```

## Models and execution

Core wrappers that connect latents, generated weights, and target forwards.

```{toctree}
:maxdepth: 1

models
runtime
```

## Mapping pipeline

Mappers, modulation, generators, and strategies that define *how* parameters are produced.

```{toctree}
:maxdepth: 1

mappers
modulation
generators
strategies
```

## Training

Losses, trainers, and training callbacks.

```{toctree}
:maxdepth: 1

losses
trainers
callbacks
```

## Configuration and persistence

Declarative configs, registries, checkpoints, and distributed helpers.

```{toctree}
:maxdepth: 1

config
registry
checkpoint
distributed
```
