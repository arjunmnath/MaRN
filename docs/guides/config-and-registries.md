# Configuration and registries

> **Status**: Implemented in Milestone 7.

This guide describes how to configure mapping networks using typed Pydantic models, load configuration files, and extend the toolkit with custom component registries.

## Configuration models

The configuration system uses Pydantic (v2) to validate parameters, forbid unknown keys, and support robust serialization.

### MappingConfig

`MappingConfig` is the top-level configuration that groups generator and mapper parameters.

```python
from marn import MappingConfig, GeneratorConfig, MapperConfig

config = MappingConfig(
    generator=GeneratorConfig(
        strategy="layerwise",
        latent_dim=128,
    ),
    mapper=MapperConfig(
        type="mlp",
        hidden_dims=(64, 64),
        modulation="additive",
        alpha=0.01,
    ),
)
```

| Config Class | Attributes | Default | Description |
|---|---|---|---|
| **`MapperConfig`** | `type` | `"mlp"` | Registry key for the mapper. |
| | `hidden_dims` | `()` | Hidden layer dimensions. |
| | `modulation` | `"additive"` | Registry key for the modulation. |
| | `alpha` | `0.01` | Additive modulation strength. |
| | `hidden_dim` | `128` | Hidden dimension for residual MLP. |
| | `depth` | `2` | Number of residual blocks. |
| **`GeneratorConfig`** | `strategy` | `"layerwise"` | Strategy: `"layerwise"`, `"slvt"`, or `"grouped"`. |
| | `latent_dim` | `256` | Latent vector size. |
| | `groups` | `None` | Mapping of group name to list of parameters. |
| | `max_projection_elements` | `100_000_000` | Upper bound for SLVT projection matrices. |
| **`LossConfig`** | `task` | `TaskLossConfig()` | Task loss configuration. |
| | `lambda_stability` | `0.1` | Weight for stability loss. |
| | `lambda_smoothness` | `0.01` | Weight for smoothness loss. |
| | `lambda_alignment` | `0.01` | Weight for alignment loss. |
| | `stability_epsilon` | `0.01` | Perturbation scale. |
| | `stability_num_samples` | `1` | Number of noise samples. |
| | `smoothness_method` | `"stochastic"` | `"stochastic"` or `"exact"`. |
| | `smoothness_projections` | `4` | Number of projection vectors. |
| | `trainable_coefficients`| `False` | Whether coefficients are trainable. |
| | `enable_stability` | `False` | Enable stability loss component. |
| | `enable_smoothness` | `False` | Enable smoothness loss component. |
| | `enable_alignment` | `False` | Enable alignment loss component. |
| **`TrainerConfig`** | `max_epochs` | `100` | Maximum number of training epochs. |
| | `optimizer` | `"adam"` | Optimizer name: `"adam"`, `"adamw"`, or `"sgd"`. |
| | `learning_rate` | `1e-3` | Base learning rate. |
| | `weight_decay` | `0.0` | Weight decay factor. |
| | `scheduler` | `None` | Scheduler name: `"cosine"`, `"step"`, `"plateau"`, `"exponential"`. |
| | `scheduler_kwargs` | `{}` | Arguments passed to the scheduler. |
| | `gradient_clip_norm` | `None` | Max norm for gradient clipping. |
| | `gradient_clip_value`| `None` | Max value for gradient clipping. |
| | `accumulation_steps` | `1` | Number of batches for gradient accumulation. |
| | `amp_enabled` | `False` | Enable Automatic Mixed Precision (AMP). |
| | `device` | `"auto"` | Target device (`"cpu"`, `"cuda"`, or `"auto"`). |
| | `seed` | `None` | Random seed for reproducibility. |

## Loading YAML configuration files

Use `load_config` to load configurations from a YAML file or a Python dictionary.

```python
from marn import load_config

# Load from file
config = load_config("configs/layerwise_adam.yaml")

# Load from dictionary
config = load_config({
    "generator": {"latent_dim": 64},
    "mapper": {"type": "residual_mlp", "depth": 3}
})
```

### Validation errors

When Pydantic encounters type mismatches or unknown keys (due to `extra="forbid"`), it raises a `ValidationError`:

```python
from pydantic import ValidationError
from marn import load_config

try:
    load_config({"generator": {"latent_dim": "invalid"}})
except ValidationError as e:
    print(e)  # Clear explanation of path and cause
```

## Component registries

The package uses a generic `Registry` class to manage pluggable components. Registries map string keys to component classes.

```python
from marn import (
    MAPPER_REGISTRY,
    MODULATION_REGISTRY,
    LOSS_REGISTRY,
    GENERATOR_REGISTRY,
)

print(MAPPER_REGISTRY.available())
# Output: ('mlp', 'residual_mlp')
```

### Registries API

- `register(key, cls)`: Register a custom component class. Raises `ValueError` if the key is already taken.
- `get(key)`: Retrieve the registered class. Raises `KeyError` if the key does not exist, listing available keys.
- `available()`: Returns a sorted tuple of all registered keys.

### Adding custom plugins

You can register custom implementations to extend the toolkit. Once registered, they can be referenced by string names in configurations.

```python
from torch import nn
from marn import BaseMapper, MAPPER_REGISTRY, MappingModel

class MyCustomMapper(BaseMapper):
    def __init__(self, latent_dim, output_dim, modulation):
        super().__init__(latent_dim, output_dim, modulation)
        # Initialization logic
        
    def forward(self, latent):
        # Mapping logic
        ...

# Register custom component
MAPPER_REGISTRY.register("custom_mapper", MyCustomMapper)

# Instantiate config using custom component
config = load_config({
    "mapper": {"type": "custom_mapper"}
})
```
