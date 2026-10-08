# Extension recipes

This guide documents how to extend the `marn` package by implementing custom components and registering them with the built-in registries. This makes it easy to integrate custom architectures, loss formulations, callback observers, and parameter allocation strategies.

---

## 1. Custom mappers

A mapper projects a low-dimensional latent vector to a flat generated parameter descriptor. To write a custom mapper:
1. Subclass `BaseMapper`.
2. Implement `__init__(self, latent_dim: int, output_dim: int)` (plus any custom hyperparameters).
3. Implement `forward(self, latent: Tensor) -> Tensor`.
4. Register the class with `MAPPER_REGISTRY`.

### Implementation example

```python
import torch
from torch import nn, Tensor
from marn import BaseMapper, MAPPER_REGISTRY, MappingModel

# 1. Define custom mapper
class CustomSinusoidalMapper(BaseMapper):
    def __init__(self, latent_dim: int, output_dim: int, scale: float = 1.0) -> None:
        super().__init__(latent_dim, output_dim)
        self.scale = scale
        # Simple projection layer
        self.proj = nn.Linear(latent_dim, output_dim)
        
    def forward(self, latent: Tensor) -> Tensor:
        # Validate input latent shape
        self._validate_latent(latent)
        
        # Apply projection and a sinusoidal activation
        flat_output = torch.sin(self.proj(latent)) * self.scale
        return flat_output

# 2. Register it with the MAPPER_REGISTRY
MAPPER_REGISTRY.register("sinusoidal", CustomSinusoidalMapper)
```

Once registered, you can build a model config referencing this mapper type:

```python
from marn import load_config

config = load_config({
    "generator": {"latent_dim": 64},
    "mapper": {
        "type": "sinusoidal",
        "scale": 0.5  # Passed directly to the mapper constructor
    }
})
```

---

## 2. Custom loss components

A loss component evaluates predictions against target labels or calculates regularization values on the parameter generation state. To create a custom loss:
1. Subclass `BaseLoss`.
2. Implement `forward(self, context: TrainingContext) -> LossOutput`.
3. Optionally register the loss class with `LOSS_REGISTRY`.

### Implementation example

Here is a custom task loss that calculates Mean Absolute Error (L1 loss):

```python
import torch
from torch import nn
from marn import BaseLoss, LossOutput, TrainingContext, LOSS_REGISTRY

class L1TaskLoss(BaseLoss):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss()
        
    def forward(self, context: TrainingContext) -> LossOutput:
        # Read predictions and targets from the context
        loss_val = self.loss_fn(context.predictions, context.targets)
        
        # Return LossOutput
        # - total: weighted scalar Tensor for backpropagation
        # - components: dict of named, autograd-live component Tensors
        # - metrics: dict of unweighted float metrics for logging
        return LossOutput(
            total=loss_val,
            components={"l1_loss": loss_val},
            metrics={"l1_loss_val": float(loss_val.item())}
        )

# Register if you want to reference this in config configurations
LOSS_REGISTRY.register("l1_task", L1TaskLoss)
```

Use the custom loss with the `MappingTrainer`:

```python
from marn import MappingTrainer, MappingLoss

# Use standard MappingLoss wrapper which supports scaling/regularization
loss_fn = MappingLoss(task_loss=L1TaskLoss())

trainer = MappingTrainer(
    model=model,
    train_loader=train_loader,
    loss_fn=loss_fn,
)
```

---

## 3. Custom callback observers

Callbacks hook into the training lifecycle to perform monitoring, logging, checkpointing, or early termination. To write a custom callback:
1. Subclass `Callback`.
2. Override any of the lifecycle methods such as `on_fit_start`, `on_epoch_end`, `on_batch_end`, etc.

### Implementation example

This callback stops training if the loss becomes NaN or exceeds a certain threshold:

```python
import logging
from marn import Callback, LossOutput

logger = logging.getLogger("marn.callbacks")

class DivergenceGuard(Callback):
    def __init__(self, max_threshold: float = 100.0) -> None:
        self.max_threshold = max_threshold
        
    def on_batch_end(self, trainer, batch_idx: int, loss_output: LossOutput) -> None:
        loss_val = float(loss_output.total.item())
        if torch.isnan(loss_output.total) or loss_val > self.max_threshold:
            logger.warning(f"Divergence detected at batch {batch_idx}: loss = {loss_val}. Stopping.")
            trainer.should_stop = True
```

Add it to the trainer's list of callbacks:

```python
trainer = MappingTrainer(
    model=model,
    train_loader=train_loader,
    callbacks=[DivergenceGuard(max_threshold=50.0)]
)
```

---

## 4. Custom generation strategies

Generation strategies govern how target model parameters are grouped and assigned to different mappers and latent vectors. To define a custom strategy:
1. Subclass `GenerationStrategy`.
2. Implement `build(self, spec: ParameterSpec, latent_dim: int, mapper_factory: MapperFactory) -> ParameterGenerator`.
3. Create a corresponding generator class by subclassing `ParameterGenerator` to implement `generate_parameters` and `named_latent_vectors`.

### Implementation example

Here is a custom strategy that groups target parameters based on whether their name contains `"weight"` vs. `"bias"`:

```python
from collections.abc import Iterator
import torch
from torch import nn, Tensor
from marn import (
    GenerationStrategy,
    ParameterGenerator,
    ParameterSpec,
    ParameterTree,
    BaseMapper,
    AdditiveModulation
)

# 1. Define the generator class
class WeightBiasGenerator(ParameterGenerator):
    def __init__(
        self,
        spec: ParameterSpec,
        latent_dim: int,
        mapper_factory
    ) -> None:
        super().__init__(spec)
        
        # Partition target parameters into weights and biases
        self.weight_names = [name for name in spec.keys() if "weight" in name]
        self.bias_names = [name for name in spec.keys() if "bias" in name]
        
        # Create one latent variable and one mapper for weights
        self.weight_latent = nn.Parameter(torch.randn(latent_dim))
        weight_out_dim = sum(spec[n].numel for n in self.weight_names)
        self.weight_mapper = mapper_factory(latent_dim, weight_out_dim)
        
        # Create one latent variable and one mapper for biases
        self.bias_latent = nn.Parameter(torch.randn(latent_dim))
        bias_out_dim = sum(spec[n].numel for n in self.bias_names)
        self.bias_mapper = mapper_factory(latent_dim, bias_out_dim)
        
    def generate_parameters(self) -> ParameterTree:
        # Generate raw weight descriptors
        flat_weights = self.weight_mapper(self.weight_latent)
        flat_biases = self.bias_mapper(self.bias_latent)
        
        # Partition and reshape flat descriptors back into target parameters
        tree = ParameterTree()
        
        # Reshape weight tensors
        w_offset = 0
        for name in self.weight_names:
            entry = self.parameter_spec[name]
            numel = entry.numel
            val = flat_weights[w_offset : w_offset + numel].view(entry.shape)
            # Modulation is optional; here we return direct mapped weights
            tree[name] = val
            w_offset += numel
            
        # Reshape bias tensors
        b_offset = 0
        for name in self.bias_names:
            entry = self.parameter_spec[name]
            numel = entry.numel
            val = flat_biases[b_offset : b_offset + numel].view(entry.shape)
            tree[name] = val
            b_offset += numel
            
        return tree

    def named_latent_vectors(self) -> Iterator[tuple[str, Tensor]]:
        yield "weights", self.weight_latent
        yield "biases", self.bias_latent

# 2. Define the strategy class
class WeightBiasStrategy(GenerationStrategy):
    def build(
        self, spec: ParameterSpec, latent_dim: int, mapper_factory
    ) -> ParameterGenerator:
        return WeightBiasGenerator(spec, latent_dim, mapper_factory)
```

Use the custom strategy with `MappingModel`:

```python
model = MappingModel(
    target_model=nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2)),
    latent_dim=32,
    strategy=WeightBiasStrategy()
)
```

---

## 5. Config-driven setup via YAML

Using the registries, you can define a completely configuration-driven training script that parses YAML and instantiates the pipeline.

### YAML Configuration (`config.yaml`)

```yaml
generator:
  strategy: "layerwise"
  latent_dim: 128
mapper:
  type: "mlp"
  hidden_dims: [64, 64]
  modulation: "additive"
  alpha: 0.01
loss:
  lambda_stability: 0.05
  enable_stability: true
trainer:
  max_epochs: 10
  optimizer: "adamw"
  learning_rate: 0.001
  device: "cpu"
```

### Setup script

```python
from torch import nn
from marn import (
    load_config,
    MappingModel,
    MappingTrainer,
    MappingLoss,
    MAPPER_REGISTRY,
    MODULATION_REGISTRY,
    LOSS_REGISTRY,
)

# Load configuration
config = load_config("config.yaml")

# Define target
target = nn.Sequential(nn.Linear(784, 100), nn.ReLU(), nn.Linear(100, 10))

# 1. Resolve mapper and modulation from registries
mapper_cls = MAPPER_REGISTRY.get(config.mapper.type)
modulation_cls = MODULATION_REGISTRY.get(config.mapper.modulation)
modulation = modulation_cls(alpha=config.mapper.alpha)

# 2. Define mapper factory
def mapper_factory(latent_dim: int, output_dim: int):
    return mapper_cls(
        latent_dim,
        output_dim,
        hidden_dims=config.mapper.hidden_dims,
        modulation=modulation,
    )

# 3. Instantiate model
model = MappingModel(
    target_model=target,
    latent_dim=config.generator.latent_dim,
    strategy=config.generator.strategy,
    mapper_factory=mapper_factory,
)

# 4. Resolve loss from registries
task_loss_cls = LOSS_REGISTRY.get("classification")
loss_fn = MappingLoss(task_loss=task_loss_cls())

# 5. Instantiate trainer
trainer = MappingTrainer(
    model=model,
    train_loader=train_loader,
    loss_fn=loss_fn,
    config=config.trainer,
)

trainer.fit()
```
