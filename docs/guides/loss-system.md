# Loss system

> **Status**: Implemented in Milestone 6. Requires MappingModel (Milestone 5)
> and the parameter runtime (Milestone 1).

The loss system implements the paper's composite mapping loss:

```
L = L_task + λ_st·L_stability + λ_sm·L_smoothness + λ_al·L_alignment
```

All components receive a `TrainingContext` and return a `LossOutput`, keeping
the interface uniform and the trainer agnostic to which losses are active.

## Quick start

```python
from marn import (
    MappingModel, MappingLoss, ClassificationLoss,
    StabilityLoss, SmoothnessLoss, TrainingContext,
)

model = MappingModel(target, latent_dim=64)
loss_fn = MappingLoss(
    task_loss=ClassificationLoss(),
    stability_loss=StabilityLoss(epsilon=0.01),
    smoothness_loss=SmoothnessLoss(method="stochastic"),
    lambda_stability=0.1,
    lambda_smoothness=0.01,
)

result = model(inputs)
context = TrainingContext(
    predictions=result.predictions,
    targets=labels,
    latent_vectors=result.latent_vectors,
    generated_parameters=result.generated_parameters,
    perturbed_predictions=perturbed_output,  # if stability active
)
output = loss_fn(context)
output.total.backward()
```

## TrainingContext

Built by the trainer from `ForwardResult` plus batch targets:

```python
@dataclass
class TrainingContext:
    predictions: Any                          # target output
    targets: Any                              # ground truth
    latent_vectors: dict[str, Tensor]         # from ForwardResult
    generated_parameters: ParameterTree       # from ForwardResult
    perturbed_predictions: Any | None = None  # for stability loss
    mapper_weights: dict[str, Tensor] = {}    # for alignment loss
```

Each loss component reads only the fields it needs.

## LossOutput

```python
@dataclass
class LossOutput:
    total: Tensor                    # weighted scalar for backward()
    components: dict[str, Tensor]    # named component scalars
    metrics: dict[str, float]        # detached floats for logging
```

## Loss components

### TaskLoss

Wraps any `(predictions, targets) -> scalar` callable:

```python
TaskLoss(F.l1_loss)
ClassificationLoss(label_smoothing=0.1)  # cross-entropy + accuracy metric
RegressionLoss()                          # MSE
```

### StabilityLoss

Penalizes prediction sensitivity to latent perturbations:

```python
StabilityLoss(epsilon=0.01, num_samples=1)
```

Reads `context.perturbed_predictions` — this must be populated externally by
running the model with `z + ε·N(0,1)` perturbed latents.

### SmoothnessLoss

Penalizes the Frobenius norm of the mapper Jacobian `||J||²_F`:

```python
SmoothnessLoss(method="exact")        # O(output_dim) backward passes
SmoothnessLoss(method="stochastic", num_projections=4)  # Hutchinson estimator
```

| Method | Cost | Recommended when |
|--------|------|-----------------|
| `"exact"` | O(output_dim) backward passes | Generated params < 1000 |
| `"stochastic"` | O(num_projections) backward passes | `latent_dim > 128` |

> [!WARNING]
> The exact method computes one backward pass per output element. For a
> typical model this is prohibitively expensive. Use stochastic.

### AlignmentLoss

Cosine distance between latent vectors and column-mean summaries of mapper
weight matrices:

```python
AlignmentLoss()
```

Reads `context.mapper_weights` — a dict mapping latent names to their mapper's
weight matrices. For layerwise/grouped strategies, distances are averaged
across all latent–weight pairs.

## Composite MappingLoss

By default, all three regularization components (`stability_loss`, `smoothness_loss`, `alignment_loss`) are enabled with sensible default implementations (`"auto"` sentinel). Pass `None` explicitly to disable a component.

```python
MappingLoss(
    task_loss=ClassificationLoss(),
    stability_loss="auto",                # StabilityLoss(epsilon=0.01), or None to disable
    smoothness_loss="auto",               # SmoothnessLoss(method="stochastic", num_projections=1), or None to disable
    alignment_loss="auto",                # AlignmentLoss(), or None to disable
    lambda_stability=0.1,
    lambda_smoothness=0.01,
    lambda_alignment=0.01,
    trainable_coefficients=False,
)
```

### Trainable coefficients

When `trainable_coefficients=True`, lambda values become `nn.Parameter`s
passed through `softplus` at compute time:

```python
loss_fn = MappingLoss(
    task_loss=ClassificationLoss(),
    stability_loss=StabilityLoss(),
    trainable_coefficients=True,
)
# loss_fn.lambda_stability is now a Tensor (softplus of raw parameter)
# Guaranteed non-negative at all times
```

The raw parameters are initialized via inverse softplus so that
`softplus(raw) ≈ lambda` at the start of training.

### Static coefficients

When `trainable_coefficients=False` (default), lambdas are plain floats
and do not appear in `model.parameters()`.

## Memory implications

| Component | Memory | Notes |
|-----------|--------|-------|
| TaskLoss | O(batch) | Standard loss computation |
| StabilityLoss | O(batch) | Requires re-running target forward |
| SmoothnessLoss (exact) | O(output × latent) | Very expensive for large models |
| SmoothnessLoss (stochastic) | O(latent × projections) | Scalable alternative |
| AlignmentLoss | O(latent) | Column-mean is cheap |

## Extension patterns

Implement `BaseLoss` to create custom loss components:

```python
from marn import BaseLoss, TrainingContext, LossOutput

class MyCustomLoss(BaseLoss):
    def forward(self, context: TrainingContext) -> LossOutput:
        value = my_computation(context.latent_vectors)
        return LossOutput(
            total=value,
            components={"custom": value},
            metrics={"custom": value.detach().item()},
        )
```

## API reference

| Class | Constructor | Description |
|-------|------------|-------------|
| `TaskLoss` | `(loss_fn)` | Wrap any callable |
| `ClassificationLoss` | `(label_smoothing=0.0)` | Cross-entropy |
| `RegressionLoss` | `()` | MSE |
| `StabilityLoss` | `(epsilon=0.01, num_samples=1)` | Perturbation sensitivity |
| `SmoothnessLoss` | `(method, num_projections=4)` | Jacobian penalty |
| `AlignmentLoss` | `()` | Cosine distance |
| `MappingLoss` | `(task_loss, ...)` | Composite with coefficients |
| `TrainingContext` | `(...)` | Context dataclass |
| `LossOutput` | `(total, components, metrics)` | Result dataclass |
