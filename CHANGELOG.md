# Changelog

All notable changes to the `marn` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Added
- **Default MappingLoss Components**:
  - Automatically instantiates default `StabilityLoss`, `SmoothnessLoss`, and `AlignmentLoss` in `MappingLoss` when not explicitly supplied. Use `None` to explicitly disable components.
  - Automatically registers and feeds the required context fields (`perturbed_predictions` and `mapper_weights`) from the trainer when default/custom losses are active.
  - Graceful support for model validation under `torch.no_grad()` inside `SmoothnessLoss` by skipping autograd Jacobian estimation when gradients are disabled.

---

## [0.1.0] - 2026-06-21

Initial release of the `marn` package for training PyTorch models through low-dimensional generated parameter manifolds.

### Added
- **Compiled Parameter Runtime**:
  - Reusable `ParameterSpec` metadata builder to compile parameter offsets once.
  - Immutable `ParameterTree` for zero-copy flat-vector views.
- **Stateless Target Execution**:
  - Stateless execution wrapper using `torch.func.functional_call`.
  - BatchNorm running statistics isolation during parameter-generation runs.
- **Mappers & Modulation**:
  - `BaseMapper` interface and orthogonal fixed weight projections (`MLPMapper`, `ResidualMLPMapper`).
  - `BaseModulation` interface supporting additive, learned affine, and low-rank modulation strategies.
- **Parameter Generators & Strategies**:
  - Global projection (`SingleVectorGenerator`/SLVT), layerwise projection (`LayerwiseGenerator`), and custom `GroupedGenerator` strategies.
- **High-level User Interfaces**:
  - `MappingModel` constructor composing target, latent variables, and strategies.
- **Loss Subsystem**:
  - Composite `MappingLoss` balancing task loss with stability regularization (latent noise perturbation), smoothness regularization (exact and Hutchinson stochastic Jacobian norm), and alignment regularization (weight matrix summary cosine alignment).
- **Configuration & Registries**:
  - Pydantic v2 schemas (`MappingConfig`, `LossConfig`, `TrainerConfig`) with YAML loading.
  - Extension registries (`MAPPER_REGISTRY`, `LOSS_REGISTRY`, `MODULATION_REGISTRY`, `GENERATOR_REGISTRY`).
- **Trainer & Callbacks**:
  - Standard training loop (`MappingTrainer`) with AMP, gradient accumulation/clipping, and batch adapters.
  - Observer lifecycle hooks (`Callback`, `EarlyStopping`, `MetricLogger`).
- **Checkpointing**:
  - Compact versioned checkpointing excluding generated target parameters.
  - Pre-load target module `ParameterSpec` architecture validation.
- **Scaling Utilities**:
  - DDP wrapper (`wrap_ddp`) synchronizing only trainable latents.
  - Soft-lazy generation (`LazyLayerwiseGenerator`) freeing intermediate flat tensors early.
  - CPU/CUDA max memory profiling (`profile_peak_memory`, `benchmark_strategies`).
- **Documentation & Packaging**:
  - Official Read the Docs configuration and web linking commands.
  - Extension recipes guide and four runnable example scripts.
  - PEP 561 compliance marker (`py.typed`).
