## Product objective

Expose an API with the ergonomics of Lightning or Hugging Face Trainer while training an arbitrary
PyTorch target model through a low-dimensional parameter manifold:

```python
model = MappingModel(
    target_model=ResNet18(),
    latent_dim=2048,
    strategy="layerwise",
)
trainer = MappingTrainer(model, train_loader, val_loader)
trainer.fit()
```

The target model is executed but never mutated or directly optimized. Trainable latent vectors are
mapped to named target parameters and passed through `torch.func.functional_call`. Layer-wise
generation is the default; the paper's single-latent-vector training (SLVT) remains available as a
baseline.

### Public behavior

- Target-model agnostic for ordinary `torch.nn.Module` models with positional and keyword inputs.
- Memory-efficient layer-wise generation by default, plus SLVT and grouped strategies.
- Distributed training support with only trainable latent variables synchronized by default.
- `torch.compile` compatible hot paths: tensor-only forward logic, stable containers, no mutation.
- Pluggable losses, mappers, generators, modulation strategies, callbacks, and registries.
- Clear separation of target execution, parameter generation, loss computation, optimization,
  configuration, checkpointing, and serialization.
- Helpful validation errors for unsupported shared/tied parameters, parametrizations, shape
  mismatches, missing outputs, and incompatible loss contexts.

### Runtime and memory rules

- Use `torch.func.functional_call` and never assign into `target_model` parameters.
- Preserve target buffers (for example BatchNorm running statistics) explicitly and define whether
  training updates are allowed. The safe default is frozen/stateless buffers.
- Detect tied/shared target parameters and either preserve aliasing or fail clearly; silently
  duplicating them is forbidden.
- Reconstructed tensors should be views where possible; avoid vector split/copy/reshape churn.
- A later `LazyParameterTree` may generate per-layer tensors on demand for very large targets, but
  must document which model graphs can consume parameters lazily.
- Generated full target weights are ephemeral and excluded from checkpoints.


### Training, distribution, and callbacks

- Reasonable default optimizer over trainable mapping parameters only, with user override.
- CPU, CUDA, and mixed-precision device movement; gradient accumulation and clipping.
- DDP-compatible modules and deterministic sampler epoch handling. Frozen mapper tensors are
  buffers; only actual trainable tensors participate in gradient synchronization.
- Callback observer hooks for fit/epoch/batch/validation/checkpoint lifecycle.
- Built-in early stopping and structured logger; callback failures include hook and callback names.
- Trainer accepts train and optional validation loaders and supports classification/regression
  batches through a configurable batch adapter.

### Test and quality policy

- Unit tests for every public contract, validation path, gradient route, dtype/device behavior, and
  checkpoint round trip.
- Integration tests train tiny Linear, CNN, residual, and recurrent targets without modifying
  their original parameters.
- Compare functional output with direct-module output for identical parameter trees.
- Assert mapper buffers stay frozen, latent vectors receive gradients, and optimizer parameter
  groups contain only intended tensors.
- Smoke-test `torch.compile` and single-process DDP where supported; skip with an explicit reason
  when the runtime lacks a feature.
- Ruff, strict mypy, pytest, and coverage run cleanly. Public APIs have docstrings and examples.
- Each milestone adds or updates Markdown documentation under `docs/`; planned APIs are clearly
  distinguished from implemented behavior and examples must match the current public surface.

### Non Agentic MileStone 1 - Benchmarking
- [x] Create a dedicated `benchmarks/` directory in the repository
- [x] Implement driver script `run_benchmarks.py` to automate benchmarks and output a summary report
- [x] Benchmark 1: Image Classification (CNN1/CNN2) on MNIST/FashionMNIST
- [x] Benchmark 2: Deepfake Detection (CNN1/CNN2) on Celeb-DF/FF++
- [x] Benchmark 3: Image Segmentation (U-Net/CNN3) on Cityscapes-like data
- [x] Benchmark 4: Mapped LSTM on Air Pollution time series
- [x] Benchmark 5: ResNet50 Fine-Tuning on Celeb-DF/FF++
- [x] Benchmark 6: Loss Ablation (Task Loss vs stability, smoothness, alignment)
- [x] Benchmark 7: Impact of Add-Ons (LRD, Pruning)

### Non Agentic MileStone 2 - Cookbook
- [x] Create a dedicated `cookbook/` directory in the repository
- [x] Recipe 1: SLVT Image Classification on simple CNN (Standard entry-level flow)
- [x] Recipe 2: Layer-wise Training (LWT) for Deeper Networks (Transitioning strategy to save memory)
- [x] Recipe 3: Mapping Loss and Regularization Tuning (Stability, Smoothness, and Alignment loss tuning)
- [x] Recipe 4: Low Rank Decomposition (LRD) on Linear Layers (Optimizing with U V^T constraints)
- [x] Recipe 5: Parameter Modulation Fine-Tuning (Fine-tuning pre-trained networks with block-wise scale)
- [x] Recipe 6: Integration with Network Pruning (Combining structural compression with parameter manifolds)
- [x] Recipe 7: Custom Mappers, Modulations, and Callback observers (Extending the package registry)
