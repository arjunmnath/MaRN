# Scaling and Distributed Execution

This guide covers running `marn` at scale:

- Distributed Data Parallel (DDP) wrapping
- `torch.compile` compatibility
- Lazy per-layer parameter generation
- Memory benchmarking across strategies

---

## Distributed Data Parallel (DDP)

### Synchronization scope

A `MappingModel` holds two kinds of tensors:

| Tensor type | Storage | `requires_grad` | Synchronized by DDP? |
|---|---|---|---|
| Trainable latent vectors | `nn.Parameter` | ✅ Yes | ✅ Yes |
| Fixed mapper weights | `register_buffer` | ❌ No | ❌ No (not needed) |
| Frozen target parameters | `register_buffer` | ❌ No | ❌ No (not needed) |

**Why mapper buffers don't need synchronization**: Fixed mapper weights are
initialized identically on every rank (same architecture, same
`torch.manual_seed`) and never receive gradients. They are byte-for-byte
identical across all workers for the entire training run. Synchronizing them
would waste bandwidth without benefit.

This means DDP communication overhead scales only with the number of *trainable
latent dimensions* — typically orders of magnitude smaller than the target
model's parameter count.

### Setup and teardown

```python
import os
from marn.distributed import setup_ddp, cleanup_ddp, wrap_ddp
from marn import MappingModel
from torch import nn

def train(rank: int, world_size: int) -> None:
    # Set env vars before calling setup_ddp when using env:// init
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # Initialize the process group
    setup_ddp(rank, world_size, backend="nccl")   # use "gloo" for CPU

    # Build and wrap the model
    target = nn.Linear(512, 10)
    model = MappingModel(target, latent_dim=64, strategy="layerwise")
    model = model.cuda(rank)
    ddp_model = wrap_ddp(model, device_ids=[rank])

    # Normal training loop — gradients are synchronized automatically
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=3e-4)
    for batch in loader:
        result = ddp_model(batch)
        loss = criterion(result.predictions, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    cleanup_ddp()
```

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py
```

### Backend choice

| Backend | When to use |
|---|---|
| `nccl` | Multi-GPU training (CUDA required) |
| `gloo` | CPU training or cross-platform CI |
| `mpi` | HPC clusters with MPI installed |

### Accessing the underlying model

```python
ddp_model = wrap_ddp(model, device_ids=[rank])

# Access MappingModel methods
ddp_model.module.trainable_parameter_count
ddp_model.module.compression_ratio

# Checkpointing — save from rank 0 only
if rank == 0:
    save_checkpoint("ckpt.pt", ddp_model.module)
```

### DistributedSampler

Use `torch.utils.data.DistributedSampler` to ensure each rank sees a disjoint
data shard and set the epoch so shuffling is consistent:

```python
from torch.utils.data import DistributedSampler, DataLoader

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

for epoch in range(max_epochs):
    sampler.set_epoch(epoch)   # deterministic shuffle per epoch
    trainer.fit_epoch(loader)
```

---

## `torch.compile` Compatibility

`marn` is designed to be `torch.compile`-compatible:

- All forward logic is tensor-only (no Python-side branching on tensor values).
- `ParameterTree` uses a plain `dict`; stable container shapes allow tracing.
- `torch.func.functional_call` is traced by Dynamo.

### Basic usage

```python
import torch
from marn import MappingModel
from torch import nn

model = MappingModel(nn.Linear(512, 10), latent_dim=64)
compiled = torch.compile(model, fullgraph=False)

result = compiled(inputs)
```

### `fullgraph=True` constraints

Full-graph compilation (`fullgraph=True`) requires that the entire forward pass
is capturable as a single Dynamo graph. This may cause issues with:

- **BatchNorm in training mode** — running statistics updates cause a graph
  break. Use `model.eval()` or `torch.no_grad()` for inference compilation.
- **`LazyLayerwiseGenerator`** — the `del flat_tensor` + `empty_cache()` call
  inside the loop may cause a graph break. Use `LayerwiseGenerator` instead
  when `fullgraph=True` is required.
- **Custom mapper callbacks** — any Python-side branching on tensor content
  causes a graph break.

`fullgraph=False` (the default) is recommended for training; it silently falls
back to eager for any un-traceable subgraphs.

---

## Lazy Per-Layer Parameter Generation

### Why it matters

With the default `LayerwiseGenerator`, *all* groups' flat buffers exist
simultaneously before any of them are unflattened. For a target with many large
layers this can require more peak memory than necessary.

`LazyLayerwiseGenerator` generates each group's flat tensor, immediately
unflattens it into named tensors, then **deletes the flat intermediate** before
moving to the next group. On CUDA it also calls `torch.cuda.empty_cache()` to
return freed memory to the pool immediately.

### Usage

```python
from marn.generators import LazyLayerwiseGenerator
from marn.runtime import ParameterSpec
from torch import nn

spec = ParameterSpec.from_module(large_model)
gen = LazyLayerwiseGenerator(spec, latent_dim=128)

# Drop-in replacement for LayerwiseGenerator
params = gen.generate_parameters()   # lower peak memory
```

### Compatibility constraints

| Context | Compatible? |
|---|---|
| Eager training (default) | ✅ Yes |
| Gradient computation | ✅ Yes |
| `torch.compile(fullgraph=False)` | ✅ Usually (may see graph breaks) |
| `torch.compile(fullgraph=True)` | ⚠️ No — `empty_cache` breaks the graph |
| DDP | ✅ Yes |
| Very large targets (memory-critical) | ✅ Yes (primary use case) |

Disable the cache flush for micro-benchmarking or when you know CUDA is not
present:

```python
gen = LazyLayerwiseGenerator(spec, latent_dim=64, empty_cache_on_cuda=False)
```

### Future: `LazyParameterTree`

A true streaming implementation — generate one layer, execute target
forward for that layer, discard, repeat — is **not compatible** with
`torch.func.functional_call`, which requires the entire parameter dict to be
present at call time. See `marn.runtime.lazy` for the design
constraints that must be satisfied before this can be implemented.

---

## Memory Benchmarking

Use `benchmark_strategies` to measure peak memory and generation time for all
three strategies on a given `ParameterSpec`:

```python
from marn.distributed import benchmark_strategies
from marn.runtime import ParameterSpec
from torch import nn

spec = ParameterSpec.from_module(nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.Conv2d(64, 128, 3),
    nn.Linear(128, 10),
))

results = benchmark_strategies(spec, latent_dim=64, n_warmup=3, n_repeat=10)
for strategy, stats in results.items():
    print(
        f"{strategy:20s}  "
        f"peak={stats['peak_bytes'] / 1e6:7.2f} MB  "
        f"time={stats['generate_time_ms']:6.2f} ms"
    )
```

Expected output shape (values vary by architecture):

```
slvt                  peak=  12.34 MB  time=  0.83 ms
layerwise             peak=   1.02 MB  time=  0.21 ms
lazy_layerwise        peak=   0.61 MB  time=  0.25 ms
```

### Profiling individual operations

For finer-grained measurements, use `profile_peak_memory` directly:

```python
from marn.distributed import profile_peak_memory

gen = LayerwiseGenerator(spec, latent_dim=64)
tree, peak = profile_peak_memory(gen.generate_parameters, device="cpu")
print(f"Peak: {peak / 1e6:.2f} MB")
```

### Benchmark methodology notes

- Peak is the **maximum** across `n_repeat` measured calls (after `n_warmup`
  warm-up calls are discarded).
- On CPU, `tracemalloc` captures Python-level allocations; native allocations
  inside PyTorch's C++ backend may be under-reported. Use CUDA profiling for
  accurate GPU numbers.
- On CUDA, `torch.cuda.reset_peak_memory_stats()` is called before each
  measurement, so peaks are per-call, not cumulative.
- For accurate comparison, always benchmark on the same hardware and PyTorch
  version with the same `latent_dim`.
