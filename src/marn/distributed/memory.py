"""Memory profiling and strategy benchmarking utilities.

These helpers measure peak memory consumption and generation time for the three
parameter-generation strategies (SLVT, layerwise, grouped) so users can make
an informed choice for their target architecture.

CPU profiling uses :mod:`tracemalloc` to track Python-level allocation.  CUDA
profiling uses :func:`torch.cuda.max_memory_allocated` which captures the true
peak across the full allocation lifetime of the measured function.

Example::

    from marn.distributed.memory import benchmark_strategies
    from marn.runtime import ParameterSpec

    spec = ParameterSpec.from_module(some_module)
    results = benchmark_strategies(spec, latent_dim=64)
    for strategy, stats in results.items():
        print(f"{strategy}: peak={stats['peak_bytes']/1e6:.1f} MB  "
              f"time={stats['generate_time_ms']:.2f} ms")
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable

import torch

from marn.runtime import ParameterSpec


def profile_peak_memory(
    fn: Callable[..., Any],
    *args: Any,
    device: str = "cpu",
    **kwargs: Any,
) -> tuple[Any, int]:
    """Measure the peak memory used by *fn* during its execution.

    On CPU the measurement uses :mod:`tracemalloc` to track Python/C-level
    allocations.  On CUDA it resets ``torch.cuda.max_memory_allocated`` before
    the call and reads it afterwards.

    Args:
        fn: Callable to profile.
        *args: Positional arguments forwarded to *fn*.
        device: ``"cpu"`` (default) or ``"cuda"``.  Only affects which counter
            is used; the function itself must have already placed tensors on the
            correct device.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        A ``(result, peak_bytes)`` tuple where *result* is the return value of
        *fn* and *peak_bytes* is the peak allocation in bytes.

    Example::

        result, peak = profile_peak_memory(model.generator.generate_parameters)
        print(f"Peak: {peak / 1e6:.2f} MB")
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        result = fn(*args, **kwargs)
        peak_bytes = torch.cuda.max_memory_allocated()
    else:
        tracemalloc.start()
        try:
            result = fn(*args, **kwargs)
            _, peak = tracemalloc.get_traced_memory()
            peak_bytes = peak
        finally:
            tracemalloc.stop()

    return result, peak_bytes


def benchmark_strategies(
    spec: ParameterSpec,
    latent_dim: int = 64,
    device: str = "cpu",
    n_warmup: int = 2,
    n_repeat: int = 5,
) -> dict[str, dict[str, Any]]:
    """Compare SLVT, layerwise, and grouped strategies on a given spec.

    Builds one generator per strategy, runs *n_warmup* warm-up calls, then
    measures peak memory and wall-clock generation time over *n_repeat* calls.
    The reported peak is the maximum across all measured calls.

    Args:
        spec: Target ``ParameterSpec`` to benchmark against.
        latent_dim: Latent dimension to use for all strategies.
        device: ``"cpu"`` or ``"cuda"``.
        n_warmup: Number of warm-up calls before measurement (default 2).
        n_repeat: Number of timed calls (default 5).

    Returns:
        ``{strategy_name: {"peak_bytes": float, "generate_time_ms": float}}``

        On error (e.g. SLVT projection too large) the dict for that strategy
        additionally contains an ``"error"`` key with the exception message.

    Example::

        results = benchmark_strategies(spec, latent_dim=64)
        # {"slvt": {"peak_bytes": ..., "generate_time_ms": ...}, ...}
    """

    from marn.generators.layerwise import LayerwiseGenerator
    from marn.generators.lazy import LazyLayerwiseGenerator
    from marn.generators.single_vector import SingleVectorGenerator

    results: dict[str, dict[str, Any]] = {}

    def _bench(name: str, gen: Any) -> None:
        gen = gen.to(device)
        # Warm-up
        for _ in range(n_warmup):
            with torch.no_grad():
                gen.generate_parameters()

        peak_bytes = 0
        total_ms = 0.0
        for _ in range(n_repeat):
            with torch.no_grad():
                t0 = time.perf_counter()
                _, peak = profile_peak_memory(gen.generate_parameters, device=device)
                t1 = time.perf_counter()
            peak_bytes = max(peak_bytes, peak)
            total_ms += (t1 - t0) * 1000.0

        results[name] = {
            "peak_bytes": float(peak_bytes),
            "generate_time_ms": total_ms / n_repeat,
        }

    # SLVT — guard against very large projections
    try:
        slvt = SingleVectorGenerator(
            spec,
            latent_dim,
            allow_large=True,
        )
        _bench("slvt", slvt)
    except Exception as exc:  # noqa: BLE001
        results["slvt"] = {
            "peak_bytes": float("inf"),
            "generate_time_ms": float("nan"),
            "error": str(exc),
        }

    _bench("layerwise", LayerwiseGenerator(spec, latent_dim))
    _bench("lazy_layerwise", LazyLayerwiseGenerator(spec, latent_dim))

    return results
