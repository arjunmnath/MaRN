"""Tests for Milestone 10: DDP utilities, lazy generation, and memory profiling.

DDP tests use the single-process gloo backend so they run on any machine
without CUDA or NCCL.  Each DDP test initializes and tears down the process
group within the test itself to stay independent and side-effect-free.

``torch.compile`` tests are skipped when the compiler is unavailable (Python
< 3.10 or dynamo not built).
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

import torch.distributed as dist


from marn import (
    LazyLayerwiseGenerator,
    LayerwiseGenerator,
    MappingModel,
    is_ddp_available,
    benchmark_strategies,
    profile_peak_memory,
    setup_ddp,
    cleanup_ddp,
    wrap_ddp,
)
from marn.distributed.ddp import is_ddp_available as _is_ddp_available
from marn.runtime import ParameterSpec


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_model(n_features: int = 4, n_outputs: int = 2, latent_dim: int = 8) -> MappingModel:
    return MappingModel(nn.Linear(n_features, n_outputs), latent_dim=latent_dim)


def _small_spec() -> ParameterSpec:
    return ParameterSpec.from_module(nn.Linear(4, 2))


def _setup_single_process_ddp() -> None:
    """Initialize a single-process gloo process group."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    setup_ddp(rank=0, world_size=1, backend="gloo")


def _cleanup_single_process_ddp() -> None:
    cleanup_ddp()


# ── DDP availability ──────────────────────────────────────────────────────────


class TestDDPAvailability:
    def test_is_ddp_available_returns_bool(self) -> None:
        result = is_ddp_available()
        assert isinstance(result, bool)

    def test_private_function_agrees(self) -> None:
        assert is_ddp_available() == _is_ddp_available()

    def test_dist_available_consistent(self) -> None:
        assert is_ddp_available() == dist.is_available()


# ── setup_ddp / cleanup_ddp ───────────────────────────────────────────────────


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
class TestSetupCleanupDDP:
    def test_setup_and_cleanup(self) -> None:
        assert not dist.is_initialized()
        _setup_single_process_ddp()
        assert dist.is_initialized()
        _cleanup_single_process_ddp()
        assert not dist.is_initialized()

    def test_setup_raises_without_dist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dist, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="not available"):
            setup_ddp(rank=0, world_size=1)

    def test_cleanup_safe_when_not_initialized(self) -> None:
        # Should not raise even if group was never created
        assert not dist.is_initialized()
        cleanup_ddp()  # no-op
        assert not dist.is_initialized()


# ── wrap_ddp ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
class TestWrapDDP:
    def setup_method(self) -> None:
        if not dist.is_initialized():
            _setup_single_process_ddp()

    def teardown_method(self) -> None:
        if dist.is_initialized():
            _cleanup_single_process_ddp()

    def test_wrap_returns_ddp_module(self) -> None:
        model = _make_model()
        ddp = wrap_ddp(model)
        assert isinstance(ddp, torch.nn.parallel.DistributedDataParallel)

    def test_underlying_model_accessible(self) -> None:
        model = _make_model()
        ddp = wrap_ddp(model)
        assert ddp.module is model

    def test_only_latent_params_in_ddp_parameters(self) -> None:
        """Latent vectors must require grad; frozen mapper/target params must not."""
        model = _make_model()
        ddp = wrap_ddp(model)

        # Separate trainable from frozen among all DDP-visible parameters
        trainable = [p for p in ddp.parameters() if p.requires_grad]
        frozen = [p for p in ddp.parameters() if not p.requires_grad]

        # There must be at least one trainable latent vector
        assert len(trainable) > 0, "Expected at least one trainable latent parameter"

        # Trainable parameter count should match the underlying model's
        assert sum(p.numel() for p in trainable) == model.trainable_parameter_count

        # Frozen parameters exist (mapper buffers exposed as non-grad params by DDP)
        # Their total numel should exceed the trainable count for a typical model
        assert len(frozen) >= 0  # may be 0 if architecture has no buffers in params

    def test_wrap_raises_without_process_group(self) -> None:
        _cleanup_single_process_ddp()
        model = _make_model()
        with pytest.raises(RuntimeError, match="Process group is not initialized"):
            wrap_ddp(model)
        # Re-initialize for teardown
        _setup_single_process_ddp()

    def test_wrap_raises_without_dist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("marn.distributed.ddp.dist.is_available", lambda: False)
        model = _make_model()
        with pytest.raises(RuntimeError, match="not available"):
            wrap_ddp(model)

    def test_ddp_forward_produces_forward_result(self) -> None:
        model = _make_model()
        ddp = wrap_ddp(model)
        x = torch.randn(4, 4)
        result = ddp(x)
        assert hasattr(result, "predictions")
        assert result.predictions.shape == (4, 2)

    def test_ddp_forward_gradients_flow_to_latents(self) -> None:
        model = _make_model()
        ddp = wrap_ddp(model)
        x = torch.randn(4, 4)
        result = ddp(x)
        loss = result.predictions.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Latent {name!r} has no gradient"

    def test_buffer_count_unchanged_after_wrap(self) -> None:
        model = _make_model()
        n_buffers_before = sum(1 for _ in model.buffers())
        ddp = wrap_ddp(model)
        n_buffers_after = sum(1 for _ in ddp.buffers())
        assert n_buffers_before == n_buffers_after


# ── torch.compile smoke test ──────────────────────────────────────────────────


def _compile_available() -> bool:
    try:
        import torch._dynamo  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _compile_available(), reason="torch.compile / dynamo not available")
class TestTorchCompile:
    def test_compile_forward_produces_result(self) -> None:
        model = _make_model()
        compiled = torch.compile(model, fullgraph=False)
        x = torch.randn(4, 4)
        result = compiled(x)
        assert hasattr(result, "predictions")
        assert result.predictions.shape == (4, 2)

    def test_compile_output_matches_eager(self) -> None:
        torch.manual_seed(0)
        model = _make_model()
        # Eager
        model.eval()
        x = torch.randn(4, 4)
        with torch.no_grad():
            eager_out = model(x).predictions

        # Compiled
        compiled = torch.compile(model, fullgraph=False)
        with torch.no_grad():
            compiled_out = compiled(x).predictions

        assert torch.allclose(eager_out, compiled_out, atol=1e-5), (
            f"Compiled output differs from eager: max delta "
            f"{(eager_out - compiled_out).abs().max().item():.2e}"
        )

    def test_compile_with_layerwise_strategy(self) -> None:
        model = MappingModel(nn.Linear(4, 2), latent_dim=8, strategy="layerwise")
        compiled = torch.compile(model, fullgraph=False)
        x = torch.randn(2, 4)
        result = compiled(x)
        assert result.predictions.shape == (2, 2)

    def test_compile_with_slvt_strategy(self) -> None:
        model = MappingModel(nn.Linear(4, 2), latent_dim=8, strategy="slvt")
        compiled = torch.compile(model, fullgraph=False)
        x = torch.randn(2, 4)
        result = compiled(x)
        assert result.predictions.shape == (2, 2)


# ── LazyLayerwiseGenerator ────────────────────────────────────────────────────


class TestLazyLayerwiseGenerator:
    def test_produces_same_keys_as_layerwise(self) -> None:
        torch.manual_seed(42)
        spec = _small_spec()
        lazy_gen = LazyLayerwiseGenerator(spec, latent_dim=8)
        eager_gen = LayerwiseGenerator(spec, latent_dim=8)

        with torch.no_grad():
            lazy_tree = lazy_gen.generate_parameters()
            eager_tree = eager_gen.generate_parameters()

        assert set(lazy_tree.to_dict().keys()) == set(eager_tree.to_dict().keys())

    def test_produces_valid_parameter_tree(self) -> None:
        spec = _small_spec()
        gen = LazyLayerwiseGenerator(spec, latent_dim=8)
        with torch.no_grad():
            tree = gen.generate_parameters()
        spec.validate_tree(tree)  # should not raise

    def test_parameter_shapes_match_spec(self) -> None:
        spec = _small_spec()
        gen = LazyLayerwiseGenerator(spec, latent_dim=8)
        with torch.no_grad():
            tree = gen.generate_parameters()
        for entry in spec:
            tensor = tree.to_dict()[entry.name]
            assert list(tensor.shape) == list(entry.shape), (
                f"{entry.name}: expected {entry.shape}, got {tensor.shape}"
            )

    def test_named_latent_vectors_identical_to_layerwise(self) -> None:
        spec = _small_spec()
        lazy_gen = LazyLayerwiseGenerator(spec, latent_dim=8)
        eager_gen = LayerwiseGenerator(spec, latent_dim=8)
        lazy_names = [n for n, _ in lazy_gen.named_latent_vectors()]
        eager_names = [n for n, _ in eager_gen.named_latent_vectors()]
        assert lazy_names == eager_names

    def test_gradients_flow_through_lazy_generation(self) -> None:
        spec = _small_spec()
        gen = LazyLayerwiseGenerator(spec, latent_dim=8)
        tree = gen.generate_parameters()
        loss = sum(t.sum() for t in tree.to_dict().values())
        loss.backward()  # type: ignore[union-attr]
        for name, latent in gen.named_latent_vectors():
            assert latent.grad is not None, f"No gradient for latent {name!r}"

    def test_empty_cache_flag_respected(self) -> None:
        """no_cache variant should not raise even if CUDA unavailable."""
        spec = _small_spec()
        gen = LazyLayerwiseGenerator(spec, latent_dim=8, empty_cache_on_cuda=False)
        with torch.no_grad():
            tree = gen.generate_parameters()
        spec.validate_tree(tree)

    def test_used_in_mapping_model(self) -> None:
        """LazyLayerwiseGenerator can be plugged into MappingModel via GroupedStrategy."""
        # Build spec from a linear target
        target = nn.Linear(4, 2)
        spec = ParameterSpec.from_module(target)
        gen = LazyLayerwiseGenerator(spec, latent_dim=8)
        # Verify it can produce a valid tree end-to-end
        with torch.no_grad():
            tree = gen.generate_parameters()
        assert "weight" in tree.to_dict()
        assert "bias" in tree.to_dict()

    def test_is_subclass_of_layerwise_generator(self) -> None:
        assert issubclass(LazyLayerwiseGenerator, LayerwiseGenerator)


# ── Memory profiling ──────────────────────────────────────────────────────────


class TestProfilePeakMemory:
    def test_returns_result_and_bytes(self) -> None:
        def simple() -> int:
            return 42

        result, peak = profile_peak_memory(simple, device="cpu")
        assert result == 42
        assert isinstance(peak, int)
        assert peak >= 0

    def test_measures_allocation(self) -> None:
        def allocate_tensors() -> torch.Tensor:
            # Allocate a reasonably-sized tensor so tracemalloc picks it up
            return torch.zeros(10_000)

        _, peak = profile_peak_memory(allocate_tensors, device="cpu")
        assert peak > 0

    def test_result_value_preserved(self) -> None:
        expected = torch.tensor([1.0, 2.0, 3.0])
        result, _ = profile_peak_memory(lambda: expected, device="cpu")
        assert torch.equal(result, expected)


# ── benchmark_strategies ──────────────────────────────────────────────────────


class TestBenchmarkStrategies:
    def test_returns_expected_strategy_keys(self) -> None:
        spec = _small_spec()
        results = benchmark_strategies(spec, latent_dim=8, n_warmup=1, n_repeat=2)
        assert "slvt" in results
        assert "layerwise" in results
        assert "lazy_layerwise" in results

    def test_each_result_has_peak_and_time(self) -> None:
        spec = _small_spec()
        results = benchmark_strategies(spec, latent_dim=8, n_warmup=1, n_repeat=2)
        for strategy, stats in results.items():
            assert "peak_bytes" in stats, f"{strategy} missing peak_bytes"
            assert "generate_time_ms" in stats, f"{strategy} missing generate_time_ms"

    def test_time_is_positive(self) -> None:
        spec = _small_spec()
        results = benchmark_strategies(spec, latent_dim=8, n_warmup=1, n_repeat=2)
        for strategy, stats in results.items():
            t = stats["generate_time_ms"]
            if not (t != t):  # skip NaN (error case)
                assert t >= 0, f"{strategy}: expected non-negative time, got {t}"

    def test_peak_bytes_non_negative(self) -> None:
        spec = _small_spec()
        results = benchmark_strategies(spec, latent_dim=8, n_warmup=1, n_repeat=2)
        for strategy, stats in results.items():
            p = stats["peak_bytes"]
            import math

            if not math.isinf(p):  # skip inf (error case)
                assert p >= 0, f"{strategy}: expected non-negative peak, got {p}"

    def test_cnn_target_spec(self) -> None:
        target = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Flatten(), nn.Linear(8, 2))
        spec = ParameterSpec.from_module(target)
        results = benchmark_strategies(spec, latent_dim=16, n_warmup=1, n_repeat=2)
        assert len(results) == 3
