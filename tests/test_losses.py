"""Tests for the loss system: task, stability, smoothness, alignment, and composite."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from marn import (
    AlignmentLoss,
    ClassificationLoss,
    LossOutput,
    MappingLoss,
    RegressionLoss,
    SmoothnessLoss,
    StabilityLoss,
    TaskLoss,
    TrainingContext,
)
from marn.runtime import ParameterTree


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_context(
    predictions: Tensor | None = None,
    targets: Tensor | None = None,
    latent_vectors: dict[str, Tensor] | None = None,
    generated_parameters: ParameterTree | None = None,
    perturbed_predictions: Tensor | None = None,
    mapper_weights: dict[str, Tensor] | None = None,
) -> TrainingContext:
    """Build a minimal training context with sensible defaults."""
    if predictions is None:
        predictions = torch.randn(4, 3, requires_grad=True)
    if targets is None:
        targets = torch.randint(0, 3, (4,))
    if latent_vectors is None:
        z = torch.randn(8, requires_grad=True)
        latent_vectors = {"layer0": z}
    if generated_parameters is None:
        # Create a simple parameter tree that retains grad from latents
        w = torch.randn(3, 4, requires_grad=True)
        b = torch.randn(3, requires_grad=True)
        generated_parameters = ParameterTree({"weight": w, "bias": b})
    return TrainingContext(
        predictions=predictions,
        targets=targets,
        latent_vectors=latent_vectors,
        generated_parameters=generated_parameters,
        perturbed_predictions=perturbed_predictions,
        mapper_weights=mapper_weights or {},
    )


def _make_linked_context() -> TrainingContext:
    """Build a context where generated parameters depend nonlinearly on latent vectors.

    Uses a nonlinear mapping (tanh) so the Jacobian depends on z and
    smoothness loss gradients flow back to z.
    """
    z = torch.randn(4, requires_grad=True)
    # Nonlinear mapping: generated = W @ tanh(z) — Jacobian depends on z
    W = torch.randn(6, 4)
    generated_flat = W @ torch.tanh(z)
    weight = generated_flat[:4].view(2, 2)
    bias = generated_flat[4:]
    tree = ParameterTree({"weight": weight, "bias": bias})
    predictions = torch.randn(2, 2, requires_grad=True)
    targets = torch.randn(2, 2)
    return TrainingContext(
        predictions=predictions,
        targets=targets,
        latent_vectors={"layer0": z},
        generated_parameters=tree,
    )


# ── Task losses ──────────────────────────────────────────────────────────


class TestTaskLossClassification:
    def test_cross_entropy_output(self) -> None:
        context = _make_context()
        loss = ClassificationLoss()
        output = loss(context)

        assert isinstance(output, LossOutput)
        assert output.total.ndim == 0  # scalar
        assert "task" in output.components
        assert "task" in output.metrics
        assert "accuracy" in output.metrics
        assert 0.0 <= output.metrics["accuracy"] <= 1.0

    def test_cross_entropy_gradient(self) -> None:
        predictions = torch.randn(4, 3, requires_grad=True)
        context = _make_context(predictions=predictions)
        loss = ClassificationLoss()
        output = loss(context)
        output.total.backward()
        assert predictions.grad is not None

    def test_label_smoothing(self) -> None:
        context = _make_context()
        loss_no_smooth = ClassificationLoss(label_smoothing=0.0)
        loss_smooth = ClassificationLoss(label_smoothing=0.1)
        out1 = loss_no_smooth(context)
        out2 = loss_smooth(context)
        # Different label smoothing should produce different losses
        # (not guaranteed, but extremely likely with random inputs)
        assert out1.total.item() != out2.total.item()


class TestTaskLossRegression:
    def test_mse_output(self) -> None:
        predictions = torch.randn(4, 3, requires_grad=True)
        targets = torch.randn(4, 3)
        context = _make_context(predictions=predictions, targets=targets)
        loss = RegressionLoss()
        output = loss(context)

        assert isinstance(output, LossOutput)
        assert output.total.ndim == 0
        assert output.total.item() >= 0
        assert "task" in output.components
        assert "task" in output.metrics

    def test_mse_gradient(self) -> None:
        predictions = torch.randn(4, 3, requires_grad=True)
        targets = torch.randn(4, 3)
        context = _make_context(predictions=predictions, targets=targets)
        loss = RegressionLoss()
        output = loss(context)
        output.total.backward()
        assert predictions.grad is not None


class TestTaskLossCustomCallable:
    def test_custom_loss_fn(self) -> None:
        predictions = torch.randn(4, 3, requires_grad=True)
        targets = torch.randn(4, 3)
        context = _make_context(predictions=predictions, targets=targets)
        loss = TaskLoss(F.l1_loss)
        output = loss(context)

        assert output.total.item() >= 0
        output.total.backward()
        assert predictions.grad is not None


# ── Stability loss ───────────────────────────────────────────────────────


class TestStabilityLoss:
    def test_output(self) -> None:
        predictions = torch.randn(4, 3)
        perturbed = predictions + torch.randn_like(predictions) * 0.1
        context = _make_context(predictions=predictions, perturbed_predictions=perturbed)
        loss = StabilityLoss(epsilon=0.01)
        output = loss(context)

        assert isinstance(output, LossOutput)
        assert output.total.item() > 0
        assert "stability" in output.components
        assert "stability" in output.metrics

    def test_missing_perturbed_predictions_raises(self) -> None:
        context = _make_context()
        loss = StabilityLoss()
        with pytest.raises(ValueError, match="perturbed_predictions"):
            loss(context)

    def test_identical_predictions_give_zero_loss(self) -> None:
        predictions = torch.randn(4, 3)
        context = _make_context(predictions=predictions, perturbed_predictions=predictions)
        loss = StabilityLoss()
        output = loss(context)
        assert output.total.item() == pytest.approx(0.0, abs=1e-7)

    def test_invalid_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            StabilityLoss(epsilon=0.0)

    def test_invalid_num_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            StabilityLoss(num_samples=0)


# ── Smoothness loss ──────────────────────────────────────────────────────


class TestSmoothnessLoss:
    def test_exact_non_negative(self) -> None:
        context = _make_linked_context()
        loss = SmoothnessLoss(method="exact")
        output = loss(context)

        assert output.total.item() >= 0
        assert "smoothness" in output.components
        assert "smoothness" in output.metrics

    def test_stochastic_gradient_flow(self) -> None:
        context = _make_linked_context()
        loss = SmoothnessLoss(method="stochastic", num_projections=2)
        output = loss(context)

        assert output.total.item() >= 0
        output.total.backward()
        z = context.latent_vectors["layer0"]
        assert z.grad is not None

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="exact.*stochastic"):
            SmoothnessLoss(method="invalid")  # type: ignore[arg-type]

    def test_invalid_num_projections_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            SmoothnessLoss(num_projections=0)


# ── Alignment loss ───────────────────────────────────────────────────────


class TestAlignmentLoss:
    def test_cosine_distance_valid(self) -> None:
        z = torch.randn(8, requires_grad=True)
        weight = torch.randn(16, 8)
        context = _make_context(
            latent_vectors={"layer0": z},
            mapper_weights={"layer0": weight},
        )
        loss = AlignmentLoss()
        output = loss(context)

        assert isinstance(output, LossOutput)
        # Cosine distance is in [0, 2]
        assert 0 <= output.total.item() <= 2.0
        assert "alignment" in output.components
        assert "alignment" in output.metrics

    def test_missing_mapper_weights_raises(self) -> None:
        context = _make_context(mapper_weights={})
        loss = AlignmentLoss()
        with pytest.raises(ValueError, match="mapper_weights"):
            loss(context)

    def test_gradient_flow(self) -> None:
        z = torch.randn(8, requires_grad=True)
        weight = torch.randn(16, 8)
        context = _make_context(
            latent_vectors={"layer0": z},
            mapper_weights={"layer0": weight},
        )
        loss = AlignmentLoss()
        output = loss(context)
        output.total.backward()
        assert z.grad is not None


# ── Composite MappingLoss ────────────────────────────────────────────────


class TestMappingLossComposition:
    def test_all_components_sum(self) -> None:
        predictions = torch.randn(4, 3, requires_grad=True)
        perturbed = predictions + torch.randn_like(predictions) * 0.1
        z = torch.randn(4, requires_grad=True)
        W = torch.randn(6, 4)
        gen_flat = W @ z
        tree = ParameterTree(
            {
                "weight": gen_flat[:4].view(2, 2),
                "bias": gen_flat[4:],
            }
        )
        weight_matrix = torch.randn(8, 4)
        context = TrainingContext(
            predictions=predictions,
            targets=torch.randn(4, 3),
            latent_vectors={"layer0": z},
            generated_parameters=tree,
            perturbed_predictions=perturbed,
            mapper_weights={"layer0": weight_matrix},
        )

        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=StabilityLoss(epsilon=0.01),
            smoothness_loss=SmoothnessLoss(method="stochastic", num_projections=1),
            alignment_loss=AlignmentLoss(),
            lambda_stability=0.1,
            lambda_smoothness=0.01,
            lambda_alignment=0.01,
        )
        output = loss_fn(context)

        assert "task" in output.components
        assert "stability" in output.components
        assert "smoothness" in output.components
        assert "alignment" in output.components
        assert output.total.ndim == 0

    def test_disabled_components_produce_task_only(self) -> None:
        predictions = torch.randn(4, 3, requires_grad=True)
        targets = torch.randn(4, 3)
        context = _make_context(predictions=predictions, targets=targets)

        # Explicitly disable all regularization components
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=None,
            smoothness_loss=None,
            alignment_loss=None,
        )
        output = loss_fn(context)

        assert "task" in output.components
        assert "stability" not in output.components
        assert "smoothness" not in output.components
        assert "alignment" not in output.components

    def test_auto_defaults_create_all_components(self) -> None:
        """MappingLoss() with no explicit losses should auto-create defaults."""
        loss_fn = MappingLoss(task_loss=RegressionLoss())
        assert loss_fn.stability_loss is not None
        assert loss_fn.smoothness_loss is not None
        assert loss_fn.alignment_loss is not None

    def test_auto_with_explicit_none_disables(self) -> None:
        """Passing None explicitly should disable that component."""
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=None,
            smoothness_loss=None,
            alignment_loss=None,
        )
        assert loss_fn.stability_loss is None
        assert loss_fn.smoothness_loss is None
        assert loss_fn.alignment_loss is None

    def test_auto_mixed_override(self) -> None:
        """Can mix auto, None, and custom losses."""
        custom_stability = StabilityLoss(epsilon=0.05)
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=custom_stability,
            smoothness_loss=None,
            # alignment_loss defaults to auto
        )
        assert loss_fn.stability_loss is custom_stability
        assert loss_fn.smoothness_loss is None
        assert loss_fn.alignment_loss is not None  # auto-created

    def test_trainable_coefficients(self) -> None:
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            trainable_coefficients=True,
        )
        # Should have 3 trainable coefficient parameters
        param_names = [name for name, _ in loss_fn.named_parameters()]
        assert "_raw_lambda_stability" in param_names
        assert "_raw_lambda_smoothness" in param_names
        assert "_raw_lambda_alignment" in param_names

        # Coefficients should be non-negative (softplus)
        assert loss_fn.lambda_stability >= 0
        assert loss_fn.lambda_smoothness >= 0
        assert loss_fn.lambda_alignment >= 0

    def test_static_coefficients_no_parameters(self) -> None:
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            trainable_coefficients=False,
        )
        # Should have no trainable parameters from coefficients
        coeff_params = [name for name, _ in loss_fn.named_parameters() if "lambda" in name]
        assert len(coeff_params) == 0

    def test_loss_metrics_present(self) -> None:
        predictions = torch.randn(4, 3, requires_grad=True)
        targets = torch.randn(4, 3)
        perturbed = predictions + torch.randn_like(predictions) * 0.1
        context = _make_context(
            predictions=predictions,
            targets=targets,
            perturbed_predictions=perturbed,
        )

        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=StabilityLoss(epsilon=0.01),
            smoothness_loss=None,
            alignment_loss=None,
            lambda_stability=0.5,
        )
        output = loss_fn(context)

        assert "task" in output.metrics
        assert "stability" in output.metrics
        assert "lambda_stability" in output.metrics
        assert output.metrics["lambda_stability"] == pytest.approx(0.5)


class TestGradientFlowThroughComponents:
    def test_task_loss_gradient_flow(self) -> None:
        predictions = torch.randn(4, 3, requires_grad=True)
        targets = torch.randn(4, 3)
        context = _make_context(predictions=predictions, targets=targets)
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=None,
            smoothness_loss=None,
            alignment_loss=None,
        )
        output = loss_fn(context)
        output.total.backward()
        assert predictions.grad is not None

    def test_stability_loss_gradient_flow(self) -> None:
        predictions = torch.randn(4, 3, requires_grad=True)
        perturbed = predictions + torch.randn(4, 3) * 0.1
        context = _make_context(
            predictions=predictions,
            targets=torch.randn(4, 3),
            perturbed_predictions=perturbed,
        )
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=StabilityLoss(),
            smoothness_loss=None,
            alignment_loss=None,
        )
        output = loss_fn(context)
        output.total.backward()
        assert predictions.grad is not None

    def test_smoothness_loss_latent_gradient(self) -> None:
        context = _make_linked_context()
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=None,
            smoothness_loss=SmoothnessLoss(method="stochastic", num_projections=1),
            alignment_loss=None,
        )
        output = loss_fn(context)
        output.total.backward()
        z = context.latent_vectors["layer0"]
        assert z.grad is not None

    def test_alignment_loss_latent_gradient(self) -> None:
        z = torch.randn(8, requires_grad=True)
        weight = torch.randn(16, 8)
        predictions = torch.randn(4, 3, requires_grad=True)
        context = _make_context(
            predictions=predictions,
            targets=torch.randn(4, 3),
            latent_vectors={"layer0": z},
            mapper_weights={"layer0": weight},
        )
        loss_fn = MappingLoss(
            task_loss=RegressionLoss(),
            stability_loss=None,
            smoothness_loss=None,
            alignment_loss=AlignmentLoss(),
        )
        output = loss_fn(context)
        output.total.backward()
        assert z.grad is not None
