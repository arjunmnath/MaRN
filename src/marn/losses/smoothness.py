"""Mapper Jacobian smoothness penalties.

Penalizes the norm of the Jacobian of the mapping from latent vectors to
generated parameters, encouraging smooth parameter manifolds.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from marn.losses.base import BaseLoss
from marn.losses.context import TrainingContext
from marn.losses.outputs import LossOutput


def _exact_jacobian_frobenius_squared(
    generated_flat: Tensor, latent_vectors: dict[str, Tensor]
) -> Tensor:
    """Compute ||J||^2_F exactly via per-element backward passes.

    For each output element, compute gradients w.r.t. all latent vectors and
    sum their squared norms.  This is O(output_dim) backward passes and is
    only practical for small generators.
    """
    total = torch.tensor(0.0, device=generated_flat.device, dtype=generated_flat.dtype)
    latents = list(latent_vectors.values())

    for i in range(generated_flat.numel()):
        grads = torch.autograd.grad(
            generated_flat[i],
            latents,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        for grad in grads:
            if grad is not None:
                total = total + grad.square().sum()
    return total


def _stochastic_jacobian_frobenius_squared(
    generated_flat: Tensor,
    latent_vectors: dict[str, Tensor],
    num_projections: int,
) -> Tensor:
    """Estimate ||J||^2_F via Hutchinson's trace estimator.

    Uses ``num_projections`` random Rademacher vectors ``v`` to estimate
    ``E[||Jv||^2] = E[v^T J^T J v] = ||J||^2_F``.  Each projection requires
    one backward pass through the generator graph.
    """
    latents = list(latent_vectors.values())
    contributions: list[Tensor] = []

    for _ in range(num_projections):
        # Rademacher random vector
        v = torch.sign(torch.randn_like(generated_flat))
        # Compute v^T J by backward pass: grad of (generated_flat . v) w.r.t. latents
        # Use dot product to get a scalar, then take grad of that scalar
        scalar_product = (generated_flat * v).sum()
        grads = torch.autograd.grad(
            scalar_product,
            latents,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        for grad in grads:
            if grad is not None:
                contributions.append(grad.square().sum())

    if not contributions:
        return torch.tensor(0.0, device=generated_flat.device, dtype=generated_flat.dtype)
    return torch.stack(contributions).sum() / num_projections


class SmoothnessLoss(BaseLoss):
    """Penalize the Frobenius norm of the mapper Jacobian.

    Two methods are available:

    - ``"exact"``: Computes ``||J||^2_F`` exactly.  Cost is O(output_dim)
      backward passes.  Use only when the total generated parameter count
      is small (e.g. < 1000).
    - ``"stochastic"``: Hutchinson's trace estimator with configurable
      random projections.  Cost is O(num_projections) backward passes.
      Recommended for ``latent_dim > 128``.

    The loss reads ``context.generated_parameters`` and
    ``context.latent_vectors``. The generated parameters must retain their
    computational graph (i.e. not be detached).

    Args:
        method: ``"exact"`` or ``"stochastic"``.
        num_projections: Number of random projections for the stochastic
            estimator.  Ignored when ``method="exact"``.
    """

    def __init__(
        self,
        method: Literal["exact", "stochastic"] = "stochastic",
        num_projections: int = 4,
    ) -> None:
        super().__init__()
        if method not in ("exact", "stochastic"):
            raise ValueError(f"method must be 'exact' or 'stochastic', got {method!r}")
        if num_projections < 1:
            raise ValueError("num_projections must be at least 1")
        self.method = method
        self.num_projections = num_projections

    def forward(self, context: TrainingContext) -> LossOutput:
        # Flatten all generated parameters into one vector
        flat_parts = [tensor.reshape(-1) for tensor in context.generated_parameters.values()]
        generated_flat = torch.cat(flat_parts)

        # Jacobian estimation requires autograd.  During validation (under
        # torch.no_grad) or when the generated parameters are detached, we
        # cannot compute gradients — return zero gracefully.
        if not generated_flat.requires_grad or not torch.is_grad_enabled():
            zero = torch.tensor(0.0, device=generated_flat.device, dtype=generated_flat.dtype)
            return LossOutput(
                total=zero,
                components={"smoothness": zero},
                metrics={"smoothness": 0.0},
            )

        if self.method == "exact":
            value = _exact_jacobian_frobenius_squared(generated_flat, context.latent_vectors)
        else:
            value = _stochastic_jacobian_frobenius_squared(
                generated_flat, context.latent_vectors, self.num_projections
            )

        total_latent_dim = sum(z.numel() for z in context.latent_vectors.values())
        if total_latent_dim > 0:
            value = value / total_latent_dim

        return LossOutput(
            total=value,
            components={"smoothness": value},
            metrics={"smoothness": value.detach().item()},
        )
