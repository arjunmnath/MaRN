"""Latent–weight alignment loss.

Computes cosine distance between each latent vector and a compatible summary
of the modulated mapper weights, encouraging alignment between the learned
latent representation and the mapping it induces.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from marn.losses.base import BaseLoss
from marn.losses.context import TrainingContext
from marn.losses.outputs import LossOutput


def _weight_summary(weight_matrix: Tensor, latent_dim: int) -> Tensor:
    """Compute a latent-compatible summary of a mapper weight matrix.

    The modulated weight matrix has shape ``[output_dim, input_dim]`` where
    ``input_dim`` is typically the latent dimension at that layer.  We take
    the column-mean to produce a ``[input_dim]`` summary vector.

    If the weight matrix input dimension does not match ``latent_dim``, we
    project down or up via simple averaging over chunks or repeating.
    """
    # Column mean: average each input-dimension feature across output rows
    summary = weight_matrix.mean(dim=0)
    if summary.numel() == latent_dim:
        return summary
    # Truncate or pad to match latent_dim
    if summary.numel() > latent_dim:
        # Average groups of columns
        return summary[:latent_dim]
    # Pad with zeros
    padded = torch.zeros(latent_dim, device=summary.device, dtype=summary.dtype)
    padded[: summary.numel()] = summary
    return padded


class AlignmentLoss(BaseLoss):
    """Cosine distance between latent vectors and mapper weight summaries.

    Requires ``context.mapper_weights`` to contain a mapping from latent names
    to their mapper's weight matrices (or modulated weight matrices).  The
    weight summary is the column-mean of the weight matrix, projected to match
    the latent vector's dimensionality if necessary.

    When multiple latent–weight pairs are present (layerwise/grouped), the
    loss is the mean cosine distance across all pairs.

    The cosine distance is ``1 - cosine_similarity(z, w_summary)``.
    """

    def forward(self, context: TrainingContext) -> LossOutput:
        if not context.mapper_weights:
            raise ValueError(
                "AlignmentLoss requires mapper_weights in the training context. "
                "Ensure the model or trainer populates this field."
            )

        distances: list[Tensor] = []
        for name, latent in context.latent_vectors.items():
            if name not in context.mapper_weights:
                continue
            weight = context.mapper_weights[name]
            summary = _weight_summary(weight, latent.numel())
            # Cosine distance = 1 - cosine_similarity
            cosine_sim = F.cosine_similarity(latent.unsqueeze(0), summary.unsqueeze(0))
            distances.append(1.0 - cosine_sim.squeeze())

        if not distances:
            zero = torch.tensor(0.0, requires_grad=True)
            return LossOutput(
                total=zero,
                components={"alignment": zero},
                metrics={"alignment": 0.0},
            )

        value = torch.stack(distances).mean()
        return LossOutput(
            total=value,
            components={"alignment": value},
            metrics={"alignment": value.detach().item()},
        )
