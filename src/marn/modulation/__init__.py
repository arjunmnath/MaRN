"""Latent-conditioned modulation strategies."""

from marn.modulation.additive import AdditiveModulation
from marn.modulation.affine import AffineModulation
from marn.modulation.base import BaseModulation
from marn.modulation.low_rank import LowRankModulation

__all__ = ["AdditiveModulation", "AffineModulation", "BaseModulation", "LowRankModulation"]
