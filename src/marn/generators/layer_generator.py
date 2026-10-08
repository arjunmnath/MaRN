"""One independently mapped generated-parameter group."""

import math

import torch
from torch import Tensor, nn

from marn.mappers import BaseMapper


class LayerGenerator(nn.Module):
    """Own one trainable latent and a mapper for one flat parameter group."""

    def __init__(self, latent_dim: int, output_dim: int, mapper: BaseMapper) -> None:
        super().__init__()
        if mapper.latent_dim != latent_dim or mapper.output_dim != output_dim:
            raise ValueError("Mapper dimensions do not match the layer generator")
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.latent = nn.Parameter(torch.empty(latent_dim))
        nn.init.normal_(self.latent, std=1.0 / math.sqrt(latent_dim))
        self.mapper = mapper

    def forward(self, latent: Tensor | None = None) -> Tensor:
        input_latent = latent if latent is not None else self.latent
        descriptor: Tensor = self.mapper(input_latent)
        if descriptor.ndim != 1 or descriptor.numel() != self.output_dim:
            raise RuntimeError(
                f"Mapper returned shape {tuple(descriptor.shape)}; expected ({self.output_dim},)"
            )
        return descriptor
