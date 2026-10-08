import pytest
import torch
from torch import nn

from marn import (
    LRDStrategy,
    LRDGenerator,
    MappingModel,
    ParameterSpec,
)


def test_lrd_generator_shapes_and_reconstruction() -> None:
    # 1. Create a tiny linear layer
    target = nn.Linear(4, 4)

    # 2. Build MappingModel using LRDStrategy
    latent_dim = 8
    rank = 2
    strategy = LRDStrategy(rank=rank)
    model = MappingModel(target, latent_dim=latent_dim, strategy=strategy)

    # 3. Verify that the generator was created with correct internal configuration
    generator = model.generator
    assert isinstance(generator, LRDGenerator)
    assert generator.rank == rank

    # linear.weight is [4, 4], so rank=2 < min(4,4) -> decomposed
    # linear.bias is [4] -> not decomposed
    # Weight mapper output should be: rank * (4 + 4) = 16
    # Bias mapper output should be: 4
    # Total generated parameters should be 20.
    # Check that layers match these dimensions:
    layers = list(generator.layers)
    assert len(layers) == 1  # grouped layerwise, single module "linear" (or "<root>")
    assert layers[0].output_dim == 20  # 16 (for weight LRD) + 4 (for bias)

    # 4. Test forward pass
    X = torch.randn(3, 4)
    result = model(X)
    assert result.predictions.shape == (3, 4)

    # 5. Verify the generated weight is a low-rank matrix W = U @ V^T
    gen_dict = result.generated_parameters.to_dict()
    W = gen_dict["weight"]
    assert W.shape == (4, 4)

    # Since rank is 2, the rank of W must be at most 2.
    # We can check its singular values: the last 2 singular values should be 0 (or close to 0)
    U, S, V = torch.linalg.svd(W)
    assert S[2].item() < 1e-5
    assert S[3].item() < 1e-5


def test_invalid_rank_raises() -> None:
    target = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="rank must be positive"):
        LRDGenerator(
            parameter_spec=ParameterSpec.from_module(target),
            latent_dim=8,
            rank=0,
        )

    with pytest.raises(ValueError, match="rank must be positive"):
        LRDStrategy(rank=-1)
