import pytest
import torch
from torch import nn

from marn import (
    FineTuningStrategy,
    FineTuningGenerator,
    MappingModel,
    ParameterSpec,
)


def test_finetuning_generator_shapes_and_modulation() -> None:
    # 1. Create a tiny target model
    target = nn.Linear(4, 2)
    # Get original parameter values
    pretrained_params = {name: param.clone() for name, param in target.named_parameters()}

    # 2. Build MappingModel using FineTuningStrategy
    latent_dim = 8
    L = 2
    alpha = 0.1
    strategy = FineTuningStrategy(pretrained_params, L=L, alpha=alpha)
    model = MappingModel(target, latent_dim=latent_dim, strategy=strategy)

    # 3. Test forward pass
    X = torch.randn(3, 4)
    result = model(X)

    # Verify shape of predictions
    assert result.predictions.shape == (3, 2)

    # 4. Verify generated parameter values
    # Weight has size 8 (2x4). L=2, so we generate 4 modulation elements.
    # Bias has size 2 (2). L=2, so we generate 1 modulation element.
    # Total trainable parameters:
    # Weight group size = 8. Output dim of mapper = 4.
    # Bias group size = 2. Output dim of mapper = 1.
    # Latent vectors: "weight" and "bias" or grouped by module.
    # In layerwise/finetuning strategy, it groups by owning module: "linear" (or "<root>")
    # Let's check the group name:
    assert list(result.latent_vectors.keys()) == ["<root>"]

    # Verify that the generated parameters are modulated from pre-trained
    gen_dict = result.generated_parameters.to_dict()
    for name, pretrained in pretrained_params.items():
        generated = gen_dict[name]
        # Generated should be different from pretrained
        assert not torch.allclose(generated, pretrained)

        # Calculate the actual applied modulation
        diff = (generated - pretrained) / alpha

        # Every L consecutive elements in the flattened diff should be identical (expanded by interleave)
        flat_diff = diff.reshape(-1)
        for i in range(0, flat_diff.numel(), L):
            chunk = flat_diff[i : min(i + L, flat_diff.numel())]
            if chunk.numel() > 1:
                assert torch.allclose(chunk[:-1], chunk[1:])


def test_invalid_block_size_raises() -> None:
    target = nn.Linear(4, 2)
    pretrained = {name: param.clone() for name, param in target.named_parameters()}

    with pytest.raises(ValueError, match="Block size L must be positive"):
        FineTuningGenerator(
            parameter_spec=ParameterSpec.from_module(target),
            latent_dim=8,
            pretrained_parameters=pretrained,
            L=0,
        )
