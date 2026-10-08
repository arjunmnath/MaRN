# Cookbook

The [cookbook](https://github.com/arjunmnath/MaRN/tree/main/cookbook) directory in the repository
contains **eight standalone training scripts**. Each recipe uses synthetic or bundled data so you
can run it immediately after installing `marn`. They mirror the [user guide](../guides/index.md)
and exercise the public APIs in the [API reference](../api/index.md).

```{toctree}
:maxdepth: 1
:hidden:

01_single_latent_classification
02_layerwise_training
03_mapping_loss_regularization
04_low_rank_decomposition
05_efficient_finetuning
06_pruning_and_quantization
07_advanced_custom_components
08_checkpointing_and_resumption
```

## How to run

From the repository root (with dependencies installed):

```bash
poetry run python cookbook/01_single_latent_classification.py
```

Replace the script name for other recipes. The same commands work with `pip install -e .` and
`python cookbook/...` if you are not using Poetry.

## All recipes

| # | Script | Concepts |
| --- | --- | --- |
| 1 | [Single Latent Classification](01_single_latent_classification.md) | SLVT, CNN target, `ClassificationLoss`, trainer loop, callbacks |
| 2 | [Layerwise Training](02_layerwise_training.md) | Layerwise vs. SLVT scaling and memory |
| 3 | [Mapping Loss Regularization](03_mapping_loss_regularization.md) | Stability, smoothness, alignment weights |
| 4 | [Low Rank Decomposition](04_low_rank_decomposition.md) | LRD strategy for large linear/conv layers |
| 5 | [Efficient Finetuning](05_efficient_finetuning.md) | Block-wise modulation scale (PEFT-style) |
| 6 | [Pruning and Quantization](06_pruning_and_quantization.md) | Structural pruning plus mapped training |
| 7 | [Advanced Custom Components](07_advanced_custom_components.md) | Custom mapper/modulation and registry hooks |
| 8 | [Checkpointing and Resumption](08_checkpointing_and_resumption.md) | Save/load, bitwise resume, continued training |

## Map recipes to documentation

| Recipe | Related guides |
| --- | --- |
| 01, 02 | [Generators and strategies](../guides/generators-and-strategies.md), [MappingModel](../guides/mapping-model.md), [Trainer guide](../guides/trainer-guide.md) |
| 03 | [Loss system](../guides/loss-system.md) |
| 04 | [Generators and strategies](../guides/generators-and-strategies.md), [Mappers and modulation](../guides/mappers-and-modulation.md) |
| 05 | [Generators and strategies](../guides/generators-and-strategies.md) |
| 06 | [Scaling](../guides/scaling.md) |
| 07 | [Extension recipes](../guides/extension-recipes.md), [Configuration and registries](../guides/config-and-registries.md) |
| 08 | [Checkpointing](../guides/checkpointing.md) |
