# Mapping Networks Cookbook

Welcome to the `marn` Cookbook. This collection provides full-fledged, runnable training examples illustrating growing complexity, advanced usage, and custom extension patterns of the mapping networks library.

All examples use self-contained synthetic datasets or standard targets so they can be run immediately without downloading external data.

## Table of Contents

1. **[01_single_latent_classification.py](file:///Users/arjunmnath/dev/marn/cookbook/01_single_latent_classification.py)**
   - **Concepts**: Single Latent Vector Training (SLVT), CNN target models, `ClassificationLoss`, dataset pipelines, basic training/validation loop, and logging.
   - **Difficulty**: Beginner

2. **[02_layerwise_training.py](file:///Users/arjunmnath/dev/marn/cookbook/02_layerwise_training.py)**
   - **Concepts**: Scaling up models with Layer-wise Training (LWT) to handle deep architectures, comparison of parameter/memory scaling between SLVT and LWT, and optimization dynamics.
   - **Difficulty**: Intermediate

3. **[03_mapping_loss_regularization.py](file:///Users/arjunmnath/dev/marn/cookbook/03_mapping_loss_regularization.py)**
   - **Concepts**: Composite `MappingLoss` configuration with $L_{\text{stability}}$, $L_{\text{smoothness}}$, and $L_{\text{alignment}}$. Tuning regularization lambdas, using trainable softplus coefficients.
   - **Difficulty**: Intermediate

4. **[04_low_rank_decomposition.py](file:///Users/arjunmnath/dev/marn/cookbook/04_low_rank_decomposition.py)**
   - **Concepts**: Low Rank Decomposition (LRD) strategy for large linear/convolutional layers, reconstructing parameters as $W \approx U V^T$, and training mappers on factored weights.
   - **Difficulty**: Advanced

5. **[05_efficient_finetuning.py](file:///Users/arjunmnath/dev/marn/cookbook/05_efficient_finetuning.py)**
   - **Concepts**: Parameter-efficient fine-tuning (PEFT) on pretrained models. Applying block-wise modulation scale ($\alpha$) to freeze the core model and train a low-dimensional mapping network.
   - **Difficulty**: Advanced

6. **[06_pruning_and_quantization.py](file:///Users/arjunmnath/dev/marn/cookbook/06_pruning_and_quantization.py)**
   - **Concepts**: Combining structural network pruning (via `torch.nn.utils.prune`) with parameter mappings to show orthogonal training/inference size reductions.
   - **Difficulty**: Advanced

7. **[07_advanced_custom_components.py](file:///Users/arjunmnath/dev/marn/cookbook/07_advanced_custom_components.py)**
   - **Concepts**: Custom extension of the library. Implementing a custom mapper subclass (`CustomSinusoidalMapper`), a custom modulation scheme, and registering custom components.
   - **Difficulty**: Expert

8. **[08_checkpointing_and_resumption.py](file:///Users/arjunmnath/dev/marn/cookbook/08_checkpointing_and_resumption.py)**
   - **Concepts**: Saving and loading model/trainer states. Implementing checkpointing, verifying exact prediction matches, and continuing training to completion.
   - **Difficulty**: Intermediate

## How to Run the Recipes

Ensure you are inside the virtual environment:

```bash
poetry run python cookbook/01_single_latent_classification.py
```
