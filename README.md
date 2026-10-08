# Manifold Regularized Networks (MaRN)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-%3E%3D2.6%20%7C%20%3C3.0-orange)](https://pytorch.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Documentation Status](https://readthedocs.org/projects/mapping-networks/badge/?version=latest)](https://mapping-networks.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2602.19134-b31b1b.svg)](https://arxiv.org/abs/2602.19134)

`marn` is a model-agnostic, production-oriented PyTorch package for training target models through low-dimensional parameter manifolds. It is inspired by the paper [**Mapping Networks**](https://doi.org/10.48550/arXiv.2602.19134). The library decouples model architecture from parameter representation, allowing you to optimize neural networks by updating compact, trainable latent coordinates instead of mutating the target module's parameters.

---

## Installation

Install `marn` from your local workspace:

```bash
pip install marn
```

Or add it using Poetry:

```bash
poetry add marn
```

---

## Quickstart

Train a standard PyTorch model through a low-dimensional layer-wise latent representation:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from marn import MappingModel, MappingLoss, ClassificationLoss, MappingTrainer

# 1. Create target model and training data
target = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
loader = DataLoader(data, batch_size=16)

# 2. Wrap target model with low-dimensional mapping strategy
model = MappingModel(
    target_model=target,
    latent_dim=32,
    strategy="layerwise"
)

# 3. Configure composite loss and trainer
loss_fn = MappingLoss(task_loss=ClassificationLoss())
trainer = MappingTrainer(
    model=model,
    train_loader=loader,
    loss_fn=loss_fn,
    learning_rate=1e-3,
)

# 4. Train the latent parameters
trainer.fit(epochs=5)
```

---

## Core Concepts

- **Latent Manifold**: Trainable coordinates $z$ which undergo transformation to map to target parameter space.
- **BaseMapper**: Projects $z$ to flat generated parameter descriptors (e.g., using fixed, orthogonal projections).
- **BaseModulation**: Integrates generated descriptors back into target parameters (e.g., additive $W_{ij} \leftarrow W_{ij} + \alpha z_i$, or affine).
- **Generation Strategy**: Defines the mapping scope. 
  - `"slvt"` (Single Latent Vector Training) projects the entire model from a single global latent vector.
  - `"layerwise"` constructs independent smaller latent coordinates per layer.
  - `"grouped"` allows custom parameter subdivision.
- **MappingLoss**: A composite loss that balances task loss (classification/regression) with stability, smoothness, and cosine alignment regularization components.

---

## Paper Correspondence

The code maps directly to the concepts defined in the paper:
- **Fixed Projection Matrices**: Registered as PyTorch buffers inside `MLPMapper` so they stay frozen and are excluded from DDP/optimizer updates.
- **Additive Modulation**: Implemented in `AdditiveModulation` representing $w_{ij} \leftarrow w_{ij} + \alpha \cdot z_i$.
- **Regularization Terms**: Fully implemented in `MappingLoss`:
  - **Stability Loss** ($L_{\text{stability}}$): penalizes changes in output predictions when adding small noise perturbation to the latent vector (`StabilityLoss`).
  - **Smoothness Loss** ($L_{\text{smoothness}}$): penalizes the Jacobian norm of the mapper to enforce a smooth manifold (`SmoothnessLoss`).
  - **Alignment Loss** ($L_{\text{alignment}}$): maximizes alignment via cosine distance between latent vectors and weight summaries (`AlignmentLoss`).

