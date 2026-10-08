# Quickstart

Get started with `marn` in 5 minutes.

## Minimal Implemented Example

Here is how to set up and train a target model using parameter mapping:

```python
import torch
from torch import nn
from marn import MappingModel, MappingLoss, ClassificationLoss, MappingTrainer

# 1. Define a target model
target_model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

# 2. Instantiate MappingModel
# This wraps the target model and generates its parameters from low-dimensional latent vectors.
model = MappingModel(target_model, latent_dim=64, strategy="layerwise")

# 3. Configure a trainer
trainer = MappingTrainer(
    model=model,
    train_loader=train_loader,  # Your training DataLoader
    loss_fn=MappingLoss(ClassificationLoss()),
)

# 4. Fit the model
trainer.fit(epochs=5)
```
