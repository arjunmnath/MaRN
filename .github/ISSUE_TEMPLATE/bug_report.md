---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'
assignees: ''

---

**Describe the Bug**
A clear and concise description of what the bug is.

**Environment Information**
- OS: [e.g. macOS, Ubuntu 22.04]
- Python Version: [e.g. 3.12.3]
- PyTorch Version: [e.g. 2.6.0]
- Hardware/Device: [e.g. CPU, Single GPU (CUDA), Apple Silicon (MPS)]
- `marn` Version: [e.g. 0.1.0]

**Target Model Architecture**
Briefly describe the target model you are trying to map (e.g., standard ResNet18, custom CNN, linear layer, LSTM). Are there tied parameters, custom buffers, or `torch.compile` wrapped modules?

**Minimal Reproducible Example**
Provide a self-contained, minimal Python script that reproduces the bug. Use synthetic data where possible.

```python
import torch
from marn import MappingModel, MappingTrainer
# Add code here
```

**Traceback and Logs**
If applicable, paste the full traceback or warning logs here.

```text
# Paste error traceback
```

**Additional Context**
Add any other context about the problem here (e.g., specific mapper or modulation strategy, loss components used, etc.).
