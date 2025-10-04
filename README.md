# MLP-KAN

PyTorch layers for building Kolmogorov–Arnold Networks (KAN) with either learnable sinusoidal features or an interpolation-based parameterisation. The implementation relies on `torch.vmap` for efficient evaluation and exposes utilities to encourage smoothness in interpolated activations.

## Installation

The project follows the PEP 517 `pyproject.toml` layout. Install the package in editable mode while developing:

```bash
pip install -e .
```

A pre-existing PyTorch 2.0 (or later) environment is required.

## Quickstart

```python
import torch
import torch.nn as nn
from mlp_kan import KANLayer, KANInterpoLayer, smooth_penalty

# Sinusoidal feature KAN layer stack
model = nn.Sequential(
    KANLayer(2, 5),
    KANLayer(5, 1),
)

x = torch.randn(16, 2)
y = model(x)
assert y.shape == (16, 1)

# Interpolation-based layer with optional smoothness penalty
interp_model = nn.Sequential(
    KANInterpoLayer(2, 5, num_x=128, x_min=-3, x_max=3),
    KANInterpoLayer(5, 1, num_x=256),
)
penalty = smooth_penalty(interp_model)
```

## Experiments

Two reference scripts illustrate training and visualisation workflows:

- `experiment.py` trains a small model using sinusoidal features.
- `experiment_interpolation.py` explores the interpolation-based variant with the smoothness regulariser.

Run either script after installing the project in editable mode. Generated plots are saved to the `images`/`temp` folders referenced in the notebooks.

## Visualisation

The repository includes example plots captured during the experiments, demonstrating how individual activations learn components of the target function.
