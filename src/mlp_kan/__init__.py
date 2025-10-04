"""MLP-KAN: Kolmogorov–Arnold Network layers for PyTorch."""

from .kan_layer import KANLayer, KANInterpoLayer, smooth_penalty
from .interpolation import linear_interpolation, heaviside_theta

__all__ = [
    "KANLayer",
    "KANInterpoLayer",
    "smooth_penalty",
    "linear_interpolation",
    "heaviside_theta",
]

__version__ = "0.1.0"
