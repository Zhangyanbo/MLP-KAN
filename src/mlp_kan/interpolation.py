"""Interpolation primitives used by KAN layers."""

from __future__ import annotations

import torch


def heaviside_theta(x: torch.Tensor, mu: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Differentiable Heaviside theta with compact support of width ``r``."""
    x = x - mu
    r = torch.as_tensor(r, device=x.device, dtype=x.dtype)
    return (torch.clamp(x + r, 0, r) - torch.clamp(x, 0, r)) / r


def _linear_interpolation(x: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Linear interpolation for a single scalar input."""
    mu = X
    r = X[1] - X[0]
    heaviside = torch.vmap(heaviside_theta, in_dims=(None, 0, None))
    y = heaviside(x, mu, r).reshape(-1) * Y
    return y.sum()


def linear_interpolation(x: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Vectorised linear interpolation across the support ``X`` and values ``Y``."""
    shape = x.shape
    x = x.reshape(-1)
    interp = torch.vmap(_linear_interpolation, in_dims=(-1, None, None), out_dims=-1)
    return interp(x, X, Y).reshape(shape)
