"""Core KAN layers and utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interpolation import linear_interpolation


def phi(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor, n_sin: int) -> torch.Tensor:
    """Apply the sinusoidal feature mapping followed by a learnable MLP."""
    num_sin = int(n_sin)
    if num_sin < 0:
        raise ValueError("n_sin must be non-negative")

    if num_sin:
        omega = 2 ** torch.arange(0, num_sin, device=x.device, dtype=x.dtype)
        omega = omega.reshape(-1, 1)
        omega_x = F.linear(x, omega, bias=None)
        x = torch.cat([x, torch.sin(omega_x), torch.cos(omega_x)], dim=-1)

    x = F.linear(x, w1, bias=b1)
    x = F.silu(x)
    x = F.linear(x, w2, bias=b2)
    return x


class KANLayer(nn.Module):
    """Kolmogorov–Arnold Network layer with learnable sinusoidal features."""

    def __init__(self, dim_in: int, dim_out: int, fcn_hidden: int = 32, fcn_n_sin: int = 3) -> None:
        super().__init__()
        feature_dim = 1 + fcn_n_sin * 2
        self.W1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden, feature_dim))
        self.W2 = nn.Parameter(torch.randn(dim_in, dim_out, 1, fcn_hidden))
        self.B1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden))
        self.B2 = nn.Parameter(torch.randn(dim_in, dim_out, 1))

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fcn_hidden = fcn_hidden
        self.fcn_n_sin = int(fcn_n_sin)

        self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.zeros_(self.B1)
        nn.init.zeros_(self.B2)

    def map(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorised evaluation of all phi functions for a single input sample."""

        f_map = torch.vmap(
            torch.vmap(phi, (None, 0, 0, 0, 0, None), 0),
            (0, 0, 0, 0, 0, None),
            0,
        )
        return f_map(
            x.unsqueeze(-1),
            self.W1,
            self.W2,
            self.B1,
            self.B2,
            self.fcn_n_sin,
        ).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        batch, dim_in = x.shape
        if dim_in != self.dim_in:
            raise ValueError(f"Expected input with dim_in={self.dim_in}, got {dim_in}")

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x)
        return phis.sum(dim=1)

    def take_function(self, i: int, j: int):
        """Return the phi function parameterised by indices ``(i, j)``."""

        def activation(x: torch.Tensor) -> torch.Tensor:
            return phi(x, self.W1[i, j], self.W2[i, j], self.B1[i, j], self.B2[i, j], self.fcn_n_sin)

        return activation


class KANInterpoLayer(nn.Module):
    """Kolmogorov–Arnold Network layer parameterised via interpolation."""

    def __init__(self, dim_in: int, dim_out: int, num_x: int = 64, x_min: float = -2.0, x_max: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("X", torch.linspace(x_min, x_max, num_x))
        self.Y = nn.Parameter(torch.randn(dim_in, dim_out, num_x))

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.Y)

    def map(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorised interpolation for a single input sample."""

        f_map = torch.vmap(
            torch.vmap(linear_interpolation, (None, None, 0), 0),
            (0, None, 0),
            0,
        )
        return f_map(x.unsqueeze(-1), self.X, self.Y).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        batch, dim_in = x.shape
        if dim_in != self.dim_in:
            raise ValueError(f"Expected input with dim_in={self.dim_in}, got {dim_in}")

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x)
        return phis.sum(dim=1)

    def take_function(self, i: int, j: int):
        """Return the interpolated function parameterised by indices ``(i, j)``."""

        def activation(x: torch.Tensor) -> torch.Tensor:
            return linear_interpolation(x, self.X, self.Y[i, j])

        return activation


def smooth_penalty(model: nn.Module) -> torch.Tensor:
    """Compute a smoothness penalty for interpolation-based KAN layers."""

    def _penalty_from_layer(layer: KANInterpoLayer) -> torch.Tensor:
        dx = layer.X[1] - layer.X[0]
        grad = layer.Y[:, :, 1:] - layer.Y[:, :, :-1]
        return torch.norm(grad, 2) / dx

    if isinstance(model, KANInterpoLayer):
        return _penalty_from_layer(model)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    penalty = torch.zeros((), device=device)
    for layer in model:
        if isinstance(layer, KANInterpoLayer):
            penalty = penalty + _penalty_from_layer(layer)
    return penalty
