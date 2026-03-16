"""Coordinate-wise velocity field networks for product manifold flow matching.

Each manifold type gets a dedicated MLP with manifold-specific output constraints:
- Euclidean: unconstrained output
- Categorical: output projected to simplex tangent space (sums to zero)
- Ordinal: scalar angular velocity with periodic input encoding
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for time conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ManifoldMLP(nn.Module):
    """Base MLP for velocity field prediction.

    5-layer MLP with 256 hidden units as specified in the paper.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256,
                 n_layers: int = 5, time_dim: int = 64, context_dim: int = 0):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        total_input = input_dim + time_dim + context_dim
        layers = []
        in_dim = total_input
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        t_emb = self.time_embed(t)
        inputs = [x, t_emb]
        if context is not None:
            inputs.append(context)
        return self.net(torch.cat(inputs, dim=-1))


class ContinuousVelocityNet(nn.Module):
    """Velocity network for continuous variables on R^d_c.

    Unconstrained output (Euclidean tangent space = R^d_c).
    """

    def __init__(self, dim: int, hidden_dim: int = 256, n_layers: int = 5,
                 context_dim: int = 0):
        super().__init__()
        self.mlp = ManifoldMLP(dim, dim, hidden_dim, n_layers,
                               context_dim=context_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x, t, context)


class CategoricalVelocityNet(nn.Module):
    """Velocity network for a single categorical variable on Delta^{n-1}.

    Output is projected to the tangent space of the simplex:
    T_p(Delta) = {v in R^n : sum_i v_i = 0}.
    """

    def __init__(self, n_categories: int, hidden_dim: int = 256,
                 n_layers: int = 5, context_dim: int = 0):
        super().__init__()
        self.n_categories = n_categories
        self.mlp = ManifoldMLP(n_categories, n_categories, hidden_dim,
                               n_layers, context_dim=context_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        v = self.mlp(x, t, context)
        # Project to tangent space: v - mean(v)
        v = v - v.mean(dim=-1, keepdim=True)
        return v


class OrdinalVelocityNet(nn.Module):
    """Velocity network for ordinal variables on S^1.

    Uses periodic (sin/cos) encoding of input angles.
    Output is a scalar angular velocity per ordinal variable.
    """

    def __init__(self, n_ordinal: int, hidden_dim: int = 256,
                 n_layers: int = 5, context_dim: int = 0):
        super().__init__()
        self.n_ordinal = n_ordinal
        # Periodic encoding doubles input dimension
        input_dim = n_ordinal * 2
        self.mlp = ManifoldMLP(input_dim, n_ordinal, hidden_dim, n_layers,
                               context_dim=context_dim)

    def _periodic_encode(self, theta: torch.Tensor) -> torch.Tensor:
        """Encode angles as (sin(theta), cos(theta)) pairs."""
        return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)

    def forward(self, theta: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        x = self._periodic_encode(theta)
        return self.mlp(x, t, context)


class ProductVelocityNet(nn.Module):
    """Combined velocity field network on the product manifold.

    Manages separate networks for continuous, categorical, and ordinal components.
    Accepts Einstein midpoint aggregation context for inter-type conditioning.
    """

    def __init__(self, cont_dim: int, cat_dims: list[int], ord_dim: int,
                 hidden_dim: int = 256, n_layers: int = 5, context_dim: int = 0):
        super().__init__()
        self.cont_dim = cont_dim
        self.cat_dims = cat_dims
        self.ord_dim = ord_dim

        if cont_dim > 0:
            self.cont_net = ContinuousVelocityNet(
                cont_dim, hidden_dim, n_layers, context_dim
            )

        self.cat_nets = nn.ModuleList([
            CategoricalVelocityNet(n_cat, hidden_dim, n_layers, context_dim)
            for n_cat in cat_dims
        ])

        if ord_dim > 0:
            self.ord_net = OrdinalVelocityNet(
                ord_dim, hidden_dim, n_layers, context_dim
            )

    def forward(self, x_c: torch.Tensor | None, x_d: list[torch.Tensor],
                x_o: torch.Tensor | None, t: torch.Tensor,
                context: torch.Tensor | None = None) -> tuple:
        """Predict velocity fields for each manifold component.

        Returns:
            u_c: continuous velocity [B, d_c]
            u_d: list of categorical velocities [[B, n_1], ...]
            u_o: ordinal velocity [B, m]
        """
        u_c = self.cont_net(x_c, t, context) if x_c is not None else None

        u_d = [
            net(x_d_j, t, context)
            for net, x_d_j in zip(self.cat_nets, x_d)
        ]

        u_o = self.ord_net(x_o, t, context) if x_o is not None else None

        return u_c, u_d, u_o
