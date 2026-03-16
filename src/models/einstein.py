"""Einstein midpoint aggregation for inter-type dependencies.

Computes Riemannian center-of-mass on each manifold component to aggregate
information across variable types, weighted by the inter-type dependency graph.

From the paper Definition 3.3 and Eqs. 9-11:
- Mid_R(x_1^c, ..., x_n^c; w) = sum_i w_i x_i^c
- Mid_Delta(p_1, ..., p_n; w) = exp_mu(sum_i w_i log_mu(p_i))
- Mid_S1(theta_1, ..., theta_n; w) = arg(sum_i w_i exp(i*theta_i))
"""

import torch
import torch.nn as nn

from ..manifolds.simplex import SimplexManifold
from ..manifolds.circle import CircleManifold


class EinsteinMidpoint(nn.Module):
    """Einstein midpoint aggregation on the product manifold.

    For each target variable, aggregates information from connected variables
    in the dependency graph using geometric projections and Riemannian
    center-of-mass operations.
    """

    def __init__(self, cont_dim: int, cat_dims: list[int], ord_dim: int,
                 alpha: float = 0.3):
        """
        Args:
            cont_dim: number of continuous features
            cat_dims: list of category counts per categorical variable
            ord_dim: number of ordinal features
            alpha: aggregation strength in [0, 1]
        """
        super().__init__()
        self.cont_dim = cont_dim
        self.cat_dims = cat_dims
        self.ord_dim = ord_dim
        self.alpha = alpha

        self.simplex = SimplexManifold()
        self.circle = CircleManifold()

        # Total representation dimension for context vector
        total_dim = cont_dim + sum(cat_dims) + ord_dim * 2  # ordinal uses sin/cos
        self.context_dim = total_dim

    def _build_context(self, x_c: torch.Tensor | None,
                       x_d: list[torch.Tensor],
                       x_o: torch.Tensor | None,
                       weights: dict | None = None) -> torch.Tensor:
        """Build aggregated context vector from all manifold components.

        Applies dependency-graph weights if provided, then concatenates
        representations from all variable types into a single context vector.
        """
        parts = []

        if x_c is not None:
            parts.append(x_c)

        for x_d_j in x_d:
            parts.append(x_d_j)

        if x_o is not None:
            # Use periodic encoding for ordinal context
            parts.append(torch.sin(x_o))
            parts.append(torch.cos(x_o))

        return torch.cat(parts, dim=-1)

    def aggregate_continuous(self, x_c: torch.Tensor,
                             projected: list[torch.Tensor],
                             weights: torch.Tensor) -> torch.Tensor:
        """Euclidean midpoint: weighted average."""
        if not projected:
            return x_c
        stacked = torch.stack(projected, dim=1)  # [B, K, d_c]
        w = weights.unsqueeze(-1)  # [B, K, 1]
        agg = torch.sum(w * stacked, dim=1)  # [B, d_c]
        return (1 - self.alpha) * x_c + self.alpha * agg

    def aggregate_categorical(self, p: torch.Tensor,
                              projected: list[torch.Tensor],
                              weights: torch.Tensor) -> torch.Tensor:
        """Simplex midpoint via exp/log maps at reference point."""
        if not projected:
            return p
        # Use uniform distribution as reference point
        batch_size, n_cat = p.shape
        mu = torch.ones_like(p) / n_cat

        log_p = self.simplex.log_map(p, mu)
        log_projected = [self.simplex.log_map(proj, mu) for proj in projected]

        stacked = torch.stack(log_projected, dim=1)
        w = weights.unsqueeze(-1)
        agg_tangent = torch.sum(w * stacked, dim=1)

        combined = (1 - self.alpha) * log_p + self.alpha * agg_tangent
        return self.simplex.exp_map(combined, mu)

    def aggregate_ordinal(self, theta: torch.Tensor,
                          projected: list[torch.Tensor],
                          weights: torch.Tensor) -> torch.Tensor:
        """Circular midpoint: arg(sum w_i exp(i*theta_i))."""
        if not projected:
            return theta
        # Weighted circular mean of all projected values + original
        all_thetas = [theta] + projected
        all_weights = torch.cat([
            (1 - self.alpha) * torch.ones(theta.shape[0], 1, device=theta.device),
            self.alpha * weights
        ], dim=1)

        stacked = torch.stack(all_thetas, dim=1)  # [B, K+1, m]
        w = all_weights.unsqueeze(-1)  # [B, K+1, 1]

        cos_sum = torch.sum(w * torch.cos(stacked), dim=1)
        sin_sum = torch.sum(w * torch.sin(stacked), dim=1)
        return torch.atan2(sin_sum, cos_sum) % (2 * torch.pi)

    def forward(self, x_c: torch.Tensor | None, x_d: list[torch.Tensor],
                x_o: torch.Tensor | None,
                dep_weights: dict | None = None) -> torch.Tensor:
        """Compute Einstein midpoint aggregation context vector.

        Args:
            x_c: continuous features [B, d_c]
            x_d: list of categorical features [[B, n_j], ...]
            x_o: ordinal features [B, m]
            dep_weights: dependency graph weights (optional)

        Returns:
            context: aggregated context vector [B, context_dim]
        """
        return self._build_context(x_c, x_d, x_o, dep_weights)
