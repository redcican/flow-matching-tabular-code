"""Product manifold M = R^{d_c} x prod_j Delta^{n_j-1} x prod_l S^1.

Combines Euclidean, simplex, and circle manifolds for mixed-type tabular data.
"""

import torch
from dataclasses import dataclass

from .euclidean import EuclideanManifold
from .simplex import SimplexManifold
from .circle import CircleManifold


@dataclass
class ColumnInfo:
    """Metadata about columns in the tabular dataset."""
    continuous_dims: int  # d_c: number of continuous features
    categorical_dims: list[int]  # [n_1, n_2, ...]: categories per categorical variable
    ordinal_dims: int  # m: number of ordinal features


class ProductManifold:
    """Product manifold for mixed-type tabular data.

    Manages coordinate-wise operations across the three manifold types.
    """

    def __init__(self, col_info: ColumnInfo):
        self.col_info = col_info
        self.euclidean = EuclideanManifold()
        self.simplex = SimplexManifold()
        self.circle = CircleManifold()

        # Compute total dimension and slice indices
        self.cont_dim = col_info.continuous_dims
        self.cat_dims = col_info.categorical_dims
        self.ord_dim = col_info.ordinal_dims
        self.total_cat_dim = sum(self.cat_dims)
        self.total_dim = self.cont_dim + self.total_cat_dim + self.ord_dim

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Split product manifold point into components.

        Returns:
            x_c: continuous part [B, d_c]
            x_d: list of categorical parts [[B, n_1], [B, n_2], ...]
            x_o: ordinal part [B, m]
        """
        idx = 0
        x_c = x[:, idx:idx + self.cont_dim] if self.cont_dim > 0 else None
        idx += self.cont_dim

        x_d = []
        for n_cat in self.cat_dims:
            x_d.append(x[:, idx:idx + n_cat])
            idx += n_cat

        x_o = x[:, idx:idx + self.ord_dim] if self.ord_dim > 0 else None

        return x_c, x_d, x_o

    def combine(self, x_c: torch.Tensor | None, x_d: list[torch.Tensor],
                x_o: torch.Tensor | None) -> torch.Tensor:
        """Combine manifold components into a single tensor."""
        parts = []
        if x_c is not None:
            parts.append(x_c)
        parts.extend(x_d)
        if x_o is not None:
            parts.append(x_o)
        return torch.cat(parts, dim=-1)

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """Coordinate-wise geodesic interpolation on the product manifold."""
        x0_c, x0_d, x0_o = self.split(x0)
        x1_c, x1_d, x1_o = self.split(x1)

        # Euclidean interpolation
        xt_c = self.euclidean.interpolate(x0_c, x1_c, t) if x0_c is not None else None

        # Fisher-Rao interpolation for each categorical variable
        xt_d = [
            self.simplex.interpolate(x0_d_j, x1_d_j, t)
            for x0_d_j, x1_d_j in zip(x0_d, x1_d)
        ]

        # Circular interpolation for ordinal variables
        xt_o = self.circle.interpolate(x0_o, x1_o, t) if x0_o is not None else None

        return self.combine(xt_c, xt_d, xt_o)

    def velocity(self, x0: torch.Tensor, x1: torch.Tensor,
                 t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """Coordinate-wise target velocity field on the product manifold."""
        x0_c, x0_d, x0_o = self.split(x0)
        x1_c, x1_d, x1_o = self.split(x1)
        xt_c, xt_d, xt_o = self.split(xt)

        u_c = self.euclidean.velocity(x0_c, x1_c, t, xt_c) if x0_c is not None else None

        u_d = [
            self.simplex.velocity(x0_d_j, x1_d_j, t, xt_d_j)
            for x0_d_j, x1_d_j, xt_d_j in zip(x0_d, x1_d, xt_d)
        ]

        u_o = self.circle.velocity(x0_o, x1_o, t, xt_o) if x0_o is not None else None

        return self.combine(u_c, u_d, u_o)

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             xt: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Coordinate-wise loss: L = L_c + L_d + L_o."""
        pred_c, pred_d, pred_o = self.split(pred)
        target_c, target_d, target_o = self.split(target)
        xt_c, xt_d, xt_o = self.split(xt)

        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        if pred_c is not None:
            losses["continuous"] = self.euclidean.loss(pred_c, target_c, xt_c)
            total = total + losses["continuous"]

        if pred_d:
            cat_loss = torch.tensor(0.0, device=pred.device)
            for pred_j, target_j, xt_j in zip(pred_d, target_d, xt_d):
                cat_loss = cat_loss + self.simplex.loss(pred_j, target_j, xt_j)
            losses["categorical"] = cat_loss / max(len(pred_d), 1)
            total = total + losses["categorical"]

        if pred_o is not None:
            losses["ordinal"] = self.circle.loss(pred_o, target_o, xt_o)
            total = total + losses["ordinal"]

        losses["total"] = total
        return total, losses

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Product manifold distance (Eq. 25 in the paper).

        d_M(x,y) = sqrt(||x_c - y_c||^2 + sum_j d_FR(x_d_j, y_d_j)^2 + sum_l d_S1(x_o_l, y_o_l)^2)
        """
        x_c, x_d, x_o = self.split(x)
        y_c, y_d, y_o = self.split(y)

        dist_sq = torch.zeros(x.shape[0], device=x.device)

        if x_c is not None:
            dist_sq = dist_sq + self.euclidean.distance(x_c, y_c) ** 2

        for x_d_j, y_d_j in zip(x_d, y_d):
            dist_sq = dist_sq + self.simplex.distance(x_d_j, y_d_j) ** 2

        if x_o is not None:
            # Sum over ordinal dimensions
            for l in range(self.ord_dim):
                dist_sq = dist_sq + self.circle.distance(
                    x_o[:, l], y_o[:, l]
                ) ** 2

        return torch.sqrt(dist_sq)

    def sample_prior(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise from the prior on the product manifold."""
        parts = []

        if self.cont_dim > 0:
            parts.append(self.euclidean.sample_prior((batch_size, self.cont_dim), device))

        for n_cat in self.cat_dims:
            parts.append(self.simplex.sample_prior((batch_size,), n_cat, device))

        if self.ord_dim > 0:
            parts.append(self.circle.sample_prior((batch_size, self.ord_dim), device))

        return torch.cat(parts, dim=-1)
