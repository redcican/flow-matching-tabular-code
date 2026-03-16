"""Geometric projection operators between manifold types.

From the paper Eq. 15-18:
- Pi_{R}(Delta): p -> log(p) - mean(log(p))    (categorical -> continuous)
- Pi_{Delta}(R): x -> softmax(Wx + b)           (continuous -> categorical)
- Pi_{S1}(R): x -> 2*pi*sigmoid(x)              (continuous -> ordinal)
- Pi_{R}(S1): theta -> (sin(theta), cos(theta))  (ordinal -> continuous)
"""

import torch
import torch.nn as nn
import math

EPS = 1e-8


class ManifoldProjections(nn.Module):
    """Cross-manifold projection operators for inter-type dependencies."""

    def __init__(self, cont_dim: int, cat_dims: list[int], ord_dim: int):
        super().__init__()
        self.cont_dim = cont_dim
        self.cat_dims = cat_dims
        self.ord_dim = ord_dim

        # Learned parameters for continuous -> categorical projection
        self.cat_projectors = nn.ModuleList([
            nn.Linear(cont_dim, n_cat) if cont_dim > 0 else None
            for n_cat in cat_dims
        ])

    def categorical_to_continuous(self, p: torch.Tensor) -> torch.Tensor:
        """Pi_{R}(Delta): p -> log(p) - mean(log(p)).

        Log-ratio transformation (centered log-ratio), the information-theoretic
        dual of softmax. Standard in compositional data analysis.
        """
        log_p = torch.log(torch.clamp(p, min=EPS))
        return log_p - log_p.mean(dim=-1, keepdim=True)

    def continuous_to_categorical(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Pi_{Delta}(R): x -> softmax(Wx + b).

        Uses learned linear transformation followed by softmax normalization.
        """
        projector = self.cat_projectors[idx]
        if projector is None:
            raise ValueError("No continuous dimensions available for projection")
        return torch.softmax(projector(x), dim=-1)

    def continuous_to_ordinal(self, x: torch.Tensor) -> torch.Tensor:
        """Pi_{S1}(R): x -> 2*pi*sigmoid(x).

        Maps continuous values to angles on [0, 2*pi) via sigmoid.
        """
        return 2 * math.pi * torch.sigmoid(x)

    def ordinal_to_continuous(self, theta: torch.Tensor) -> torch.Tensor:
        """Pi_{R}(S1): theta -> (sin(theta), cos(theta)).

        Trigonometric embedding that preserves the metric structure of S^1.
        """
        return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
