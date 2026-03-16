"""Euclidean manifold R^d for continuous variables."""

import torch


class EuclideanManifold:
    """Operations on Euclidean space R^d_c for continuous variables."""

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear geodesic interpolation: gamma(t) = (1-t)*x0 + t*x1."""
        t = t.unsqueeze(-1) if t.dim() < x0.dim() else t
        return (1 - t) * x0 + t * x1

    def velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor,
                 xt: torch.Tensor) -> torch.Tensor:
        """Target velocity field: u_t(x_t | x_1) = x_1 - x_0."""
        return x1 - x0

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Euclidean distance ||x - y||_2."""
        return torch.norm(x - y, dim=-1)

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             xt: torch.Tensor) -> torch.Tensor:
        """Euclidean MSE loss: ||u_pred - u_target||_2^2."""
        return torch.mean(torch.sum((pred - target) ** 2, dim=-1))

    def sample_prior(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample from standard Gaussian N(0, I)."""
        return torch.randn(shape, device=device)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Identity projection (no constraints on R^d)."""
        return x
