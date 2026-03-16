"""Unit circle S^1 for ordinal variables.

Ordinal variables are embedded on S^1 to preserve their cyclic/order structure.
Points are represented as angles theta in [0, 2*pi).
"""

import torch
import math

EPS = 1e-8


class CircleManifold:
    """Operations on the unit circle S^1 for ordinal variables."""

    def wrap(self, theta: torch.Tensor) -> torch.Tensor:
        """Wrap angle to [0, 2*pi)."""
        return theta % (2 * math.pi)

    def signed_diff(self, theta1: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
        """Signed angular difference in [-pi, pi]."""
        diff = (theta1 - theta0) % (2 * math.pi)
        return diff - 2 * math.pi * (diff > math.pi).float()

    def interpolate(self, theta0: torch.Tensor, theta1: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """Geodesic interpolation on S^1 via shortest arc."""
        t = t.unsqueeze(-1) if t.dim() < theta0.dim() else t
        delta = self.signed_diff(theta1, theta0)
        return self.wrap(theta0 + t * delta)

    def velocity(self, theta0: torch.Tensor, theta1: torch.Tensor,
                 t: torch.Tensor, theta_t: torch.Tensor) -> torch.Tensor:
        """Velocity field: u_t = signed angular difference (constant along geodesic)."""
        return self.signed_diff(theta1, theta0)

    def distance(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """Geodesic distance on S^1: min arc length."""
        return torch.abs(self.signed_diff(theta1, theta2))

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             theta_t: torch.Tensor) -> torch.Tensor:
        """Circular MSE loss on angular velocities."""
        diff = self.signed_diff(pred, target)
        return torch.mean(diff ** 2)

    def circular_mean(self, thetas: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
        """Weighted circular mean: arg(sum_i w_i * exp(i*theta_i)).

        This is the Riemannian center of mass on S^1.
        """
        cos_sum = torch.sum(weights * torch.cos(thetas), dim=-1)
        sin_sum = torch.sum(weights * torch.sin(thetas), dim=-1)
        return torch.atan2(sin_sum, cos_sum) % (2 * math.pi)

    def sample_prior(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample uniformly on [0, 2*pi)."""
        return torch.rand(shape, device=device) * 2 * math.pi

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project to [0, 2*pi)."""
        return self.wrap(x)
