"""Probability simplex Delta^{n-1} with Fisher-Rao geometry for categorical variables.

The Fisher-Rao metric on the simplex is handled via the diffeomorphism
phi: Delta^{n-1} -> S^{n-1}_+, phi(p) = sqrt(p), which maps the simplex
to the positive orthant of the unit sphere. Under this map, the Fisher-Rao
distance becomes the great circle distance on the sphere.
"""

import torch

EPS = 1e-8


class SimplexManifold:
    """Operations on the probability simplex with Fisher-Rao geometry."""

    def to_sphere(self, p: torch.Tensor) -> torch.Tensor:
        """Map simplex to positive orthant of sphere: phi(p) = sqrt(p).

        Satisfies |phi(p)|^2 = sum_i p_i = 1 for p in Delta^{n-1}.
        """
        return torch.sqrt(torch.clamp(p, min=EPS))

    def to_simplex(self, v: torch.Tensor) -> torch.Tensor:
        """Inverse map from sphere to simplex: phi^{-1}(v) = v * v."""
        p = v * v
        return p / torch.clamp(p.sum(dim=-1, keepdim=True), min=EPS)

    def interpolate(self, p0: torch.Tensor, p1: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """Fisher-Rao geodesic interpolation on the simplex.

        Computes geodesic via spherical interpolation (SLERP) on
        the unit sphere, then maps back to the simplex.
        """
        t = t.unsqueeze(-1) if t.dim() < p0.dim() else t

        xi = self.to_sphere(p0)  # sqrt(p0)
        zeta = self.to_sphere(p1)  # sqrt(p1)

        # Angle between points on sphere
        cos_theta = torch.clamp(
            torch.sum(xi * zeta, dim=-1, keepdim=True), -1 + EPS, 1 - EPS
        )
        theta = torch.acos(cos_theta)

        # SLERP on sphere
        sin_theta = torch.sin(theta)
        near_zero = (theta.abs() < EPS)

        # Standard slerp coefficients
        coeff0 = torch.sin((1 - t) * theta) / torch.clamp(sin_theta, min=EPS)
        coeff1 = torch.sin(t * theta) / torch.clamp(sin_theta, min=EPS)

        # Linear fallback for small angles
        coeff0 = torch.where(near_zero, 1 - t, coeff0)
        coeff1 = torch.where(near_zero, t, coeff1)

        v_t = coeff0 * xi + coeff1 * zeta

        return self.to_simplex(v_t)

    def velocity(self, p0: torch.Tensor, p1: torch.Tensor,
                 t: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
        """Fisher-Rao velocity field u_t^FR(p | p_1).

        From the paper Eq. 6:
        u_t^FR(p|p1) = (2*theta/sin(theta)) * sqrt(p) * [-cos((1-t)*theta)*xi + cos(t*theta)*zeta]
        for theta > eps, and 2*sqrt(p) * (zeta - xi) for theta <= eps.
        """
        t_expanded = t.unsqueeze(-1) if t.dim() < p0.dim() else t

        xi = self.to_sphere(p0)
        zeta = self.to_sphere(p1)
        sqrt_pt = self.to_sphere(pt)

        cos_theta = torch.clamp(
            torch.sum(xi * zeta, dim=-1, keepdim=True), -1 + EPS, 1 - EPS
        )
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        near_zero = (theta.abs() < EPS)

        # Standard case: (2*theta/sin(theta)) * sqrt(p) * [-cos((1-t)*theta)*xi + cos(t*theta)*zeta]
        scale = 2.0 * theta / torch.clamp(sin_theta, min=EPS)
        direction = (
            -torch.cos((1 - t_expanded) * theta) * xi
            + torch.cos(t_expanded * theta) * zeta
        )
        u_standard = scale * sqrt_pt * direction

        # Limit case: 2*sqrt(p) * (zeta - xi)
        u_limit = 2.0 * sqrt_pt * (zeta - xi)

        return torch.where(near_zero, u_limit, u_standard)

    def distance(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Fisher-Rao distance: d_FR(p, q) = 2 * arccos(sum_i sqrt(p_i * q_i))."""
        inner = torch.sum(
            torch.sqrt(torch.clamp(p, min=EPS) * torch.clamp(q, min=EPS)),
            dim=-1,
        )
        inner = torch.clamp(inner, -1 + EPS, 1 - EPS)
        return 2.0 * torch.acos(inner)

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             pt: torch.Tensor) -> torch.Tensor:
        """Fisher-Rao metric loss: sum_i (u_pred_i - u_target_i)^2 / (4 * p_i).

        This is the squared norm under the Fisher information metric at point p.
        """
        diff_sq = (pred - target) ** 2
        weights = 1.0 / (4.0 * torch.clamp(pt, min=EPS))
        return torch.mean(torch.sum(diff_sq * weights, dim=-1))

    def log_map(self, p: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at base point on the simplex (via sphere)."""
        xi_base = self.to_sphere(base)
        xi_p = self.to_sphere(p)

        cos_theta = torch.clamp(
            torch.sum(xi_base * xi_p, dim=-1, keepdim=True), -1 + EPS, 1 - EPS
        )
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)

        near_zero = (theta.abs() < EPS)

        # Tangent vector on sphere: (theta/sin(theta)) * (xi_p - cos(theta)*xi_base)
        v = theta / torch.clamp(sin_theta, min=EPS) * (xi_p - cos_theta * xi_base)
        v_limit = xi_p - xi_base
        return torch.where(near_zero, v_limit, v)

    def exp_map(self, v: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Exponential map at base point on the simplex (via sphere)."""
        xi_base = self.to_sphere(base)
        norm_v = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=EPS)

        near_zero = (norm_v < EPS)

        xi_result = torch.cos(norm_v) * xi_base + torch.sin(norm_v) * (v / norm_v)
        xi_limit = xi_base + v

        xi_out = torch.where(near_zero, xi_limit, xi_result)
        return self.to_simplex(xi_out)

    def sample_prior(self, shape: tuple, n_categories: int,
                     device: torch.device) -> torch.Tensor:
        """Sample from Dirichlet(1,...,1) = uniform distribution on the simplex."""
        alpha = torch.ones(*shape, n_categories, device=device)
        return torch.distributions.Dirichlet(alpha).sample()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto the probability simplex via softmax."""
        return torch.softmax(x, dim=-1)
