"""ODE solver for generation via product manifold flow matching.

Integrates the learned velocity field from t=0 (noise) to t=1 (data)
using adaptive ODE integration (dopri5) via torchdiffeq.
"""

import torch
from torchdiffeq import odeint

from .flow_matching import ProductManifoldFlowMatching


class VelocityFieldODE(torch.nn.Module):
    """Wraps the velocity field for torchdiffeq interface.

    torchdiffeq expects f(t, x) -> dx/dt
    """

    def __init__(self, model: ProductManifoldFlowMatching):
        super().__init__()
        self.model = model

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        t_batch = t.expand(batch_size)
        return self.model.predict_velocity(x, t_batch)


class ManifoldODESolver:
    """Generate synthetic data via ODE integration on the product manifold.

    Solves dx/dt = u_theta(x, t) from t=0 to t=1 where:
    - x(0) ~ prior distribution (Gaussian x Dirichlet x Uniform)
    - x(1) ~ learned data distribution
    """

    def __init__(self, model: ProductManifoldFlowMatching, atol: float = 1e-5,
                 rtol: float = 1e-5, method: str = "dopri5"):
        self.model = model
        self.ode_func = VelocityFieldODE(model)
        self.atol = atol
        self.rtol = rtol
        self.method = method

    @torch.no_grad()
    def generate(self, n_samples: int, device: torch.device,
                 n_steps: int | None = None) -> torch.Tensor:
        """Generate synthetic samples via ODE integration.

        Args:
            n_samples: number of samples to generate
            device: torch device
            n_steps: fixed number of steps (None for adaptive)

        Returns:
            x1: generated samples on the product manifold [n_samples, D]
        """
        self.model.eval()

        # Sample initial noise from product manifold prior
        x0 = self.model.manifold.sample_prior(n_samples, device)

        # Time span: t=0 (noise) to t=1 (data)
        t_span = torch.tensor([0.0, 1.0], device=device)

        if n_steps is not None:
            t_span = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

        # ODE integration
        trajectory = odeint(
            self.ode_func,
            x0,
            t_span,
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
        )

        # Take final time point
        x1 = trajectory[-1]

        # Project onto manifold constraints
        x1 = self._project(x1)

        return x1

    @torch.no_grad()
    def generate_trajectory(self, n_samples: int, device: torch.device,
                            n_steps: int = 50) -> torch.Tensor:
        """Generate full trajectory from noise to data.

        Returns:
            trajectory: [n_steps+1, n_samples, D]
        """
        self.model.eval()

        x0 = self.model.manifold.sample_prior(n_samples, device)
        t_span = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

        trajectory = odeint(
            self.ode_func,
            x0,
            t_span,
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
        )

        return trajectory

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Project generated samples onto manifold constraints.

        Ensures:
        - Categorical variables are valid probability distributions (on simplex)
        - Ordinal variables are valid angles (on circle)
        """
        manifold = self.model.manifold
        x_c, x_d, x_o = manifold.split(x)

        # Project categorical to simplex
        projected_d = [manifold.simplex.project(d) for d in x_d]

        # Project ordinal to [0, 2*pi)
        projected_o = manifold.circle.project(x_o) if x_o is not None else None

        return manifold.combine(x_c, projected_d, projected_o)
