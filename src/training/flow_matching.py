"""Product manifold flow matching trainer.

Implements Algorithm 1 from the paper: coordinate-wise flow matching
with Fisher-Rao geodesics, circular geodesics, and Einstein midpoint
aggregation for inter-type dependencies.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..manifolds.product import ProductManifold, ColumnInfo
from ..models.velocity_nets import ProductVelocityNet
from ..models.einstein import EinsteinMidpoint


class ProductManifoldFlowMatching(nn.Module):
    """Geometric flow matching on product manifolds for mixed-type tabular data.

    Training:
        1. Sample pairs (x_0, x_1) where x_0 ~ prior, x_1 ~ data
        2. Sample t ~ U[0, 1]
        3. Compute coordinate-wise geodesic interpolation x_t
        4. Compute target velocity fields (Euclidean, Fisher-Rao, circular)
        5. Compute Einstein midpoint context from x_t
        6. Predict velocity fields conditioned on context
        7. Minimize coordinate-wise loss L = L_c + L_d + L_o

    Generation:
        ODE integration from t=0 (noise) to t=1 (data) using learned velocity fields.
    """

    def __init__(self, col_info: ColumnInfo, hidden_dim: int = 256,
                 n_layers: int = 5, alpha: float = 0.3):
        super().__init__()
        self.manifold = ProductManifold(col_info)

        # Einstein midpoint for inter-type dependency context
        self.einstein = EinsteinMidpoint(
            col_info.continuous_dims,
            col_info.categorical_dims,
            col_info.ordinal_dims,
            alpha=alpha,
        )

        # Coordinate-wise velocity networks
        self.velocity_net = ProductVelocityNet(
            cont_dim=col_info.continuous_dims,
            cat_dims=col_info.categorical_dims,
            ord_dim=col_info.ordinal_dims,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            context_dim=self.einstein.context_dim,
        )

    def compute_loss(self, x1: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute flow matching loss for a batch of data points.

        Args:
            x1: data samples on the product manifold [B, D]

        Returns:
            loss: total coordinate-wise loss
            loss_dict: breakdown by manifold component
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample noise from prior
        x0 = self.manifold.sample_prior(batch_size, device)

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Coordinate-wise geodesic interpolation
        xt = self.manifold.interpolate(x0, x1, t)

        # Target velocity field
        target = self.manifold.velocity(x0, x1, t, xt)

        # Split into manifold components
        xt_c, xt_d, xt_o = self.manifold.split(xt)

        # Einstein midpoint aggregation for context
        context = self.einstein(xt_c, xt_d, xt_o)

        # Predict velocity fields
        pred_c, pred_d, pred_o = self.velocity_net(xt_c, xt_d, xt_o, t, context)

        # Combine predictions
        pred = self.manifold.combine(pred_c, pred_d, pred_o)

        # Coordinate-wise loss
        return self.manifold.loss(pred, target, xt)

    def predict_velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field at (x, t) for ODE integration.

        This is the function f(x, t) in dx/dt = f(x, t).
        """
        x_c, x_d, x_o = self.manifold.split(x)

        context = self.einstein(x_c, x_d, x_o)

        pred_c, pred_d, pred_o = self.velocity_net(x_c, x_d, x_o, t, context)

        return self.manifold.combine(pred_c, pred_d, pred_o)


class Trainer:
    """Training loop for product manifold flow matching."""

    def __init__(self, model: ProductManifoldFlowMatching, lr: float = 1e-3,
                 device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total": 0.0, "n_batches": 0}

        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            loss, loss_dict = self.model.compute_loss(batch)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            epoch_losses["total"] += loss.item()
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            epoch_losses["n_batches"] += 1

        # Average over batches
        n = epoch_losses.pop("n_batches")
        return {k: v / n for k, v in epoch_losses.items()}

    def train(self, dataloader: DataLoader, n_epochs: int = 100,
              val_dataloader: DataLoader | None = None,
              patience: int = 10, save_path: str | None = None) -> list[dict]:
        """Full training loop with early stopping.

        Args:
            dataloader: training data
            n_epochs: maximum epochs
            val_dataloader: validation data for early stopping
            patience: early stopping patience
            save_path: path to save best model

        Returns:
            history: list of loss dicts per epoch
        """
        history = []
        best_val_loss = float("inf")
        patience_counter = 0

        pbar = tqdm(range(n_epochs), desc="Training")
        for epoch in pbar:
            train_losses = self.train_epoch(dataloader)
            history.append(train_losses)

            pbar.set_postfix(loss=f"{train_losses['total']:.4f}")

            # Validation and early stopping
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        if save_path:
                            self.model.load_state_dict(torch.load(save_path,
                                                                  weights_only=True))
                        break
            elif save_path and epoch == n_epochs - 1:
                torch.save(self.model.state_dict(), save_path)

        return history

    def _validate(self, dataloader: DataLoader) -> float:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                loss, _ = self.model.compute_loss(batch)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)
