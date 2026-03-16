# Product Manifold Flow Matching for Mixed-Type Tabular Data Generation

PyTorch implementation of geometric flow matching on product manifolds for synthetic tabular data generation. Instead of forcing all variables into continuous spaces, this method operates directly on the natural geometry of each data type:

- **Continuous variables** on Euclidean space $\mathbb{R}^{d_c}$
- **Categorical variables** on probability simplexes $\Delta^{n-1}$ with Fisher-Rao geometry
- **Ordinal variables** on unit circles $S^1$

## Method Overview

The framework learns coordinate-wise velocity fields on the product manifold $\mathcal{M} = \mathbb{R}^{d_c} \times \prod_j \Delta^{n_j-1} \times \prod_l S^1$ and generates synthetic data via ODE integration from noise to data.

**Key components:**

| Component | Description |
|-----------|-------------|
| Fisher-Rao geodesic flow | SLERP on the positive orthant of the unit sphere via $\phi(p) = \sqrt{p}$ |
| Coordinate-wise decomposition | Independent velocity networks per manifold type, trained in parallel |
| Einstein midpoint aggregation | Riemannian center-of-mass for inter-type dependency modeling |
| Product manifold distance | Geometrically consistent DCR metric for privacy evaluation |

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch >= 2.0, torchdiffeq, scikit-learn, pandas, scipy

## Usage

### Train

```bash
# Auto-detect column types
python train.py --data path/to/data.csv

# Specify column types explicitly
python train.py --data data.csv \
    --continuous age income \
    --categorical gender disease \
    --ordinal education \
    --target label
```

### Generate

```bash
python generate.py --checkpoint checkpoints/best_model.pt --n_samples 10000
```

### Evaluate

```bash
python evaluate.py --real data.csv --synthetic synthetic_data.csv \
    --checkpoint checkpoints/best_model.pt --target label
```

## Project Structure

```
├── configs/default.yaml           # Hyperparameters
├── train.py                       # Training entry point
├── generate.py                    # ODE-based synthetic data generation
├── evaluate.py                    # W1 / JS / MLE / DCR / MIA evaluation
└── src/
    ├── manifolds/
    │   ├── euclidean.py           # R^d: linear interpolation, L2 distance
    │   ├── simplex.py             # Delta^{n-1}: Fisher-Rao geodesics, SLERP, FR velocity field
    │   ├── circle.py              # S^1: circular geodesics, wrapped arithmetic
    │   └── product.py             # Product manifold: coordinate-wise operations
    ├── models/
    │   ├── velocity_nets.py       # Per-manifold MLPs with sinusoidal time embedding
    │   ├── einstein.py            # Einstein midpoint aggregation
    │   └── projections.py         # Cross-manifold projection operators
    ├── data/
    │   ├── dataset.py             # MixedTypeDataset (PyTorch Dataset)
    │   ├── preprocessing.py       # Type detection, z-score, simplex smoothing, ordinal encoding
    │   └── dependency.py          # Dependency graph via Pearson / Cramer's V / NMI
    ├── training/
    │   ├── flow_matching.py       # Algorithm 1: coordinate-wise CFM with Einstein context
    │   └── ode_solver.py          # dopri5 ODE integration with manifold projection
    └── evaluation/
        ├── fidelity.py            # Wasserstein-1, Jensen-Shannon divergence
        ├── utility.py             # ML Efficacy (TSTR with LR / RF / XGBoost)
        └── privacy.py             # DCR (product manifold metric), MIA (shadow models)
```

## Configuration

Key hyperparameters in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.hidden_dim` | 256 | Hidden units per velocity network layer |
| `model.n_layers` | 5 | MLP depth for each manifold component |
| `model.alpha` | 0.3 | Einstein midpoint aggregation strength |
| `training.lr` | 1e-3 | Adam learning rate |
| `training.batch_size` | 256 | Training batch size |
| `training.patience` | 10 | Early stopping patience |
| `generation.method` | dopri5 | ODE solver (dopri5 / euler / rk4) |
| `dependency.threshold` | 0.1 | Edge sparsity threshold for dependency graph |

## Citation

```bibtex
@article{chen2025product,
  title={Product Manifold Flow Matching for Mixed-Type Tabular Data Generation},
  author={Chen, Shikun and Xiong, Songquan},
  journal={Neural Networks},
  year={2025}
}
```

## License

MIT
