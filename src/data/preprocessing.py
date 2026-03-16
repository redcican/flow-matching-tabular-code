"""Preprocessing for mixed-type tabular data.

Handles type detection, encoding, and conversion between raw tabular format
and the product manifold representation.
"""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field

from ..manifolds.product import ColumnInfo


@dataclass
class ColumnMapping:
    """Stores metadata for reversible preprocessing."""
    continuous_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    ordinal_cols: list[str] = field(default_factory=list)

    # Continuous normalization params
    cont_means: np.ndarray | None = None
    cont_stds: np.ndarray | None = None

    # Categorical encoding: column -> {category: index}
    cat_encodings: dict[str, dict] = field(default_factory=dict)
    cat_n_categories: list[int] = field(default_factory=list)

    # Ordinal encoding: column -> sorted unique values
    ord_encodings: dict[str, list] = field(default_factory=dict)


class TabularPreprocessor:
    """Preprocess raw tabular data into product manifold representation."""

    def __init__(self, continuous_cols: list[str] | None = None,
                 categorical_cols: list[str] | None = None,
                 ordinal_cols: list[str] | None = None):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.ordinal_cols = ordinal_cols
        self.mapping = None

    def _detect_types(self, df: pd.DataFrame) -> None:
        """Auto-detect column types if not specified."""
        if self.continuous_cols is None:
            self.continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.categorical_cols is None:
            self.categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        if self.ordinal_cols is None:
            self.ordinal_cols = []

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit preprocessing parameters from training data."""
        self._detect_types(df)

        mapping = ColumnMapping(
            continuous_cols=self.continuous_cols,
            categorical_cols=self.categorical_cols,
            ordinal_cols=self.ordinal_cols,
        )

        # Continuous: compute mean and std for z-score normalization
        if self.continuous_cols:
            cont_data = df[self.continuous_cols].values.astype(np.float64)
            mapping.cont_means = np.nanmean(cont_data, axis=0)
            mapping.cont_stds = np.nanstd(cont_data, axis=0)
            mapping.cont_stds[mapping.cont_stds < 1e-8] = 1.0

        # Categorical: build encoding maps
        for col in self.categorical_cols:
            unique_vals = sorted(df[col].dropna().unique())
            mapping.cat_encodings[col] = {v: i for i, v in enumerate(unique_vals)}
            mapping.cat_n_categories.append(len(unique_vals))

        # Ordinal: map to ordered indices
        for col in self.ordinal_cols:
            unique_vals = sorted(df[col].dropna().unique())
            mapping.ord_encodings[col] = unique_vals

        self.mapping = mapping
        return self

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Transform DataFrame into product manifold tensor."""
        parts = []

        # Continuous: z-score normalize
        if self.mapping.continuous_cols:
            cont = df[self.mapping.continuous_cols].values.astype(np.float64)
            cont = (cont - self.mapping.cont_means) / self.mapping.cont_stds
            cont = np.nan_to_num(cont, 0.0)
            parts.append(torch.tensor(cont, dtype=torch.float32))

        # Categorical: one-hot on simplex (with smoothing to stay in interior)
        for col in self.mapping.categorical_cols:
            enc = self.mapping.cat_encodings[col]
            n_cat = len(enc)
            indices = df[col].map(enc).fillna(0).values.astype(int)
            # Smooth one-hot: (1-eps)*one_hot + eps/n_cat
            eps = 0.01
            onehot = np.eye(n_cat)[indices] * (1 - eps) + eps / n_cat
            parts.append(torch.tensor(onehot, dtype=torch.float32))

        # Ordinal: map to angles on [0, 2*pi)
        for col in self.mapping.ordinal_cols:
            vals = self.mapping.ord_encodings[col]
            n_levels = len(vals)
            val_to_idx = {v: i for i, v in enumerate(vals)}
            indices = df[col].map(val_to_idx).fillna(0).values.astype(np.float64)
            # Map to [0, 2*pi) with equal spacing
            angles = indices / n_levels * 2 * np.pi
            parts.append(torch.tensor(angles, dtype=torch.float32).unsqueeze(-1))

        return torch.cat(parts, dim=-1)

    def inverse_transform(self, data: torch.Tensor) -> pd.DataFrame:
        """Convert product manifold tensor back to DataFrame."""
        data_np = data.detach().cpu().numpy()
        result = {}
        idx = 0

        # Continuous: inverse z-score
        for i, col in enumerate(self.mapping.continuous_cols):
            val = data_np[:, idx] * self.mapping.cont_stds[i] + self.mapping.cont_means[i]
            result[col] = val
            idx += 1

        # Categorical: argmax of simplex probabilities
        for col in self.mapping.categorical_cols:
            enc = self.mapping.cat_encodings[col]
            n_cat = len(enc)
            probs = data_np[:, idx:idx + n_cat]
            indices = np.argmax(probs, axis=1)
            inv_enc = {i: v for v, i in enc.items()}
            result[col] = [inv_enc.get(i, inv_enc[0]) for i in indices]
            idx += n_cat

        # Ordinal: map angle back to category
        for col in self.mapping.ordinal_cols:
            vals = self.mapping.ord_encodings[col]
            n_levels = len(vals)
            angle = data_np[:, idx]
            indices = np.round(angle / (2 * np.pi) * n_levels) % n_levels
            result[col] = [vals[int(i)] for i in indices]
            idx += 1

        # Reconstruct with original column order
        all_cols = (self.mapping.continuous_cols +
                    self.mapping.categorical_cols +
                    self.mapping.ordinal_cols)
        return pd.DataFrame(result)[all_cols]

    def get_column_info(self) -> ColumnInfo:
        """Return ColumnInfo for constructing the ProductManifold."""
        return ColumnInfo(
            continuous_dims=len(self.mapping.continuous_cols),
            categorical_dims=self.mapping.cat_n_categories,
            ordinal_dims=len(self.mapping.ordinal_cols),
        )
