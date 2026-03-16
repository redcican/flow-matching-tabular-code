"""Inter-type dependency graph construction.

Computes pairwise statistical dependencies between variables using:
- Pearson correlation for continuous-continuous pairs
- Cramer's V for categorical-categorical pairs
- Normalized mutual information for mixed pairs

From the paper Definition 3.1 and Section 3.4.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import chi2_contingency


def cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cramer's V statistic for two categorical arrays."""
    confusion = pd.crosstab(x, y)
    if confusion.size <= 1:
        return 0.0
    chi2 = chi2_contingency(confusion)[0]
    n = len(x)
    r, k = confusion.shape
    denom = n * (min(r, k) - 1)
    if denom == 0:
        return 0.0
    return np.sqrt(chi2 / denom)


def compute_nmi(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Compute normalized mutual information for a mixed pair.

    Continuous variables are discretized into quantile bins before computing MI.
    """
    def _discretize(arr):
        if arr.dtype.kind == "f":
            try:
                return pd.qcut(arr, q=n_bins, labels=False, duplicates="drop")
            except ValueError:
                return pd.cut(arr, bins=n_bins, labels=False)
        return arr

    x_disc = _discretize(x)
    y_disc = _discretize(y)

    mask = ~(pd.isna(x_disc) | pd.isna(y_disc))
    if mask.sum() < 10:
        return 0.0

    return normalized_mutual_info_score(x_disc[mask], y_disc[mask])


class DependencyGraph:
    """Inter-type dependency graph G = (V, E) with weighted edges.

    Edge weights quantify statistical dependency strength between variables.
    Weights are normalized to [0, 1] and thresholded at tau for sparsity.
    """

    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: tau for sparsity (edges below this weight are removed)
        """
        self.threshold = threshold
        self.weights = None
        self.columns = None

    def fit(self, df: pd.DataFrame, continuous_cols: list[str],
            categorical_cols: list[str], ordinal_cols: list[str]) -> "DependencyGraph":
        """Compute pairwise dependency weights from training data."""
        all_cols = continuous_cols + categorical_cols + ordinal_cols
        self.columns = all_cols
        n = len(all_cols)
        W = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                col_i, col_j = all_cols[i], all_cols[j]
                xi = df[col_i].values
                xj = df[col_j].values

                i_is_cont = col_i in continuous_cols
                j_is_cont = col_j in continuous_cols
                i_is_cat = col_i in categorical_cols or col_i in ordinal_cols
                j_is_cat = col_j in categorical_cols or col_j in ordinal_cols

                if i_is_cont and j_is_cont:
                    # Pearson correlation (absolute value)
                    mask = ~(np.isnan(xi.astype(float)) | np.isnan(xj.astype(float)))
                    if mask.sum() > 2:
                        w = abs(np.corrcoef(xi[mask].astype(float),
                                            xj[mask].astype(float))[0, 1])
                        W[i, j] = w if not np.isnan(w) else 0.0
                    else:
                        W[i, j] = 0.0
                elif i_is_cat and j_is_cat:
                    # Cramer's V
                    W[i, j] = cramers_v(xi, xj)
                else:
                    # Normalized mutual information (mixed pairs)
                    W[i, j] = compute_nmi(xi, xj)

                W[j, i] = W[i, j]

        # Normalize to [0, 1]
        max_w = W.max()
        if max_w > 0:
            W = W / max_w

        # Apply threshold for sparsity
        W[W < self.threshold] = 0.0

        self.weights = W
        return self

    def get_weights(self) -> np.ndarray:
        """Return the weight matrix [n_vars, n_vars]."""
        return self.weights

    def get_neighbors(self, var_idx: int) -> list[tuple[int, float]]:
        """Return (neighbor_index, weight) pairs for a variable."""
        neighbors = []
        for j in range(len(self.columns)):
            if j != var_idx and self.weights[var_idx, j] > 0:
                neighbors.append((j, self.weights[var_idx, j]))
        return neighbors
