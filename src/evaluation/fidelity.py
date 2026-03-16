"""Statistical fidelity metrics: Wasserstein-1 distance and Jensen-Shannon divergence."""

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


def compute_w1(real: np.ndarray, synthetic: np.ndarray,
               continuous_indices: list[int]) -> float:
    """Compute marginal-wise Wasserstein-1 distance for continuous variables.

    W1(P, Q) = inf_{gamma in Gamma(P,Q)} E_{(x,y)~gamma}[|x - y|]

    Returns average W1 across all continuous marginals (x10^{-2}).
    """
    if not continuous_indices:
        return 0.0

    w1_values = []
    for idx in continuous_indices:
        w1 = wasserstein_distance(real[:, idx], synthetic[:, idx])
        w1_values.append(w1)

    return np.mean(w1_values) * 100  # x10^{-2}


def compute_js(real: np.ndarray, synthetic: np.ndarray,
               categorical_indices: list[tuple[int, int]],
               n_bins: int = 50) -> float:
    """Compute Jensen-Shannon divergence for categorical variables.

    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), M = 0.5*(P+Q)

    Args:
        real: real data array
        synthetic: synthetic data array
        categorical_indices: list of (start_idx, n_categories) tuples
        n_bins: number of bins for discretization

    Returns average JS across all categorical marginals (x10^{-2}).
    """
    if not categorical_indices:
        return 0.0

    js_values = []
    for start_idx, n_cat in categorical_indices:
        # Get category assignments via argmax
        real_cats = np.argmax(real[:, start_idx:start_idx + n_cat], axis=1)
        syn_cats = np.argmax(synthetic[:, start_idx:start_idx + n_cat], axis=1)

        # Compute empirical distributions
        real_hist = np.bincount(real_cats, minlength=n_cat).astype(float)
        syn_hist = np.bincount(syn_cats, minlength=n_cat).astype(float)

        # Normalize
        real_hist = real_hist / real_hist.sum()
        syn_hist = syn_hist / syn_hist.sum()

        # Add small epsilon for numerical stability
        eps = 1e-10
        real_hist = real_hist + eps
        syn_hist = syn_hist + eps
        real_hist = real_hist / real_hist.sum()
        syn_hist = syn_hist / syn_hist.sum()

        js = jensenshannon(real_hist, syn_hist) ** 2  # scipy returns sqrt(JS)
        js_values.append(js)

    return np.mean(js_values) * 100  # x10^{-2}
