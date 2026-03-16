"""Evaluate synthetic data quality using fidelity, utility, and privacy metrics.

Usage:
    python evaluate.py --real data/real.csv --synthetic synthetic_data.csv \
                       --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import pickle
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.fidelity import compute_w1, compute_js
from src.evaluation.utility import compute_mle
from src.evaluation.privacy import compute_dcr
from src.manifolds.product import ProductManifold


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate synthetic data quality")
    parser.add_argument("--real", type=str, required=True, help="Path to real data CSV")
    parser.add_argument("--synthetic", type=str, required=True,
                        help="Path to synthetic data CSV")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--target", type=str, default=None,
                        help="Target column for MLE evaluation")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load preprocessor
    checkpoint_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(checkpoint_dir, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)

    col_info = preprocessor.get_column_info()

    # Load data
    real_df = pd.read_csv(args.real)
    synthetic_df = pd.read_csv(args.synthetic)

    # Separate target if provided
    target_real = None
    target_syn = None
    if args.target and args.target in real_df.columns:
        target_real = real_df[args.target].values
        real_df_features = real_df.drop(columns=[args.target])
        if args.target in synthetic_df.columns:
            target_syn = synthetic_df[args.target].values
            synthetic_df_features = synthetic_df.drop(columns=[args.target])
        else:
            synthetic_df_features = synthetic_df
    else:
        real_df_features = real_df
        synthetic_df_features = synthetic_df

    # Transform to manifold representation
    real_tensor = preprocessor.transform(real_df_features)
    syn_tensor = preprocessor.transform(synthetic_df_features)

    real_np = real_tensor.numpy()
    syn_np = syn_tensor.numpy()

    results = {}

    # --- Statistical Fidelity ---
    # Continuous indices
    cont_indices = list(range(col_info.continuous_dims))

    # Categorical indices: (start_idx, n_categories)
    cat_indices = []
    idx = col_info.continuous_dims
    for n_cat in col_info.categorical_dims:
        cat_indices.append((idx, n_cat))
        idx += n_cat

    w1 = compute_w1(real_np, syn_np, cont_indices)
    js = compute_js(real_np, syn_np, cat_indices)

    results["fidelity"] = {
        "w1_distance": round(w1, 4),
        "js_divergence": round(js, 4),
    }
    print(f"W1 Distance (x10^-2): {w1:.4f}")
    print(f"JS Divergence (x10^-2): {js:.4f}")

    # --- ML Efficacy ---
    if target_real is not None and target_syn is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            real_np, target_real, test_size=0.2, random_state=42
        )
        mle_results = compute_mle(X_train, X_test, syn_np,
                                  y_train, y_test, target_syn)
        results["utility"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in mle_results.items()
        }
        print(f"Average MLE: {mle_results['average_mle']:.4f}")

    # --- Privacy ---
    manifold = ProductManifold(col_info)
    dcr_results = compute_dcr(real_tensor, syn_tensor, manifold)
    results["privacy"] = {
        k: round(v, 4) for k, v in dcr_results.items()
    }
    print(f"DCR Proportion (x10^-2): {dcr_results['dcr_proportion']:.4f}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
