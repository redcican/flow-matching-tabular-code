"""Generate synthetic data using trained product manifold flow matching model.

Usage:
    python generate.py --checkpoint checkpoints/best_model.pt --n_samples 1000
"""

import argparse
import os
import pickle
import torch
import pandas as pd

from src.training.flow_matching import ProductManifoldFlowMatching
from src.training.ode_solver import ManifoldODESolver


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic tabular data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--output", type=str, default="synthetic_data.csv",
                        help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="dopri5",
                        help="ODE solver method")
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="Generation batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load preprocessor
    checkpoint_dir = os.path.dirname(args.checkpoint)
    preprocessor_path = os.path.join(checkpoint_dir, "preprocessor.pkl")
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    col_info = preprocessor.get_column_info()

    # Create model and load weights
    model = ProductManifoldFlowMatching(col_info=col_info)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))
    model = model.to(device)

    # Generate via ODE integration
    solver = ManifoldODESolver(model, method=args.method)

    all_samples = []
    remaining = args.n_samples
    while remaining > 0:
        batch_n = min(remaining, args.batch_size)
        samples = solver.generate(batch_n, device)
        all_samples.append(samples.cpu())
        remaining -= batch_n
        print(f"Generated {args.n_samples - remaining}/{args.n_samples} samples")

    synthetic_tensor = torch.cat(all_samples, dim=0)

    # Convert back to DataFrame
    synthetic_df = preprocessor.inverse_transform(synthetic_tensor)
    synthetic_df.to_csv(args.output, index=False)

    print(f"Saved {len(synthetic_df)} synthetic samples to {args.output}")


if __name__ == "__main__":
    main()
