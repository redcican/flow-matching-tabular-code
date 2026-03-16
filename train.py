"""Main training script for product manifold flow matching.

Usage:
    python train.py --data path/to/data.csv --config configs/default.yaml
    python train.py --data path/to/data.csv --continuous age income --categorical gender disease
"""

import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from src.data.preprocessing import TabularPreprocessor
from src.data.dataset import MixedTypeDataset
from src.data.dependency import DependencyGraph
from src.training.flow_matching import ProductManifoldFlowMatching, Trainer
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Train product manifold flow matching")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--continuous", nargs="*", default=None,
                        help="Continuous column names")
    parser.add_argument("--categorical", nargs="*", default=None,
                        help="Categorical column names")
    parser.add_argument("--ordinal", nargs="*", default=None,
                        help="Ordinal column names")
    parser.add_argument("--target", type=str, default=None,
                        help="Target column to exclude")
    parser.add_argument("--output", type=str, default="checkpoints",
                        help="Output directory for model checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and preprocess data
    df = pd.read_csv(args.data)
    if args.target and args.target in df.columns:
        df = df.drop(columns=[args.target])

    preprocessor = TabularPreprocessor(
        continuous_cols=args.continuous,
        categorical_cols=args.categorical,
        ordinal_cols=args.ordinal,
    )
    preprocessor.fit(df)

    col_info = preprocessor.get_column_info()
    print(f"Column info: {col_info.continuous_dims} continuous, "
          f"{len(col_info.categorical_dims)} categorical "
          f"({col_info.categorical_dims}), "
          f"{col_info.ordinal_dims} ordinal")

    # Build dependency graph
    dep_graph = DependencyGraph(threshold=config["dependency"]["threshold"])
    dep_graph.fit(
        df,
        preprocessor.mapping.continuous_cols,
        preprocessor.mapping.categorical_cols,
        preprocessor.mapping.ordinal_cols,
    )
    n_edges = (dep_graph.get_weights() > 0).sum() // 2
    print(f"Dependency graph: {n_edges} edges above threshold "
          f"{config['dependency']['threshold']}")

    # Create dataset
    tensor_data = preprocessor.transform(df)
    dataset = MixedTypeDataset(tensor_data)

    # Train/val split
    n_val = int(len(dataset) * config["data"]["test_ratio"])
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    # Create model
    model = ProductManifoldFlowMatching(
        col_info=col_info,
        hidden_dim=config["model"]["hidden_dim"],
        n_layers=config["model"]["n_layers"],
        alpha=config["model"]["alpha"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, "best_model.pt")

    trainer = Trainer(model, lr=config["training"]["lr"], device=device)
    history = trainer.train(
        train_loader,
        n_epochs=config["training"]["n_epochs"],
        val_dataloader=val_loader,
        patience=config["training"]["patience"],
        save_path=save_path,
    )

    # Save preprocessor
    import pickle
    with open(os.path.join(args.output, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    print(f"Training complete. Model saved to {save_path}")
    print(f"Final loss: {history[-1]['total']:.4f}")


if __name__ == "__main__":
    main()
