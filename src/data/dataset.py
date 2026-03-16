"""PyTorch Dataset for mixed-type tabular data on product manifolds."""

import torch
from torch.utils.data import Dataset
import pandas as pd

from .preprocessing import TabularPreprocessor


class MixedTypeDataset(Dataset):
    """Dataset wrapper for preprocessed mixed-type tabular data.

    Stores data as product manifold tensors ready for flow matching training.
    """

    def __init__(self, data: torch.Tensor):
        """
        Args:
            data: preprocessed tensor on the product manifold [N, D]
        """
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       preprocessor: TabularPreprocessor) -> "MixedTypeDataset":
        """Create dataset from a DataFrame using a fitted preprocessor."""
        tensor = preprocessor.transform(df)
        return cls(tensor)

    @classmethod
    def from_csv(cls, path: str, preprocessor: TabularPreprocessor | None = None,
                 continuous_cols: list[str] | None = None,
                 categorical_cols: list[str] | None = None,
                 ordinal_cols: list[str] | None = None,
                 target_col: str | None = None) -> tuple["MixedTypeDataset", TabularPreprocessor]:
        """Load dataset from CSV with automatic or manual type detection.

        Args:
            path: path to CSV file
            preprocessor: fitted preprocessor (if None, will fit a new one)
            continuous_cols: column names for continuous variables
            categorical_cols: column names for categorical variables
            ordinal_cols: column names for ordinal variables
            target_col: target column to exclude from generation

        Returns:
            dataset: MixedTypeDataset
            preprocessor: fitted TabularPreprocessor
        """
        df = pd.read_csv(path)

        if target_col and target_col in df.columns:
            df = df.drop(columns=[target_col])

        if preprocessor is None:
            preprocessor = TabularPreprocessor(
                continuous_cols=continuous_cols,
                categorical_cols=categorical_cols,
                ordinal_cols=ordinal_cols,
            )
            preprocessor.fit(df)

        tensor = preprocessor.transform(df)
        return cls(tensor), preprocessor
