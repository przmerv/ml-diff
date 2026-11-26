"""Load a CSV and check that the chosen target column and task make sense."""

from pathlib import Path
from typing import Literal

import pandas as pd

Task = Literal["regression", "classification"]


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame and strip extra spaces from column names.

    Raises FileNotFoundError if the file does not exist, and ValueError if the
    file has no rows, so problems with the input are reported before any
    modeling starts.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file {path} has no rows.")
    df.columns = df.columns.str.strip()
    return df


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target column from the feature columns.

    Rows where the target is missing are dropped, because a model cannot learn
    from them or be scored on them. Raises ValueError, listing the available
    columns, if the target name is not found.
    """

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.dropna(subset=[target]).reset_index(drop=True)
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def check_task(y: pd.Series, task: Task) -> None:
    """Confirm that the target column suits the chosen task.

    Classification needs at least two distinct classes. Regression needs a
    numeric, non-boolean target. Raises ValueError with a plain explanation
    when the target and task do not match. Returns nothing when they do.
    """
    if task == "classification":
        n_classes = y.nunique()
        if n_classes < 2:
            raise ValueError(
                f"Classification needs at least 2 distinct classes, but the target has {n_classes}."
            )
    elif task == "regression":
        if pd.api.types.is_bool_dtype(y):
            raise ValueError("Regression needs a numeric target, but the target is True/False.")
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Regression needs a numeric target, but the target is {y.dtype}.")
    else:
        raise ValueError(f"Unknown task '{task}'. Choose 'classification' or 'regression'.")
