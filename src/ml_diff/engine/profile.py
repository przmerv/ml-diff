"""Summarize a dataset column by column before any modeling starts."""

import pandas as pd


def column_kind(s: pd.Series) -> str:
    """Label a column as "boolean", "numeric" or "categorical".

    Boolean is checked first because pandas also treats True/False as numbers.
    Anything that is neither boolean nor numeric, such as text labels, is
    treated as categorical. The label decides which stats are shown later.
    """

    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    elif pd.api.types.is_numeric_dtype(s):
        return "numeric"
    else:
        return "categorical"


def profile_column(s: pd.Series) -> dict[str, object]:
    """Build a one-row summary of a single column.

    Always includes the column name, kind, number and percent of missing
    values, and number of distinct values. For numeric columns, also includes
    mean, std, min and max; other kinds get None for those fields.
    """
    kind = column_kind(s)
    na_sum = s.isna().sum()

    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None

    if kind == "numeric":
        mean = s.mean()
        std = s.std()
        min_val = s.min()
        max_val = s.max()

    summary = {
        "column": s.name,
        "kind": column_kind(s),
        "missing": na_sum,
        "missing_pct": na_sum / len(s) * 100,
        "distinct": s.nunique(dropna=True),
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
    }

    return summary


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize every column of a dataset, one row per column.

    Runs profile_column on each column with a plain loop and stacks the
    results into a table, so column types, missing values and basic stats
    can be checked at a glance before choosing a target or a model.
    """
    summaries = []
    for col in df.columns:
        summaries.append(profile_column(df[col]))

    return pd.DataFrame(summaries)
