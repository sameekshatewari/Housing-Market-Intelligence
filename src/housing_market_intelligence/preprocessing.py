"""CSV validation, joining, cleaning, and temporal splitting."""

from __future__ import annotations

import math

import pandas as pd


def prepare_data(
    housing: pd.DataFrame,
    date_column: str,
    target_column: str,
    features: list[str],
    macro: pd.DataFrame | None = None,
    area_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate and clean housing data, optionally joining macro and area tables."""
    data = housing.copy()
    required = {date_column, target_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Housing data is missing columns: {', '.join(sorted(missing))}")

    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
    data = data.dropna(subset=[date_column, target_column])
    data = data.drop_duplicates().sort_values(date_column)

    if macro is not None:
        if date_column not in macro.columns:
            raise ValueError(f"Macro data is missing date column '{date_column}'.")
        macro_data = macro.copy()
        macro_data[date_column] = pd.to_datetime(macro_data[date_column], errors="coerce")
        macro_data = macro_data.dropna(subset=[date_column]).drop_duplicates(date_column)
        macro_fields = [column for column in macro_data if column != date_column]
        collisions = set(macro_fields).intersection(data.columns)
        if collisions:
            raise ValueError(f"Macro columns already exist in housing data: {', '.join(sorted(collisions))}")
        data = data.merge(macro_data, on=date_column, how="left", validate="many_to_one")

    if area_lookup is not None:
        area_key = "area_id"
        if area_key not in data.columns or area_key not in area_lookup.columns:
            raise ValueError("Housing data and area lookup must both contain 'area_id'.")
        lookup = area_lookup.drop_duplicates(area_key)
        area_fields = [column for column in lookup if column != area_key]
        collisions = set(area_fields).intersection(data.columns)
        if collisions:
            raise ValueError(f"Area columns already exist in housing data: {', '.join(sorted(collisions))}")
        data = data.merge(lookup, on=area_key, how="left", validate="many_to_one")

    missing_features = set(features).difference(data.columns)
    if missing_features:
        raise ValueError(f"Feature columns are missing: {', '.join(sorted(missing_features))}")
    if not features:
        raise ValueError("At least one predictor feature is required.")

    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[features].notna().sum().eq(0).any():
        empty = [column for column in features if data[column].notna().sum() == 0]
        raise ValueError(f"Feature columns have no numeric values: {', '.join(empty)}")
    if data.empty:
        raise ValueError("No usable housing rows remain after cleaning.")
    return data.reset_index(drop=True)


def chronological_split(
    data: pd.DataFrame, date_column: str, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations by distinct dates without leaking dates across sets."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    unique_dates = data[date_column].dropna().drop_duplicates().sort_values().tolist()
    if len(unique_dates) < 3:
        raise ValueError("At least three distinct dates are required for a chronological holdout.")
    split_index = min(max(math.ceil(len(unique_dates) * train_fraction), 2), len(unique_dates) - 1)
    cutoff = unique_dates[split_index]
    train = data[data[date_column] < cutoff].copy()
    test = data[data[date_column] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Could not form non-empty chronological train and test sets.")
    return train, test


def aggregate_monthly(
    data: pd.DataFrame, date_column: str, target_column: str, group_column: str | None = None
) -> pd.DataFrame:
    """Aggregate dated observations to monthly mean target values."""
    frame = data.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.to_period("M").dt.to_timestamp()
    frame = frame.dropna(subset=[date_column, target_column])
    keys = [date_column] + ([group_column] if group_column else [])
    return frame.groupby(keys, as_index=False)[target_column].mean().sort_values(keys)
