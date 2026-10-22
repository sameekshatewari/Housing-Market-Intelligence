"""Load Zillow's wide CSV and prepare metro-month forecasting features."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

REQUIRED_ZHVI_COLUMNS = {"RegionName", "RegionType", "StateName"}
DATE_COLUMN_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TARGET_COLUMN = "HomeValue"
DATE_COLUMN = "Date"
MARKET_COLUMNS = ["RegionName", "StateName"]
FORECAST_FEATURES = [
    "lag_1",
    "lag_3",
    "lag_6",
    "lag_12",
    "year_over_year_growth",
    "month_sin",
    "month_cos",
]


def load_zhvi_csv(path: Path) -> pd.DataFrame:
    """Read a Zillow metro ZHVI CSV and reshape its monthly columns to rows."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Zillow data file not found: {path}. Download the Metro ZHVI CSV "
            "and save it to data/zhvi_metro.csv; see README.md for instructions."
        )

    wide = pd.read_csv(path)
    wide.columns = wide.columns.astype(str).str.strip()
    missing = REQUIRED_ZHVI_COLUMNS.difference(wide.columns)
    if missing:
        raise ValueError(
            "Zillow CSV is missing required metro columns: "
            f"{', '.join(sorted(missing))}. Received columns: "
            f"{', '.join(wide.columns[:12])}"
        )

    date_columns = [
        column for column in wide.columns if DATE_COLUMN_PATTERN.fullmatch(column)
    ]
    if not date_columns:
        raise ValueError(
            "No monthly YYYY-MM-DD date columns were found in the Zillow CSV."
        )

    region_types = wide["RegionType"].astype("string").str.strip().str.lower()
    metro_rows = wide.loc[region_types.isin({"msa", "metro"})].copy()
    if metro_rows.empty:
        found_types = sorted(region_types.dropna().unique().tolist())
        raise ValueError(
            "No metro rows found: expected RegionType 'msa' or 'metro'; "
            f"found {found_types}."
        )

    metro_rows["RegionName"] = metro_rows["RegionName"].astype("string").str.strip()
    metro_rows["StateName"] = metro_rows["StateName"].astype("string").str.strip()
    id_columns = MARKET_COLUMNS
    tidy = metro_rows.melt(
        id_vars=id_columns,
        value_vars=date_columns,
        var_name=DATE_COLUMN,
        value_name=TARGET_COLUMN,
    )
    tidy[DATE_COLUMN] = pd.to_datetime(
        tidy[DATE_COLUMN], format="%Y-%m-%d", errors="coerce"
    )
    tidy[TARGET_COLUMN] = pd.to_numeric(tidy[TARGET_COLUMN], errors="coerce")
    tidy = tidy.dropna(subset=MARKET_COLUMNS + [DATE_COLUMN, TARGET_COLUMN])
    tidy = tidy.loc[
        tidy["RegionName"].ne("") & tidy["StateName"].ne("")
    ].copy()
    if tidy.empty:
        raise ValueError("No usable metro-month home values remain in the Zillow CSV.")
    if tidy.duplicated(MARKET_COLUMNS + [DATE_COLUMN]).any():
        raise ValueError("Zillow CSV contains duplicate metro-month observations.")
    return tidy.sort_values(MARKET_COLUMNS + [DATE_COLUMN]).reset_index(drop=True)


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate tidy Zillow observations and create only past-known features."""
    required = set(MARKET_COLUMNS + [DATE_COLUMN, TARGET_COLUMN])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Tidy Zillow data is missing columns: {', '.join(sorted(missing))}"
        )

    frame = data[MARKET_COLUMNS + [DATE_COLUMN, TARGET_COLUMN]].copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame["RegionName"] = frame["RegionName"].astype("string").str.strip()
    frame["StateName"] = frame["StateName"].astype("string").str.strip()
    frame = frame.dropna(subset=MARKET_COLUMNS + [DATE_COLUMN, TARGET_COLUMN])
    frame = frame.loc[
        frame["RegionName"].ne("") & frame["StateName"].ne("")
    ].copy()
    frame["_month"] = frame[DATE_COLUMN].dt.to_period("M")
    if frame.duplicated(MARKET_COLUMNS + ["_month"]).any():
        raise ValueError("Tidy data contains duplicate metro-month observations.")
    if frame.empty:
        raise ValueError("No valid metro-month observations remain after cleaning.")

    history = frame[MARKET_COLUMNS + ["_month", TARGET_COLUMN]].copy()
    for lag in (1, 3, 6, 12, 13):
        lagged = history.copy()
        lagged["_month"] = lagged["_month"] + lag
        lag_column = "lag_13_for_growth" if lag == 13 else f"lag_{lag}"
        lagged = lagged.rename(columns={TARGET_COLUMN: lag_column})
        frame = frame.merge(
            lagged,
            on=MARKET_COLUMNS + ["_month"],
            how="left",
            validate="one_to_one",
        )

    frame["year_over_year_growth"] = frame["lag_1"] / frame["lag_13_for_growth"] - 1
    month = frame[DATE_COLUMN].dt.month
    frame["month_sin"] = (2 * math.pi * month / 12).map(math.sin)
    frame["month_cos"] = (2 * math.pi * month / 12).map(math.cos)
    frame = frame.dropna(subset=FORECAST_FEATURES + ["lag_13_for_growth"])
    if frame.empty:
        raise ValueError(
            "No rows have the 13 months of history needed for forecasting features."
        )
    return frame.drop(columns=["_month", "lag_13_for_growth"]).reset_index(drop=True)


def chronological_split(
    data: pd.DataFrame,
    date_column: str = DATE_COLUMN,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split distinct months chronologically into train, validation and test."""
    if (
        not math.isfinite(train_fraction)
        or not math.isfinite(validation_fraction)
        or train_fraction <= 0
        or validation_fraction <= 0
        or train_fraction + validation_fraction >= 1
    ):
        raise ValueError(
            "Train and validation fractions must be positive and sum to less than 1."
        )
    unique_dates = data[date_column].dropna().drop_duplicates().sort_values().tolist()
    if len(unique_dates) < 6:
        raise ValueError(
            "At least six distinct months are required for train/validation/test splits."
        )

    train_end = min(max(math.ceil(len(unique_dates) * train_fraction), 2), len(unique_dates) - 2)
    validation_end = min(
        max(math.ceil(len(unique_dates) * (train_fraction + validation_fraction)), train_end + 1),
        len(unique_dates) - 1,
    )
    validation_start_date = unique_dates[train_end]
    test_start_date = unique_dates[validation_end]
    train = data[data[date_column] < validation_start_date].copy()
    validation = data[
        (data[date_column] >= validation_start_date)
        & (data[date_column] < test_start_date)
    ].copy()
    test = data[data[date_column] >= test_start_date].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Could not form non-empty chronological data splits.")
    return train, validation, test


def aggregate_monthly(
    data: pd.DataFrame,
    date_column: str = DATE_COLUMN,
    target_column: str = TARGET_COLUMN,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Aggregate observations to monthly means, optionally by market."""
    frame = data.copy()
    frame[date_column] = (
        pd.to_datetime(frame[date_column], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, target_column])
    if group_column:
        monthly = (
            frame.groupby([date_column, group_column])[target_column]
            .mean()
            .reset_index()
        )
    else:
        monthly = (
            frame.groupby(date_column)[target_column].mean().rename(target_column).reset_index()
        )
    return monthly.sort_values(by=date_column)
