"""Command-line entry point for end-to-end housing analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .models import evaluate_models
from .preprocessing import prepare_data
from .reports import save_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean housing data, compare models, explain drivers, and forecast prices."
    )
    parser.add_argument("--data", type=Path, required=True, help="Historical housing CSV.")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--target-column", default="price")
    parser.add_argument("--features", help="Comma-separated predictor names; defaults to numeric columns.")
    parser.add_argument("--macro-data", type=Path, help="Optional date-indexed economic indicator CSV.")
    parser.add_argument("--area-lookup", type=Path, help="Optional area_id lookup CSV.")
    parser.add_argument("--forecast-csv", type=Path, help="Optional future dates and known predictor inputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> dict[str, object]:
    if not args.data.is_file():
        raise FileNotFoundError(f"Historical data file does not exist: {args.data}")
    housing = pd.read_csv(args.data)
    requested = [item.strip() for item in args.features.split(",") if item.strip()] if args.features else []
    macro = pd.read_csv(args.macro_data) if args.macro_data else None
    area_lookup = pd.read_csv(args.area_lookup) if args.area_lookup else None

    if not requested and macro is not None:
        requested = [
            column for column in macro.columns
            if column != args.date_column
        ]
    if not requested:
        requested = [
            column for column in housing.select_dtypes(include="number").columns
            if column != args.target_column
        ]

    data = prepare_data(
        housing, args.date_column, args.target_column, requested,
        macro=macro, area_lookup=area_lookup,
    )
    evaluated = evaluate_models(
        data, args.date_column, args.target_column, requested, args.train_fraction
    )
    summary = save_results(
        evaluated, args.date_column, args.target_column, requested, args.output_dir
    )

    if args.forecast_csv:
        future_path = args.forecast_csv
        if not future_path.is_file():
            raise FileNotFoundError(f"Future-input file does not exist: {future_path}")
        future = pd.read_csv(future_path)
        required = set(requested + [args.date_column])
        missing = required.difference(future.columns)
        if missing:
            raise ValueError(f"Future data is missing columns: {', '.join(sorted(missing))}")
        future[args.date_column] = pd.to_datetime(future[args.date_column], errors="coerce")
        future = future.dropna(subset=[args.date_column]).sort_values(args.date_column)
        for column in requested:
            future[column] = pd.to_numeric(future[column], errors="coerce")
        if future.empty:
            raise ValueError("Future-input data has no valid dates.")
        future["predicted_price"] = evaluated["best_model"].predict(future[requested])
        future[[args.date_column, "predicted_price"]].to_csv(
            args.output_dir / "future_predictions.csv", index=False
        )
        summary["future_prediction_rows"] = int(len(future))

    print("Chronological holdout model comparison")
    print(evaluated["comparison"].to_string(index=False))
    print(f"\nBest model by MAE: {summary['best_model_by_mae']}")
    print("\nPermutation importance")
    print(evaluated["importance"].to_string(index=False))
    print(f"\nResults, charts, and local database saved to: {args.output_dir.resolve()}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        execute(args)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    return 0
