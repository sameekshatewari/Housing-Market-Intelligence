"""Command-line workflow for Zillow metro-level housing forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .models import evaluate_models
from .preprocessing import load_zhvi_csv, prepare_data
from .reports import save_results

DATASET_NAME = (
    "Zillow ZHVI: middle-tier, all homes, seasonally adjusted, monthly, metro"
)
ZILLOW_DATA_PAGE = "https://www.zillow.com/research/data/"
ZILLOW_CSV_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare time-series forecasting baselines and ML models on "
            "Zillow's metro-level ZHVI data."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/zhvi_metro.csv"),
        help="Local Zillow Metro ZHVI CSV (default: data/zhvi_metro.csv).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    tidy = load_zhvi_csv(args.data)
    prepared = prepare_data(tidy)
    evaluated = evaluate_models(
        prepared,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    summary = save_results(
        evaluated,
        output_dir=args.output_dir,
        metadata={
            "dataset_name": DATASET_NAME,
            "source_page": ZILLOW_DATA_PAGE,
            "source_csv": ZILLOW_CSV_URL,
            "source_file": args.data.name,
            "geography": "Zillow metro rows (RegionType=msa)",
            "home_value_measure": "ZHVI, middle tier, all homes, seasonally adjusted",
            "observation_frequency": "monthly",
            "metro_count": int(tidy["RegionName"].nunique()),
            "modeling_metro_count": int(prepared["RegionName"].nunique()),
            "observation_start": str(tidy["Date"].min().date()),
            "observation_end": str(tidy["Date"].max().date()),
            "modeling_start": str(prepared["Date"].min().date()),
            "modeling_end": str(prepared["Date"].max().date()),
            "tidy_rows": int(len(tidy)),
            "modeling_rows": int(len(prepared)),
        },
    )

    print("Validation scores (model selection by lowest validation MAE)")
    print(evaluated["validation_comparison"].to_string(index=False))
    print("\nFinal chronological test scores")
    print(evaluated["comparison"].to_string(index=False))
    print(f"\nSelected model: {summary['best_model_by_validation_mae']}")
    print(
        "Test period: "
        f"{summary['test_start']} through {summary['test_end']}"
    )
    print(f"Results, charts, and local database saved to: {args.output_dir.resolve()}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        execute(args)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error
    return 0
