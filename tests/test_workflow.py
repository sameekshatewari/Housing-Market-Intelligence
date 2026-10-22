from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from argparse import Namespace
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from housing_market_intelligence.cli import DATASET_NAME, execute
from housing_market_intelligence.models import BASELINE_NAMES, evaluate_models
from housing_market_intelligence.preprocessing import (
    DATE_COLUMN,
    FORECAST_FEATURES,
    TARGET_COLUMN,
    aggregate_monthly,
    chronological_split,
    load_zhvi_csv,
    prepare_data,
)
from housing_market_intelligence.reports import (
    calculate_wape_percent,
    make_market_error_report,
)


def make_wide_zhvi(months: int = 48) -> pd.DataFrame:
    dates = pd.date_range(
        "2019-01-31", periods=months, freq=pd.offsets.MonthEnd()
    )
    rows: list[dict[str, object]] = [
        {
            "RegionID": 0,
            "SizeRank": 0,
            "RegionName": "United States",
            "RegionType": "country",
            "StateName": "",
        }
    ]
    for region_index, (name, state) in enumerate(
        (("Metro One, ST", "ST"), ("Metro Two, XY", "XY"), ("Metro Three, ZZ", "ZZ"))
    ):
        row: dict[str, object] = {
            "RegionID": region_index + 1,
            "SizeRank": region_index + 1,
            "RegionName": name,
            "RegionType": "msa",
            "StateName": state,
        }
        for month_index, date in enumerate(dates):
            value = (
                180_000
                + region_index * 55_000
                + month_index * (700 + region_index * 70)
                + 900 * np.sin(month_index / 4)
            )
            row[date.strftime("%Y-%m-%d")] = round(value, 2)
        rows.append(row)
    return pd.DataFrame(rows)


class ZillowPreprocessingTests(unittest.TestCase):
    def test_wide_zhvi_csv_is_melted_and_non_metro_rows_are_excluded(self) -> None:
        wide = make_wide_zhvi(months=16)
        wide.loc[1, "2019-06-30"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zhvi.csv"
            wide.to_csv(path, index=False)
            tidy = load_zhvi_csv(path)

        self.assertEqual(
            list(tidy.columns), ["RegionName", "StateName", "Date", "HomeValue"]
        )
        self.assertEqual(tidy["RegionName"].nunique(), 3)
        self.assertNotIn("United States", set(tidy["RegionName"]))
        self.assertEqual(len(tidy), 3 * 16 - 1)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(tidy["Date"]))
        self.assertFalse(tidy["HomeValue"].isna().any())

    def test_loader_reports_missing_file_and_unexpected_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "data/zhvi_metro.csv"):
                load_zhvi_csv(root / "missing.csv")
            invalid = root / "invalid.csv"
            pd.DataFrame({"market": ["Example"], "2024-01-31": [100]}).to_csv(
                invalid, index=False
            )
            with self.assertRaisesRegex(ValueError, "RegionName"):
                load_zhvi_csv(invalid)

    def test_lags_use_prior_months_of_the_same_metro(self) -> None:
        tidy = load_zhvi_csv_from_frame(make_wide_zhvi())
        prepared = prepare_data(tidy)
        row = prepared.loc[
            (prepared["RegionName"] == "Metro One, ST")
            & (prepared[DATE_COLUMN] == pd.Timestamp("2020-02-29"))
        ].iloc[0]
        self.assertAlmostEqual(
            row["lag_1"],
            180_000 + 12 * 700 + 900 * np.sin(12 / 4),
            places=2,
        )
        prior_year = tidy.loc[
            (tidy["RegionName"] == "Metro One, ST")
            & (tidy[DATE_COLUMN] == pd.Timestamp("2019-02-28")),
            TARGET_COLUMN,
        ].iloc[0]
        self.assertAlmostEqual(row["lag_12"], prior_year)
        thirteen_months_prior = tidy.loc[
            (tidy["RegionName"] == "Metro One, ST")
            & (tidy[DATE_COLUMN] == pd.Timestamp("2019-01-31")),
            TARGET_COLUMN,
        ].iloc[0]
        self.assertAlmostEqual(
            row["year_over_year_growth"],
            row["lag_1"] / thirteen_months_prior - 1,
        )
        self.assertEqual(list(FORECAST_FEATURES), [
            "lag_1", "lag_3", "lag_6", "lag_12",
            "year_over_year_growth", "month_sin", "month_cos",
        ])

    def test_monthly_aggregation_can_keep_markets_separate(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": ["2024-01-03", "2024-01-24", "2024-01-15"],
                "RegionName": ["Metro A", "Metro A", "Metro B"],
                "HomeValue": [100.0, 200.0, 300.0],
            }
        )
        result = aggregate_monthly(frame, group_column="RegionName")
        values = dict(zip(result["RegionName"], result["HomeValue"]))
        self.assertEqual(values, {"Metro A": 150.0, "Metro B": 300.0})

    def test_chronological_split_keeps_all_markets_for_each_month_together(self) -> None:
        prepared = prepare_data(load_zhvi_csv_from_frame(make_wide_zhvi()))
        train, validation, test = chronological_split(prepared)
        self.assertLess(train[DATE_COLUMN].max(), validation[DATE_COLUMN].min())
        self.assertLess(validation[DATE_COLUMN].max(), test[DATE_COLUMN].min())
        self.assertEqual(
            set(train[DATE_COLUMN])
            | set(validation[DATE_COLUMN])
            | set(test[DATE_COLUMN]),
            set(prepared[DATE_COLUMN]),
        )


def load_zhvi_csv_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "zhvi.csv"
        frame.to_csv(path, index=False)
        return load_zhvi_csv(path)


class ForecastWorkflowTests(unittest.TestCase):
    def test_cli_runs_models_and_writes_reports_and_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "zhvi.csv"
            make_wide_zhvi().to_csv(source_path, index=False)
            output = root / "outputs"
            summary = execute(
                Namespace(
                    data=source_path,
                    output_dir=output,
                    train_fraction=0.7,
                    validation_fraction=0.15,
                )
            )

            self.assertEqual(summary["dataset_name"], DATASET_NAME)
            self.assertEqual(summary["metro_count"], 3)
            self.assertLess(
                summary["training_end"], summary["validation_start"]
            )
            self.assertLess(summary["validation_end"], summary["test_start"])
            self.assertEqual(
                summary["training_rows"]
                + summary["validation_rows"]
                + summary["test_rows"],
                summary["modeling_rows"],
            )

            comparison = pd.read_csv(output / "model_comparison.csv")
            self.assertEqual(
                set(comparison["model"]),
                {
                    "Linear regression",
                    "Random forest",
                    "Gradient boosting",
                    *BASELINE_NAMES,
                },
            )
            self.assertIn(
                summary["best_model_by_validation_mae"], set(comparison["model"])
            )
            self.assertEqual(list(comparison.columns), ["model", "mae", "rmse", "r2"])
            validation = pd.read_csv(output / "validation_comparison.csv")
            self.assertEqual(len(validation), 5)
            predictions = pd.read_csv(output / "holdout_predictions.csv")
            self.assertTrue({"RegionName", "StateName", "Date", "HomeValue"}.issubset(
                predictions.columns
            ))
            self.assertGreater(len(predictions), 0)
            market_errors = pd.read_csv(output / "validation_errors_by_metro.csv")
            self.assertEqual(len(market_errors), 3)
            self.assertEqual(
                set(market_errors["selected_model"]),
                {summary["best_model_by_validation_mae"]},
            )
            self.assertEqual(
                market_errors["selected_model_mae"].tolist(),
                sorted(market_errors["selected_model_mae"], reverse=True),
            )
            self.assertTrue(
                {
                    "persistence_mae",
                    "persistence_rmse",
                    "persistence_wape_percent",
                    "seasonal_naive_mae",
                    "seasonal_naive_rmse",
                    "seasonal_naive_wape_percent",
                    "selected_model_wape_percent",
                    "validation_start",
                    "validation_end",
                }.issubset(market_errors.columns)
            )
            self.assertEqual(
                set(market_errors["validation_start"]),
                {summary["validation_start"]},
            )
            self.assertIn("Selected by validation MAE", (output / "dashboard.html").read_text())
            self.assertIn(DATASET_NAME, (output / "dashboard.html").read_text())
            dashboard = (output / "dashboard.html").read_text()
            self.assertIn("Markets with highest validation error", dashboard)
            self.assertIn("WAPE is total absolute error", dashboard)
            for name in (
                "model_comparison.svg",
                "holdout_predictions.svg",
                "price_distribution.svg",
                "predicted_vs_actual.svg",
                "factor_importance.svg",
            ):
                ElementTree.parse(output / "charts" / name)

            chart_root = ElementTree.parse(
                output / "charts" / "holdout_predictions.svg"
            ).getroot()
            namespace = {"svg": "http://www.w3.org/2000/svg"}
            chart_lines = chart_root.findall(".//svg:polyline", namespace)
            self.assertEqual(len(chart_lines), 4)
            series_by_column = {
                line.attrib["data-series"]: line.attrib["stroke"]
                for line in chart_lines
            }
            self.assertEqual(
                set(series_by_column),
                {
                    "HomeValue",
                    f"{summary['best_model_by_validation_mae']} prediction",
                    f"{BASELINE_NAMES[0]} prediction",
                    f"{BASELINE_NAMES[1]} prediction",
                },
            )
            self.assertEqual(len(set(series_by_column.values())), 4)
            chart_text = " ".join(
                node.text or ""
                for node in chart_root.findall(".//svg:text", namespace)
            )
            self.assertIn("Actual unweighted metro mean", chart_text)
            self.assertIn("selected by validation MAE", chart_text)
            self.assertIn(BASELINE_NAMES[0], chart_text)
            self.assertIn(BASELINE_NAMES[1], chart_text)

            with closing(sqlite3.connect(output / "housing_analysis.sqlite")) as db:
                tables = {
                    row[0]
                    for row in db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertTrue(
                    {
                        "model_metrics",
                        "validation_metrics",
                        "validation_metro_metrics",
                        "holdout_predictions",
                        "factor_importance",
                        "run_metadata",
                    }.issubset(tables)
                )

    def test_market_error_report_uses_validation_predictions_and_ranks_by_mae(self) -> None:
        prepared = prepare_data(load_zhvi_csv_from_frame(make_wide_zhvi()))
        results = evaluate_models(prepared, importance_repeats=2)
        report = make_market_error_report(
            results["validation_predictions"], results["best_model_name"]
        )

        self.assertEqual(len(report), prepared["RegionName"].nunique())
        self.assertTrue(
            report["selected_model_mae"].is_monotonic_decreasing
        )
        self.assertTrue(
            (report["validation_start"] == results["validation"]["Date"].min().date().isoformat()).all()
        )
        self.assertTrue(report["selected_model_wape_percent"].notna().all())

    def test_wape_handles_zero_and_near_zero_denominators(self) -> None:
        self.assertIsNone(
            calculate_wape_percent(
                pd.Series([0.0, 0.5]), pd.Series([0.0, 1.0])
            )
        )
        self.assertEqual(
            calculate_wape_percent(
                pd.Series([100.0, 100.0]), pd.Series([90.0, 90.0])
            ),
            10.0,
        )

    def test_model_selection_uses_validation_not_test_scores(self) -> None:
        prepared = prepare_data(load_zhvi_csv_from_frame(make_wide_zhvi()))
        results = evaluate_models(prepared, importance_repeats=2)
        self.assertEqual(
            results["best_model_name"],
            results["validation_comparison"].iloc[0]["model"],
        )
        self.assertLess(
            results["train"][DATE_COLUMN].max(),
            results["validation"][DATE_COLUMN].min(),
        )
        self.assertLess(
            results["validation"][DATE_COLUMN].max(),
            results["test"][DATE_COLUMN].min(),
        )
        self.assertEqual(len(results["comparison"]), 5)

    def test_missing_real_csv_does_not_fall_back_to_generated_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "Zillow data file not found"):
                execute(
                    Namespace(
                        data=root / "missing.csv",
                        output_dir=root / "outputs",
                        train_fraction=0.7,
                        validation_fraction=0.15,
                    )
                )
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
