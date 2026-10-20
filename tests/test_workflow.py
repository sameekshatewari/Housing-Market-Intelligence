from __future__ import annotations

import sys
import tempfile
import unittest
import sqlite3
from argparse import Namespace
from contextlib import closing
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from housing_market_intelligence.models import evaluate_models
from housing_market_intelligence.cli import execute
from housing_market_intelligence.preprocessing import (
    aggregate_monthly,
    chronological_split,
    prepare_data,
)
from housing_market_intelligence.reports import save_results
from housing_market_intelligence.sample_data import generate_sample_data


class PreprocessingTests(unittest.TestCase):
    def test_joins_macro_and_area_data_and_drops_invalid_targets(self) -> None:
        housing = pd.DataFrame(
            {"date": ["2024-01-01", "2024-02-01", "invalid"], "area_id": ["A", "A", "A"],
             "price": [100.0, 110.0, 120.0]}
        )
        macro = pd.DataFrame({"date": ["2024-01-01", "2024-02-01"], "rate": [2.0, 3.0]})
        areas = pd.DataFrame({"area_id": ["A"], "market_name": ["Example"]})
        result = prepare_data(housing, "date", "price", ["rate"], macro, areas)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, "market_name"], "Example")
        self.assertEqual(result["rate"].tolist(), [2.0, 3.0])

    def test_rejects_missing_feature(self) -> None:
        with self.assertRaisesRegex(ValueError, "Feature columns are missing"):
            prepare_data(pd.DataFrame({"date": ["2024-01-01"], "price": [1]}),
                         "date", "price", ["unknown"])

    def test_chronological_split_keeps_identical_dates_together(self) -> None:
        frame = pd.DataFrame(
            {"date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01",
                                     "2024-03-01", "2024-04-01"]),
             "price": [1, 2, 3, 4, 5]}
        )
        train, test = chronological_split(frame, "date", 0.6)
        self.assertLess(train["date"].max(), test["date"].min())

    def test_monthly_aggregation(self) -> None:
        frame = pd.DataFrame(
            {"date": ["2024-01-01", "2024-01-20", "2024-02-01"], "price": [10, 20, 30]}
        )
        result = aggregate_monthly(frame, "date", "price")
        self.assertEqual(result["price"].tolist(), [15.0, 30.0])


class EndToEndTests(unittest.TestCase):
    def test_cli_joins_separate_sources_and_writes_forecasts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            generate_sample_data(data_dir, periods=36)
            output = root / "outputs"
            summary = execute(
                Namespace(
                    data=data_dir / "housing_prices.csv",
                    date_column="date",
                    target_column="price",
                    features="inflation_rate,mortgage_rate,synthetic_region_index",
                    macro_data=data_dir / "economic_indicators.csv",
                    area_lookup=data_dir / "area_lookup.csv",
                    forecast_csv=data_dir / "future_economic_inputs.csv",
                    output_dir=output,
                    train_fraction=0.8,
                )
            )
            self.assertEqual(summary["future_prediction_rows"], 30)
            self.assertTrue((output / "future_predictions.csv").is_file())
            self.assertTrue((output / "dashboard.html").is_file())

    def test_models_reports_database_and_forecasts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            generate_sample_data(data_dir, periods=36)
            data = pd.read_csv(data_dir / "housing.csv", parse_dates=["date"])
            features = ["inflation_rate", "mortgage_rate", "unemployment_rate"]
            prepared = prepare_data(data, "date", "price", features)
            results = evaluate_models(
                prepared, "date", "price", features, importance_repeats=2
            )
            output = root / "outputs"
            summary = save_results(results, "date", "price", features, output)

            self.assertEqual(len(results["comparison"]), 3)
            self.assertEqual(len(results["importance"]), len(features))
            self.assertEqual(summary["best_model_by_mae"], results["comparison"].iloc[0]["model"])
            self.assertTrue((output / "model_comparison.csv").is_file())
            self.assertTrue((output / "charts" / "holdout_predictions.svg").is_file())
            self.assertTrue((output / "dashboard.html").is_file())
            with closing(sqlite3.connect(output / "housing_analysis.sqlite")) as connection:
                count = connection.execute("SELECT COUNT(*) FROM model_metrics").fetchone()[0]
            self.assertEqual(count, 3)

            future = pd.read_csv(data_dir / "future_economic_inputs.csv")
            predicted = results["best_model"].predict(future[features])
            self.assertEqual(len(predicted), len(future))
            self.assertTrue(pd.Series(predicted).notna().all())


if __name__ == "__main__":
    unittest.main()
