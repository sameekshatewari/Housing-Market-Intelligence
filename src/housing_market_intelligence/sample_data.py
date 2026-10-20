"""Generate a reproducible, fictional multi-market demonstration dataset."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def generate_sample_data(output_dir: Path, periods: int = 72) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2018-01-01", periods=periods, freq="MS")
    areas = [
        ("MARKET-A", "North Example", 0),
        ("MARKET-B", "Coastal Example", 1),
        ("MARKET-C", "Central Example", 2),
        ("MARKET-D", "Lake Example", 3),
        ("MARKET-E", "Valley Example", 4),
    ]
    area_lookup = pd.DataFrame(
        [
            {
                "area_id": area_id,
                "market_name": name,
                "synthetic_region_index": index,
            }
            for area_id, name, index in areas
        ]
    )
    macro_rows = []
    housing_rows = []
    for month, date in enumerate(dates):
        inflation = 2.4 + 0.9 * math.sin(month / 10) + (1.6 if 37 <= month <= 48 else 0)
        mortgage = 4.0 + 1.4 * math.sin(month / 13 + 0.7) + (1.3 if 48 <= month <= 64 else 0)
        unemployment = 5.0 - 0.022 * month + 0.35 * math.sin(month / 8)
        fuel = 3.1 + 0.45 * math.sin(month / 7 + 2)
        interest = 1.5 + 1.1 * math.sin(month / 15 + 1)
        macro_rows.append(
            {
                "date": date,
                "inflation_rate": round(inflation, 3),
                "mortgage_rate": round(mortgage, 3),
                "unemployment_rate": round(unemployment, 3),
                "fuel_price": round(fuel, 3),
                "interest_rate": round(interest, 3),
            }
        )
        for area_id, _name, area_index in areas:
            baseline = 215_000 + area_index * 37_000
            trend = month * (450 + 35 * area_index)
            economic = -2_100 * mortgage + 900 * inflation - 1_250 * unemployment
            cycle = 4_500 * math.sin(month / 9 + area_index / 2)
            price = baseline + trend + economic + cycle
            housing_rows.append(
                {
                    "date": date,
                    "area_id": area_id,
                    "price": round(price, 2),
                }
            )
    macro = pd.DataFrame(macro_rows)
    housing = pd.DataFrame(housing_rows)
    merged = housing.merge(macro, on="date", validate="many_to_one").merge(
        area_lookup, on="area_id", validate="many_to_one"
    )
    housing.to_csv(output_dir / "housing_prices.csv", index=False)
    macro.to_csv(output_dir / "economic_indicators.csv", index=False)
    area_lookup.to_csv(output_dir / "area_lookup.csv", index=False)
    merged.to_csv(output_dir / "housing.csv", index=False)

    future_dates = pd.date_range(dates[-1] + pd.offsets.MonthBegin(), periods=6, freq="MS")
    future_rows = []
    for offset, date in enumerate(future_dates, start=periods):
        for area_id, _name, _area_index in areas:
            future_rows.append(
                {
                    "date": date,
                    "area_id": area_id,
                    "inflation_rate": round(2.4 + 0.9 * math.sin(offset / 10), 3),
                    "mortgage_rate": round(4.0 + 1.4 * math.sin(offset / 13 + 0.7), 3),
                    "unemployment_rate": round(5.0 - 0.022 * offset + 0.35 * math.sin(offset / 8), 3),
                    "fuel_price": round(3.1 + 0.45 * math.sin(offset / 7 + 2), 3),
                    "interest_rate": round(1.5 + 1.1 * math.sin(offset / 15 + 1), 3),
                }
            )
    future = pd.DataFrame(future_rows).merge(area_lookup, on="area_id", validate="many_to_one")
    future.to_csv(output_dir / "future_economic_inputs.csv", index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    generate_sample_data(project_root / "data")
    print(f"Generated fictional sample datasets in {project_root / 'data'}")


if __name__ == "__main__":
    main()
