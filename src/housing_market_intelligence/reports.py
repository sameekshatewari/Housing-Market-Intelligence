"""Portable result exports, SVG charts and a local SQLite analysis database."""

from __future__ import annotations

import html
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd


def _svg_document(title: str, body: str, width: int = 900, height: int = 520) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#f8fafc"/>'
        f'<text x="40" y="42" font-family="Arial" font-size="24" font-weight="bold" '
        f'fill="#0f172a">{html.escape(title)}</text>{body}</svg>'
    )


def make_charts(
    comparison: pd.DataFrame, holdout: pd.DataFrame, importance: pd.DataFrame,
    target_column: str, output_dir: Path, price_data: pd.Series | None = None,
) -> list[Path]:
    """Write self-contained SVG charts without a plotting-library dependency."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    max_error = max(float(comparison[["mae", "rmse"]].max().max()), 1.0)
    body = ""
    for index, row in comparison.reset_index(drop=True).iterrows():
        y = 105 + index * 110
        body += (
            f'<text x="40" y="{y + 28}" font-family="Arial" font-size="16" fill="#334155">'
            f'{html.escape(str(row["model"]))}</text>'
            f'<rect x="250" y="{y}" width="{520 * float(row["mae"]) / max_error:.1f}" height="32" rx="5" fill="#2563eb"/>'
            f'<rect x="250" y="{y + 39}" width="{520 * float(row["rmse"]) / max_error:.1f}" height="32" rx="5" fill="#f97316"/>'
            f'<text x="780" y="{y + 22}" font-family="Arial" font-size="13" fill="#0f172a">{float(row["mae"]):,.0f}</text>'
            f'<text x="780" y="{y + 61}" font-family="Arial" font-size="13" fill="#0f172a">{float(row["rmse"]):,.0f}</text>'
        )
    body += (
        '<rect x="300" y="455" width="18" height="18" fill="#2563eb"/>'
        '<text x="326" y="470" font-family="Arial" font-size="14" fill="#334155">MAE (lower is better)</text>'
        '<rect x="535" y="455" width="18" height="18" fill="#f97316"/>'
        '<text x="561" y="470" font-family="Arial" font-size="14" fill="#334155">RMSE (lower is better)</text>'
        '<text x="250" y="500" font-family="Arial" font-size="12" fill="#64748b">Error in price units</text>'
    )
    path = output_dir / "model_comparison.svg"
    path.write_text(_svg_document("Holdout model error comparison", body), encoding="utf-8")
    created.append(path)

    date_column = holdout.columns[0]
    model_columns = [column for column in holdout.columns if column.endswith(" prediction")]
    rows = holdout.sort_values(date_column).reset_index(drop=True)
    values = rows[[target_column] + model_columns].astype(float)
    low, high = float(values.min().min()), float(values.max().max())
    span = max(high - low, 1.0)
    plot_x, plot_y, plot_w, plot_h = 70, 85, 780, 330
    body = (
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" '
        f'y2="{plot_y + plot_h}" stroke="#64748b"/>'
    )
    series = [(target_column, "#0f172a")] + [
        (column, color) for column, color in zip(model_columns, ["#2563eb", "#16a34a", "#ea580c"])
    ]
    for label, color in series:
        points = []
        for index, value in enumerate(values[label]):
            x = plot_x + (plot_w * index / max(len(values) - 1, 1))
            y = plot_y + plot_h - (plot_h * (float(value) - low) / span)
            points.append(f"{x:.1f},{y:.1f}")
        body += f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>'
    for index, (label, color) in enumerate(series):
        x = 80 + index * 245
        body += (
            f'<line x1="{x}" y1="460" x2="{x + 28}" y2="460" stroke="{color}" stroke-width="4"/>'
            f'<text x="{x + 36}" y="465" font-family="Arial" font-size="13" fill="#334155">'
            f'{html.escape(label)}</text>'
        )
    path = output_dir / "holdout_predictions.svg"
    path.write_text(_svg_document("Observed and predicted holdout prices", body), encoding="utf-8")
    created.append(path)

    target_values = (
        price_data.astype(float) if price_data is not None
        else holdout[target_column].astype(float)
    )
    low, high = float(target_values.min()), float(target_values.max())
    span = max(high - low, 1.0)
    bins = 16
    counts = [0] * bins
    for value in target_values:
        index = min(int((float(value) - low) / span * bins), bins - 1)
        counts[index] += 1
    peak = max(counts + [1])
    plot_x, plot_y, plot_w, plot_h = 100, 95, 740, 330
    body = (
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" '
        f'y2="{plot_y + plot_h}" stroke="#64748b"/>'
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#64748b"/>'
    )
    slot = plot_w / bins
    for index, count in enumerate(counts):
        height = plot_h * count / peak
        x = plot_x + index * slot + 2
        y = plot_y + plot_h - height
        body += (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot - 4:.1f}" height="{height:.1f}" '
            'fill="#0ea5e9" stroke="#f8fafc"/>'
        )
    for fraction, anchor in ((0, "start"), (0.5, "middle"), (1, "end")):
        value = low + span * fraction
        x = plot_x + plot_w * fraction
        body += (
            f'<text x="{x:.1f}" y="455" text-anchor="{anchor}" font-family="Arial" '
            f'font-size="13" fill="#334155">{value:,.0f}</text>'
        )
    body += (
        f'<text x="28" y="{plot_y + 10}" font-family="Arial" font-size="12" fill="#334155">{peak}</text>'
        f'<text x="28" y="{plot_y + plot_h}" font-family="Arial" font-size="12" fill="#334155">0</text>'
        f'<text x="{plot_x + plot_w / 2:.1f}" y="490" text-anchor="middle" font-family="Arial" '
        f'font-size="14" fill="#334155">{html.escape(target_column)}</text>'
        f'<text x="35" y="260" transform="rotate(-90 35 260)" font-family="Arial" '
        'font-size="14" fill="#334155">Observations</text>'
    )
    path = output_dir / "price_distribution.svg"
    path.write_text(_svg_document("Housing price distribution (all observations)", body), encoding="utf-8")
    created.append(path)

    best_model_name = str(comparison.iloc[0]["model"])
    predicted_column = f"{best_model_name} prediction"
    actual = holdout[target_column].astype(float)
    predicted = holdout[predicted_column].astype(float)
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))
    span = max(high - low, 1.0)
    plot_x, plot_y, plot_w, plot_h = 245, 80, 390, 390
    body = (
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" '
        f'y2="{plot_y + plot_h}" stroke="#64748b"/>'
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#64748b"/>'
    )
    equal_start = plot_y + plot_h - plot_h * (low - low) / span
    equal_end = plot_y + plot_h - plot_h * (high - low) / span
    body += (
        f'<line x1="{plot_x}" y1="{equal_start:.1f}" x2="{plot_x + plot_w}" '
        f'y2="{equal_end:.1f}" stroke="#64748b" stroke-dasharray="8 6" stroke-width="2"/>'
    )
    for actual_value, predicted_value in zip(actual, predicted):
        x = plot_x + plot_w * (float(actual_value) - low) / span
        y = plot_y + plot_h - plot_h * (float(predicted_value) - low) / span
        body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb" fill-opacity=".7"/>'
    for fraction, anchor in ((0, "start"), (0.5, "middle"), (1, "end")):
        value = low + span * fraction
        x = plot_x + plot_w * fraction
        y = plot_y + plot_h + 18
        y_tick = plot_y + plot_h - plot_h * fraction
        body += (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial" '
            f'font-size="12" fill="#334155">{value:,.0f}</text>'
            f'<text x="{plot_x - 10}" y="{y_tick + 4:.1f}" text-anchor="end" '
            f'font-family="Arial" font-size="12" fill="#334155">{value:,.0f}</text>'
        )
    body += (
        f'<text x="{plot_x + plot_w / 2:.1f}" y="515" text-anchor="middle" font-family="Arial" '
        'font-size="14" fill="#334155">Actual price</text>'
        f'<text x="32" y="260" transform="rotate(-90 32 260)" font-family="Arial" '
        'font-size="14" fill="#334155">Predicted price</text>'
        f'<text x="120" y="65" font-family="Arial" font-size="13" fill="#334155">'
        f'Best by MAE: {html.escape(best_model_name)} (dashed line = perfect prediction)</text>'
    )
    path = output_dir / "predicted_vs_actual.svg"
    path.write_text(
        _svg_document("Predicted versus actual holdout prices", body, height=550),
        encoding="utf-8",
    )
    created.append(path)

    factors = importance.sort_values("importance_mean", ascending=True).reset_index(drop=True)
    max_abs = max(float(factors["importance_mean"].abs().max()), 1.0)
    body = ""
    for index, row in factors.iterrows():
        y = 100 + index * 55
        x0, center, full = 280, 280, 480
        amount = full * abs(float(row["importance_mean"])) / max_abs
        if row["importance_mean"] >= 0:
            x, color = center, "#16a34a"
        else:
            x, color = center - amount, "#dc2626"
        body += (
            f'<text x="40" y="{y + 22}" font-family="Arial" font-size="14" fill="#334155">'
            f'{html.escape(str(row["factor"]))}</text>'
            f'<rect x="{x:.1f}" y="{y}" width="{amount:.1f}" height="30" fill="{color}"/>'
            f'<text x="{max(x + amount + 8, center + 8):.1f}" y="{y + 21}" '
            f'font-family="Arial" font-size="13" fill="#0f172a">'
            f'{float(row["importance_mean"]):.2f}</text>'
        )
    body += f'<line x1="{center}" y1="75" x2="{center}" y2="490" stroke="#64748b"/>'
    path = output_dir / "factor_importance.svg"
    path.write_text(_svg_document("Holdout permutation importance", body), encoding="utf-8")
    created.append(path)
    return created


def make_dashboard(
    comparison: pd.DataFrame, holdout: pd.DataFrame, importance: pd.DataFrame,
    output_dir: Path, best_model_name: str,
) -> Path:
    """Create a portable HTML dashboard containing the generated charts and score tables."""
    chart_links = [
        ("Model errors", "model_comparison.svg"),
        ("Holdout predictions", "holdout_predictions.svg"),
        ("Price distribution", "price_distribution.svg"),
        ("Predicted versus actual", "predicted_vs_actual.svg"),
        ("Factor importance", "factor_importance.svg"),
    ]
    figures = "".join(
        f'<section><h2>{html.escape(title)}</h2><img src="charts/{filename}" '
        f'alt="{html.escape(title)} chart"></section>'
        for title, filename in chart_links
    )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Housing Market Intelligence dashboard</title>
<style>
body{{font-family:Arial,sans-serif;margin:2rem auto;padding:0 1rem;max-width:1100px;color:#0f172a;background:#f8fafc}}
section{{background:white;padding:1rem 1.4rem;margin:1rem 0;border-radius:10px;box-shadow:0 1px 5px #cbd5e1}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;border-bottom:1px solid #e2e8f0;text-align:left}}
.winner{{font-size:1.2rem;color:#1d4ed8;font-weight:bold}}
</style></head><body>
<h1>Housing Market Intelligence</h1>
<p class="winner">Selected model by holdout MAE: {html.escape(best_model_name)}</p>
<section><h2>Model metrics</h2>{comparison.to_html(index=False, border=0, classes="metrics")}</section>
{figures}
<section><h2>Holdout predictions</h2>{holdout.to_html(index=False, border=0, classes="predictions")}</section>
<section><h2>Feature importance</h2>{importance.to_html(index=False, border=0, classes="importance")}</section>
<p>Feature importance is predictive association, not proof of causation. Results are exploratory and depend on input data quality.</p>
</body></html>"""
    path = output_dir / "dashboard.html"
    path.write_text(content, encoding="utf-8")
    return path


def save_results(
    results: dict[str, Any], date_column: str, target_column: str,
    features: list[str], output_dir: Path,
) -> dict[str, Any]:
    """Persist tabular model results, charts, a JSON summary and a SQLite copy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results["comparison"].to_csv(output_dir / "model_comparison.csv", index=False)
    results["holdout"].to_csv(output_dir / "holdout_predictions.csv", index=False)
    results["importance"].to_csv(output_dir / "factor_importance.csv", index=False)
    summary = {
        "date_column": date_column,
        "target_column": target_column,
        "features": features,
        "best_model_by_mae": results["best_model_name"],
        "training_rows": int(len(results["train"])),
        "holdout_rows": int(len(results["test"])),
        "training_start": str(results["train"][date_column].min().date()),
        "training_end": str(results["train"][date_column].max().date()),
        "holdout_start": str(results["holdout"][date_column].min().date()),
        "holdout_end": str(results["holdout"][date_column].max().date()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    all_prices = pd.concat(
        [results["train"][target_column], results["test"][target_column]],
        ignore_index=True,
    )
    make_charts(
        results["comparison"], results["holdout"], results["importance"],
        target_column, output_dir / "charts", price_data=all_prices,
    )
    make_dashboard(results["comparison"], results["holdout"], results["importance"],
                   output_dir, results["best_model_name"])
    with closing(sqlite3.connect(output_dir / "housing_analysis.sqlite")) as connection:
        with connection:
            results["comparison"].to_sql("model_metrics", connection, if_exists="replace", index=False)
            results["holdout"].to_sql("holdout_predictions", connection, if_exists="replace", index=False)
            results["importance"].to_sql("factor_importance", connection, if_exists="replace", index=False)
    return summary
