"""Time-aware model comparison and feature interpretation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

from .preprocessing import chronological_split


def build_models() -> dict[str, object]:
    """Create the independently implemented model comparison set."""
    return {
        "Linear regression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            LinearRegression(),
        ),
        "Random forest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        ),
        "Gradient boosting": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            ),
        ),
    }


def evaluate_models(
    data: pd.DataFrame,
    date_column: str,
    target_column: str,
    features: list[str],
    train_fraction: float = 0.8,
    importance_repeats: int = 5,
) -> dict[str, Any]:
    """Fit, score, interpret and finally refit models on all historical rows."""
    train, test = chronological_split(data, date_column, train_fraction)
    x_train, y_train = train[features], train[target_column]
    x_test, y_test = test[features], test[target_column]
    scores: list[dict[str, float | str | None]] = []
    context_columns = [
        column for column in ("area_id", "market_name")
        if column in test.columns and column not in features
    ]
    holdout = test[[date_column] + context_columns].copy()
    holdout[target_column] = y_test.to_numpy()
    fitted: dict[str, object] = {}

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        holdout[f"{name} prediction"] = predicted
        scores.append(
            {
                "model": name,
                "mae": float(mean_absolute_error(y_test, predicted)),
                "rmse": float(mean_squared_error(y_test, predicted) ** 0.5),
                "r2": float(r2_score(y_test, predicted)) if len(y_test) > 1 else None,
            }
        )
        fitted[name] = model

    comparison = pd.DataFrame(scores).sort_values("mae").reset_index(drop=True)
    winner_name = str(comparison.loc[0, "model"])
    winner = fitted[winner_name]
    permutation = permutation_importance(
        winner,
        x_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=importance_repeats,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "factor": features,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
    winner.fit(data[features], data[target_column])
    return {
        "comparison": comparison,
        "holdout": holdout,
        "importance": importance,
        "best_model": winner,
        "best_model_name": winner_name,
        "train": train,
        "test": test,
    }
