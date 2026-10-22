"""Chronological baseline and machine-learning model comparison."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import (
    DATE_COLUMN,
    FORECAST_FEATURES,
    TARGET_COLUMN,
    chronological_split,
)

BASELINE_NAMES = ("Persistence (1-month naive)", "Seasonal naive (12-month)")
Model = LinearRegression | RandomForestRegressor | GradientBoostingRegressor


def build_models() -> dict[str, Model]:
    """Return a small, reproducible selection of regression models."""
    return {
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }


def _predictions(
    model_name: str, data: pd.DataFrame, model: Model | None = None
) -> Any:
    if model_name == BASELINE_NAMES[0]:
        return data["lag_1"].to_numpy()
    if model_name == BASELINE_NAMES[1]:
        return data["lag_12"].to_numpy()
    if model is None:
        raise ValueError(f"No fitted model supplied for {model_name}.")
    return model.predict(data[FORECAST_FEATURES])


def _metrics(
    model_name: str, actual: pd.Series, predicted: Any
) -> dict[str, float | str | None]:
    return {
        "model": model_name,
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def evaluate_models(
    data: pd.DataFrame,
    date_column: str = DATE_COLUMN,
    target_column: str = TARGET_COLUMN,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    importance_repeats: int = 5,
) -> dict[str, Any]:
    """Select by validation MAE, then score every approach on the later test set."""
    if importance_repeats < 1:
        raise ValueError("importance_repeats must be at least 1.")
    missing_features = set(FORECAST_FEATURES + ["lag_1", "lag_12"]).difference(data.columns)
    if missing_features:
        raise ValueError(
            "Forecast data is missing required features: "
            f"{', '.join(sorted(missing_features))}"
        )
    if target_column not in data.columns or date_column not in data.columns:
        raise ValueError(
            f"Forecast data must contain '{date_column}' and '{target_column}'."
        )
    train, validation, test = chronological_split(
        data, date_column, train_fraction, validation_fraction
    )
    model_builders = build_models()
    validation_scores: list[dict[str, float | str | None]] = []
    trained_on_train: dict[str, Model] = {}
    x_train = train[FORECAST_FEATURES]
    y_train = train[target_column]
    x_validation = validation[FORECAST_FEATURES]
    y_validation = validation[target_column]

    for name, model in model_builders.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_validation)
        trained_on_train[name] = model
        validation_scores.append(_metrics(name, y_validation, predicted))

    for name in BASELINE_NAMES:
        predicted = _predictions(name, validation)
        validation_scores.append(_metrics(name, y_validation, predicted))

    validation_comparison = (
        pd.DataFrame(validation_scores).sort_values("mae").reset_index(drop=True)
    )
    best_model_name = str(validation_comparison.loc[0, "model"])

    holdout_columns = [
        column
        for column in ("RegionName", "StateName")
        if column in test.columns
    ]
    test_predictions = test[[date_column] + holdout_columns].copy()
    test_predictions[target_column] = test[target_column].to_numpy()
    test_scores: list[dict[str, float | str | None]] = []
    refit_data = pd.concat([train, validation]).sort_values(date_column)
    refit_models: dict[str, Model] = {}
    for name, original_model in model_builders.items():
        model = clone(original_model)
        model.fit(refit_data[FORECAST_FEATURES], refit_data[target_column])
        predicted = model.predict(test[FORECAST_FEATURES])
        test_predictions[f"{name} prediction"] = predicted
        test_scores.append(_metrics(name, test[target_column], predicted))
        refit_models[name] = model

    for name in BASELINE_NAMES:
        predicted = _predictions(name, test)
        test_predictions[f"{name} prediction"] = predicted
        test_scores.append(_metrics(name, test[target_column], predicted))

    comparison = pd.DataFrame(test_scores).sort_values("mae").reset_index(drop=True)
    if best_model_name in trained_on_train:
        validation_model = trained_on_train[best_model_name]
        permutation = permutation_importance(
            validation_model,
            x_validation,
            y_validation,
            scoring="neg_mean_absolute_error",
            n_repeats=importance_repeats,
            random_state=42,
            n_jobs=-1,
        )
        importance = pd.DataFrame(
            {
                "factor": FORECAST_FEATURES,
                "importance_mean": permutation["importances_mean"],
                "importance_std": permutation["importances_std"],
            }
        ).sort_values("importance_mean", ascending=False, ignore_index=True)
    else:
        importance = pd.DataFrame(
            columns=["factor", "importance_mean", "importance_std"]
        )

    return {
        "comparison": comparison,
        "validation_comparison": validation_comparison,
        "holdout": test_predictions,
        "importance": importance,
        "best_model_name": best_model_name,
        "best_model": refit_models.get(best_model_name),
        "train": train,
        "validation": validation,
        "test": test,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
    }
