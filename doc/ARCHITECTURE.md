# Architecture

## Components

```text
CSV inputs -> cli.execute -> preprocessing.prepare_data
                               |
                               v
                      models.evaluate_models
                               |
               +---------------+----------------+
               |               |                |
               v               v                v
          CSV / JSON       SVG charts       SQLite database
               |
               +---- optional future-input CSV -> future predictions

launch_gui.py -> gui.HousingApp -> cli.execute
```

## Package modules

- `src/housing_market_intelligence/cli.py` parses options, reads CSV inputs,
  orchestrates preparation and evaluation, writes future predictions, and
  prints the result summary.
- `preprocessing.py` validates and cleans housing rows, optionally joins macro
  observations by date and market metadata by `area_id`, creates chronological
  train/holdout partitions, and offers monthly aggregation.
- `models.py` defines the linear regression, random forest, and gradient
  boosting pipelines; calculates holdout metrics; ranks
  permutation importance; and refits the selected model on the full historical
  dataset for optional future scoring.
- `reports.py` writes tabular and JSON results, constructs standalone SVG
  charts and an HTML dashboard without an image dependency, and stores
  comparison, prediction, and importance tables in a local SQLite database.
- `sample_data.py` generates deterministic fictional housing, economic,
  market-lookup, and future-input CSV files.
- `gui.py` is an optional Tkinter interface. It invokes the same analysis
  functions as the CLI rather than maintaining a separate modeling path.
- `tests/test_workflow.py` checks data joins and validation, monthly
  aggregation, time-split isolation, the three-model workflow, generated
  reports, SQLite persistence, and future scoring.

## Model lifecycle

1. Parse the user-selected predictor list.
2. Load historical observations and any optional separate macro or market
   lookup tables.
3. Validate, clean, and join rows. Missing predictor values are imputed inside
   each scikit-learn model pipeline, using statistics learned from training.
4. Split distinct dates so every row for a date is assigned to only one side.
5. Fit the three pipelines on earlier dates and score predictions on later
   holdout dates.
6. Select the lowest-MAE model, compute holdout permutation importance, and
   refit that model on all available historical observations.
7. Persist the comparison results and, if future inputs were provided, score
   those dates using their supplied predictors.

## Input/output contracts

Historical data contains a date column, numeric target, and selected features.
Optional economic data contains the same date column. Optional market metadata
contains `area_id`, which is also required in historical housing rows. Future
inputs contain the date and every selected feature.

Outputs are written under the chosen output folder. SQLite tables are
`model_metrics`, `holdout_predictions`, and `factor_importance`. SVGs are
portable vector images that browsers can render directly.
