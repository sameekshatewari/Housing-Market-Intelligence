# Architecture

## Workflow

```text
data/zhvi_metro.csv
        |
        v
cli.execute -> preprocessing.load_zhvi_csv -> tidy metro-month rows
                                             |
                                             v
                                  preprocessing.prepare_data
                                  (past-only lag features)
                                             |
                                             v
                                      models.evaluate_models
                                 train -> validation -> test
                                             |
                 +---------------------------+--------------------------+
                 |                           |                          |
           reports / CSV                SVG + HTML                 SQLite
```

`housing_market_intelligence.py` launches the CLI. `launch_gui.py` starts the
optional Tkinter interface, which calls the same CLI workflow. Both require a
locally downloaded Zillow CSV; neither generates replacement housing data.

## Package modules

- `cli.py` accepts the local CSV and output directory, coordinates loading,
  feature engineering, evaluation, reporting, and the run summary.
- `preprocessing.py` validates Zillow's wide format, filters
  `RegionType=msa`/`metro`, melts monthly columns into `RegionName`,
  `StateName`, `Date`, and `HomeValue`, and creates past-only lag/calendar
  features.
- `models.py` evaluates persistence and seasonal-naive baselines plus linear
  regression, random forest, and gradient boosting using chronological
  train/validation/test partitions.
- `reports.py` writes validation and test tables, forecast CSVs, SVG charts,
  an HTML dashboard, a JSON run summary, and a local SQLite database.
- `gui.py` is an optional desktop front end to the same command workflow.
- `tests/test_workflow.py` verifies the Zillow CSV contract, missing data,
  lag features, split isolation, model evaluation, reports, and SQLite output
  using small in-memory fixtures.

## Modeling lifecycle

1. Read the user-provided Zillow CSV. Require location and region-type
   columns, identify monthly date columns, and keep metro rows only.
2. Reshape the dates to one row per metro and month; discard missing target
   observations rather than inventing or filling values.
3. Add lags and calendar features using earlier observations within the same
   metro. Require 13 months of valid history for the prior-year growth feature.
4. Split distinct months chronologically. All metros for a month remain in
   the same partition.
5. Fit learned models on training rows and select the lowest validation-MAE
   candidate. Naive baselines use the lag values directly.
6. Refit each learned candidate on training plus validation observations and
   report its performance on the untouched later test period. Report MAE,
   RMSE, and R-squared for baselines and learned models alike.
7. Write all model scores, predictions, dataset information, and date ranges
   to the CSV, JSON, dashboard, and SQLite outputs.

## Important interpretation details

Forecasting is rolling one month ahead: at a test month, values from earlier
months (including previously observed test-period months) may be used as
lags. This represents a monthly update as new observations arrive; it is not a
fixed multi-month forecast made at the start of the test period.

The chart's metro mean is unweighted. Pooled R-squared is relative to the
constant mean of scored rows; it is not the seasonal-naive comparator. A
permutation importance score describes predictive association on validation
data and does not establish causation.

## Output and database contracts

`outputs` contains `validation_comparison.csv`, `model_comparison.csv`,
`holdout_predictions.csv`, `factor_importance.csv`, `summary.json`,
standalone SVG charts, `dashboard.html`, and `housing_analysis.sqlite`.
SQLite tables are `model_metrics`, `validation_metrics`,
`holdout_predictions`, `factor_importance`, and `run_metadata`.

The CSV input itself is not committed. Download instructions and data-source
details are in the project [README](../README.md).
