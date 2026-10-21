# Understanding the Program

## What this project does

The program prepares housing-price observations and economic predictors,
compares three regression approaches on later-in-time observations, describes
predictor usefulness, and can apply the winning model to later rows with known
economic inputs. Sample data included in this repository is generated and
fictional; it is intended to make the example runnable without external files.

## Preparation and temporal evaluation

Housing and optional macroeconomic tables join by date. Optional market
metadata joins by `area_id`. Rows without a usable date or target are dropped;
predictor columns are converted to numeric values, and missing predictor values
are imputed as part of each model pipeline.

The default split uses the earliest 80% of distinct dates to train and the
latest 20% to hold out. All rows with the same date stay on one side. This
better represents forecasting later periods than a random split, though a
single holdout interval is not a complete evaluation across regions or market
cycles.

## Compared models

- **Linear regression:** an interpretable linear baseline.
- **Random forest:** an ensemble of decision trees that can model nonlinear
  relationships.
- **Gradient boosting:** a sequence of shallow decision trees that each
  correct errors from the current ensemble.

The table reports:

- **MAE:** average absolute price error, in the target's units.
- **RMSE:** square-rooted average squared error; larger misses are penalized
  more strongly.
- **R-squared:** performance relative to predicting the holdout average. It
  may be negative if a model performs worse than that baseline.

The lowest holdout MAE selects the model for subsequent predictions. Because
the same holdout is used for model selection and interpretation, compare
results as exploratory evidence rather than an untouched final benchmark.

## Factor importance and causal limits

Permutation importance measures how the selected model's holdout MAE changes
when one input feature is shuffled. A positive score means the shuffle
increased error; a negative score means it decreased error for these data.
Importance is about predictive association in this particular sample, not
causal influence. Correlated economic variables can share or distort
importance.

## Future predictions

When `--forecast-csv` is supplied, the chosen model is refit using all
historical data and predicts the rows in that file. Every selected predictor
must already have a value for each future date. The program does not generate
future inflation, mortgage, fuel, or unemployment assumptions; users must
provide them.

## Try it

```powershell
python .\housing_market_intelligence.py --data .\data\housing.csv --date-column date --target-column price --features inflation_rate,mortgage_rate,unemployment_rate,fuel_price,interest_rate,synthetic_region_index --forecast-csv .\data\future_economic_inputs.csv
```

To use the desktop interface instead:

```powershell
python .\launch_gui.py
```

To validate the workflow with automated tests:

```powershell
python -m unittest discover -s tests -v
```
