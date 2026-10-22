# Understanding the Program

## What the project analyzes

The program compares forecasts of Zillow's metro-level ZHVI: middle-tier,
all-homes, seasonally adjusted monthly values. ZHVI is a modeled estimate of
typical home value, not an individual-home sale-price dataset.

Download the CSV from the
[Zillow Research Data page](https://www.zillow.com/research/data/) and save it
as `data\zhvi_metro.csv`. The root [README](../README.md) lists the exact
download settings and run commands. No data file is bundled; the program stops
if the local file is missing or does not have the expected Zillow format.

## Cleaning and features

The source CSV is wide: location fields identify each metro, and each
`YYYY-MM-DD` column contains that month's ZHVI. The loader keeps rows marked
`RegionType=msa` or `metro` and melts available date columns into a tidy table
with `RegionName`, `StateName`, `Date`, and `HomeValue`.

Blank or non-numeric monthly values are treated as missing and removed. They
are not filled with fabricated values. Forecast features are made separately
inside each metro: values from 1, 3, 6, and 12 months earlier; year-over-year
growth calculated from those past values; and seasonal month indicators.
Rows without all required history are omitted.

## Forecasting models and fair comparison

- **Persistence** predicts the previous month's value.
- **Seasonal naive** predicts the value from 12 months earlier.
- **Linear regression** provides a compact linear model.
- **Random forest** averages decision trees.
- **Gradient boosting** combines shallow trees fitted in sequence.

The default split uses the earliest 70% of distinct months for training, the
next 15% for validation, and the final 15% for testing. All markets from a
month stay together. Lowest validation MAE selects the model. Every approach
is then scored on the same final test dates, which are not used for selection.

Evaluation is rolling one month ahead: once a month has been observed, it can
contribute to a later month's lag. This does not measure a fixed forecast of
many months made all at once. The model is evaluated across pooled
metro-months, so performance can differ across markets and periods.

The metrics mean:

- **MAE:** mean absolute error in ZHVI value units.
- **RMSE:** square root of mean squared error; large errors contribute more.
- **R-squared:** improvement relative to a constant mean-value prediction. It
  can be negative and is not the seasonal-naive forecast.

## Charts and explanation

The dashboard includes validation and final test tables, predicted-vs-actual
values, a forecast time chart, and (when a learned model wins validation)
permutation importance. The time chart shows an unweighted mean over metro
values, not a population-weighted national index. Permutation importance
measures whether shuffling a feature changes prediction error; it does not
show that the feature caused a housing-value change.

## Run and test

```powershell
python .\housing_market_intelligence.py
python -m unittest discover -s tests -v
```

The test suite uses small in-memory examples shaped like the Zillow file; real
data is optional for running tests but necessary for real-market outputs.
