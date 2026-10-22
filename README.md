# Housing Market Intelligence

**Explore US metro housing-value trends and compare one-month-ahead forecasts.**

This portfolio project uses Zillow Research's monthly Zillow Home Value Index
(ZHVI): **middle tier, all homes, seasonally adjusted, metro geography**. The
pipeline reshapes Zillow's wide CSV into one row per metro and month, builds
past-only forecasting features, and compares simple forecasting baselines
with machine-learning regressors. The validation period selects a model; a
later, separate test period reports its final performance.

## Data source and download

Download the series from the official
[Zillow Research Data page](https://www.zillow.com/research/data/). Select the
ZHVI data with the following options:

- Geography: **Metro**
- Home type: **All homes**
- Price tier: **Middle tier**
- Frequency: **Monthly**
- Seasonal adjustment: **Seasonally adjusted**

Save the downloaded CSV in the repository's `data` folder as:

```text
data\zhvi_metro.csv
```

The loader expects Zillow's wide format: `RegionName`, `RegionType`,
`StateName`, followed by date columns named `YYYY-MM-DD`. It retains metro rows
(`RegionType=msa` or `metro`), pivots monthly columns into `RegionName`,
`StateName`, `Date`, and `HomeValue`, and drops rows with blank/non-numeric
values. It does not download data or generate replacement observations. A
missing or incompatible CSV stops the run with an error.

The live official file was inspected on **October 21, 2026**. Its date columns
ran from **2000-01-31 through 2026-09-30** at that time. Coverage changes as
Zillow updates its download; every run records the actual input filename,
loaded observation dates, and metro count in `outputs\summary.json` and the
dashboard. The official download linked by Zillow is
[the metro monthly middle-tier/all-homes/seasonally-adjusted CSV](https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv).

### Data limitations

ZHVI is Zillow's modeled estimate of typical home value, not a record of
individual sales, an appraisal, or a guaranteed transaction price. Metro
observations are estimates and may be revised. The metro geography excludes
areas outside metropolitan statistical areas, and this project treats metro
values equally; it does not weight them by population, housing stock, or
transactions. This is a forecasting and descriptive project, not a causal
analysis.

## Modeling and evaluation

Forecasts are **rolling one-month-ahead** predictions for each metro. Features
use only values already observed at the forecast date: 1-, 3-, 6-, and
12-month lags, prior-year growth, and month-of-year sine/cosine terms. Rows
without the historical observations required for those features are omitted;
missing target values are never filled in.

Models and baselines:

- **Persistence:** use the previous month's ZHVI (`lag_1`).
- **Seasonal naive:** use the value 12 months earlier (`lag_12`).
- **Linear regression**
- **Random forest**
- **Gradient boosting**

Distinct dates are split chronologically into training (70%), validation
(15%), and final test (15%) periods by default. The model with the lowest
validation MAE is selected. Every candidate is then evaluated on the same later
test period; the test scores are not used to select a model. MAE and RMSE are
reported in ZHVI value units. R-squared compares predictions with a constant
mean-value reference, may be negative, and is **not** the seasonal-naive
forecasting baseline. A validation-only report also ranks metro markets by
the selected model's MAE and compares it with both naive baselines.

The metro report includes **WAPE** (weighted absolute percentage error):
`100 × sum(|actual − predicted|) / sum(|actual|)`. It summarizes absolute
error relative to total observed value, avoiding unstable per-row percentages
when an individual observation is close to zero. WAPE is left blank if a
market's total absolute actual value is at most one target unit. WAPE is still
scale-weighted across months within a market; it does not show the distribution
of monthly percentage errors and should be read with MAE/RMSE. Because pooled
scores combine metros and months, they should be interpreted alongside the
geographic coverage and this evaluation design.

The dashboard's dated forecast chart shows the **unweighted mean across
metros** for each month, not a population-weighted national index. Feature
permutation importance (when a learned model wins validation) is predictive
association, not evidence of causation.

### Verified results snapshot

I ran the pipeline on the official CSV retrieved on October 21, 2026 (894
metros; source observations January 2000–September 2026). Linear regression
was selected using validation MAE (**$744.52**) and evaluated on the separate
test period **December 2022–September 2026**. Test scores from that download:

| Model | Test MAE | Test RMSE | Test R-squared |
|---|---:|---:|---:|
| Linear regression (selected on validation) | $620.97 | $990.28 | 0.999967 |
| Random forest | $977.34 | $2,935.08 | 0.999707 |
| Persistence (1-month naive) | $1,088.83 | $1,610.81 | 0.999912 |
| Gradient boosting | $2,492.00 | $9,423.32 | 0.996981 |
| Seasonal naive (12-month) | $10,336.47 | $14,633.84 | 0.992720 |

R-squared is pooled across metro-month observations, so it is strongly
influenced by differences in metro value levels; it should not be interpreted
as evidence that predictions are equally accurate in every market. Zillow can
revise and extend the source series, so later runs may produce different
dates, splits, model selection, and scores. No downloaded data or generated
output files are included in the repository.

## Install and run (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

After saving the Zillow CSV to `data\zhvi_metro.csv`, run:

```powershell
python .\housing_market_intelligence.py
```

To supply a different local CSV or output folder:

```powershell
python .\housing_market_intelligence.py --data .\data\zhvi_metro.csv --output-dir .\outputs
```

The optional desktop interface uses the same pipeline:

```powershell
python .\launch_gui.py
```

The program intentionally exits with a clear error when the input file is
missing; it never falls back to synthetic data.

## Outputs

Generated results go into the ignored `outputs` directory:

- `model_comparison.csv` — final test MAE, RMSE, and R-squared for every model.
- `validation_comparison.csv` — validation scores used to select the model.
- `validation_errors_by_metro.csv` — validation-only MAE, RMSE, and WAPE by
  market for the selected model and both naive baselines, ranked by selected
  model MAE.
- `holdout_predictions.csv` — each metro's actual and predicted test values.
- `factor_importance.csv` — validation permutation importance if a learned
  model was selected; empty for naive baselines.
- `summary.json` — source, data coverage, split dates, selected model, and
  its final test scores.
- `charts\*.svg` and `dashboard.html` — charts and a self-contained report.
- `housing_analysis.sqlite` — model metrics, predictions, feature importance,
  and run metadata.

## Tests

Tests create small Zillow-shaped fixture values locally. They are test inputs,
not Zillow observations, are not used by the application, and do not need the
downloaded dataset.

```powershell
python -m unittest discover -s tests -v
```

For design and interpretation notes, see
[architecture](doc/ARCHITECTURE.md),
[the program guide](doc/UNDERSTANDING.md), and
[the documentation index](doc/README.md).
