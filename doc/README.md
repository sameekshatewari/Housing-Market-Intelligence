# Project README

The recruiter-facing project overview, dataset disclosure, model results,
installation instructions, charts, and reproduction steps are in the
[main README](../README.md) at the project root.

## Run from Windows PowerShell

```powershell
Set-Location 'C:\Users\sutew\OneDrive\Desktop\Housing_project\Housing-Market-Intelligence'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\generate_sample_data.py
python .\housing_market_intelligence.py --data .\data\housing.csv --date-column date --target-column price --features inflation_rate,mortgage_rate,unemployment_rate,fuel_price,interest_rate,synthetic_region_index --forecast-csv .\data\future_economic_inputs.csv
```

Run the test suite with:

```powershell
python -m unittest discover -s tests -v
```

For design notes, see [architecture](ARCHITECTURE.md) and
[understanding the program](UNDERSTANDING.md).
