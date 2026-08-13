# Solari Demand Forecasting

## Overview
Company X operates a network of **800+ micro‑fulfillment hubs** delivering groceries within 15‑30 minutes. Accurate daily order‑volume forecasts are critical because over‑stocking wastes limited storage and perishes goods, while under‑stocking leads to cancelled orders and loss of trust.

This repository implements a **time‑series regression** solution that predicts the `OrderVolume` for every hub over a future time window using:
- Historical daily orders (`orders_train.csv`)
- Planned operational context for the test period (`orders_test.csv`)
- Hub‑level static attributes (`hub_metadata.csv`)
- Promotion, holiday, and loyalty program information

The competition evaluates submissions with **Root Mean Squared Logarithmic Error (RMSLE)**, which penalises proportional errors and gracefully handles zero‑order rows (closed hubs).

---

## Data
| File | Description |
|------|-------------|
| `orders_train.csv` | Historical daily order volume per hub with context features (rider availability, promotion, holidays, etc.). |
| `orders_test.csv`  | Same schema as `orders_train.csv` but **without** `OrderVolume`. Planned values replace unknown future data. |
| `hub_metadata.csv`| Static and semi‑static hub attributes (format, assortment tier, competitor distance, loyalty program status, launch date). |
| `sample_submission.csv` | Template for the required submission format (`Id,OrderVolume`). |

### Key columns
- `Id` – unique identifier for a (Hub, Date) pair in the test set
- `HubID` – hub identifier
- `OrderVolume` – target variable (daily orders)
- `IsOpen` – 1 if the hub operates that day, else 0
- `PromoActive`, `RegionalHoliday`, `SchoolClosureFlag` – operational context flags
- `HubFormat`, `AssortmentTier`, `CompetitorDistance` – hub characteristics
- `LoyaltyProgram`, `LoyaltyProgramSinceYear/Week`, `LoyaltyProgramInterval` – loyalty‑program related features

---

## Modeling pipeline (`final.py`)
1. **Load data** using the helper `load_raw()` from `features.py`.
2. **Create base features** – date parts, day index, competitor‑open months, loyalty weeks, etc.
3. **Add historical hub statistics** – mean, median, recent‑day trends, weekday profiles, promotion uplift, month‑of‑year seasonality.
4. **Add calendar context** – previous/next open‑day flags, promo start flags, closed‑run lengths.
5. **Add last‑year reference** – log‑scaled order volume from the same calendar week a year earlier.
6. **Train XGBoost** (regression) on the log‑transformed target (`log1p(OrderVolume)`).
7. **Predict** on the test set, average predictions across seeds (if multiple runs), clip negatives, enforce `OrderVolume = 0` when `IsOpen = 0`.
8. **Write submission** to `submission.csv`.

The script prints a short summary of the prediction distribution and the number of zero rows, which matches the competition’s expectations.

---

## Quick start (Windows)
```powershell
# 1. Clone / open this folder
cd C:\Users\hp\Downloads\files(1)

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.\.venv\Scripts\activate

# 4. Install dependencies (requirements.txt was generated automatically)
pip install -r requirements.txt

# 5. Run the model (seed can be changed, e.g., 42)
python final.py 42
```
The script will generate `submission.csv` in the project root and print a summary similar to:
```
fit (804056, 50) test (46830, 49) (4s)
seed 42 done (483s)
averaging: ['C:\Users\hp\Downloads\files(1)\pred_42.npy']

 count    46830.000000
 mean      5969.238770
 std       3630.340820
 min          0.000000
 25%       4130.615845
 50%       5870.427734
 75%       7944.912720
 max      29838.375000
Name: OrderVolume, dtype: float64
zero rows: 6548
written (483s)
```
A verification file `check.csv` (artifact) records these statistics for quick reference.

---

## Re‑running with a different seed
You can experiment with different random seeds to obtain an ensemble of predictions:
```powershell
python final.py 123   # any integer seed
```
All `pred_<seed>.npy` files will be saved; the script automatically averages them.

---

## Evaluation
The competition uses **RMSLE**:
```
RMSLE = sqrt( (1/n) * Σ ( log1p(p_i) - log1p(a_i) )^2 )
```
Because the metric works in log‑space, the model is trained on `log1p(OrderVolume)`.

---

## Project structure
```
├─ final.py               # main training / inference script
├─ features.py            # data loading & feature‑engineering helpers
├─ requirements.txt       # Python dependencies (numpy, pandas, xgboost, matplotlib)
├─ submission.csv         # generated submission file
├─ check.csv (artifact)   # summary statistics of the predictions
├─ orders_train.csv
├─ orders_test.csv
├─ hub_metadata.csv
├─ sample_submission.csv
└─ charts.py (optional)   # visualisations used for analysis
```
---
