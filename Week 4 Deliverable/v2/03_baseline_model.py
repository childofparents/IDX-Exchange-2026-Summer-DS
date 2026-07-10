# Companion script for W4 notebook.

# ### IDX Exchange Team **ds55**
# ### Beini Lan
# # 03_baseline_model (Week 4 Baseline Model)
# 
# This notebook completes the Week 4 task: train a Linear Regression baseline, evaluate test-set R2, and record baseline results.
# 
# Chronological train/test split, latest full month as the test month, no target leakage, and common metrics R2, MAPE, and MdAPE. Modeling choices remain my pipeline decisions ane documented below.

# ## Setup
# 
# The notebook finds the project root from its current location. I use Kelvin's KNN-imputed and one-hot encoded Week 3 output as this iteration's modeling matrix.

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 140)
pd.set_option("display.float_format", "{:,.4f}".format)

TEAM = "Team ds55, Beini Lan"
RANDOM_STATE = 55
TRACE_COLUMNS = ["split", "CloseDate", "SaleMonth", "ClosePrice"]
FORBIDDEN_FEATURE_TERMS = ["ListPrice", "OriginalListPrice", "DaysOnMarket", "PurchaseContractDate"]


def find_project_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "Data Science v.4.pdf").exists() and (candidate / "W3 Data Preprocessing").exists():
            return candidate
    fallback = Path("/Users/HP/Documents/IDX Exchange")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not locate the IDX Exchange project root.")

PROJECT_ROOT = find_project_root()
DATA_PATH = PROJECT_ROOT / "W3 Data Preprocessing" / "Kelvin" / "model_data_knn_after_encoding.csv.gz"
BEFORE_ENCODING_PATH = PROJECT_ROOT / "W3 Data Preprocessing" / "Kelvin" / "model_data_knn_before_encoding.csv"

OUTPUT_DIR = PROJECT_ROOT / "W4 Baseline Model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Project root: {PROJECT_ROOT}")
print(f"Encoded modeling data: {DATA_PATH}")

# ## Load Data and Leakage Audit
# 
# `CloseDate` and `SaleMonth` remain trace/split fields and are not predictors. The encoded file stores absent one-hot categories as blank values; those blanks are filled with `0.0` only when building the model matrix.

load_start = time.time()
encoded_columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
missing_trace = sorted(set(TRACE_COLUMNS).difference(encoded_columns))
if missing_trace:
    raise ValueError(f"Missing required trace columns: {missing_trace}")

feature_columns = [column for column in encoded_columns if column not in TRACE_COLUMNS]
leakage_columns = [
    column for column in feature_columns
    if any(term.lower() in column.lower() for term in FORBIDDEN_FEATURE_TERMS)
]
if leakage_columns:
    raise ValueError(f"Leakage columns found in feature matrix: {leakage_columns}")

dtype_map = {column: "float32" for column in feature_columns}
dtype_map.update({"split": "category", "CloseDate": "string", "SaleMonth": "string", "ClosePrice": "float64"})
data = pd.read_csv(DATA_PATH, dtype=dtype_map)

train = data.loc[data["split"].astype(str).eq("train")].copy()
test = data.loc[data["split"].astype(str).eq("test")].copy()
rare_category_columns = [column for column in feature_columns if column.endswith("infrequent_sklearn")]
train_target_lower, train_target_upper = np.percentile(train["ClosePrice"], [0.5, 99.5])

data_summary = pd.DataFrame([
    {"item": "rows", "value": f"{len(data):,}"},
    {"item": "feature_count", "value": f"{len(feature_columns):,}"},
    {"item": "train_rows", "value": f"{len(train):,}"},
    {"item": "test_rows", "value": f"{len(test):,}"},
    {"item": "train_months", "value": f"{train['SaleMonth'].min()} to {train['SaleMonth'].max()}"},
    {"item": "test_month", "value": str(test['SaleMonth'].mode().iat[0])},
    {"item": "forbidden_feature_matches", "value": len(leakage_columns)},
    {"item": "rare_category_columns", "value": ", ".join(rare_category_columns)},
    {"item": "train_only_closeprice_p005_p995", "value": f"${train_target_lower:,.0f} to ${train_target_upper:,.0f}"},
    {"item": "load_seconds", "value": f"{time.time() - load_start:.1f}"},
])
display(data_summary)

# ## Standardized Train/Test Split
# 
# The test month is May 2026. I report primary metrics on all 12,024 May 2026 test rows so results remain comparable across team members.
# The train-only 0.5th/99.5th percentile `ClosePrice` thresholds are shown as a diagnostic, but they are not used to remove test rows in the headline evaluation.

split_summary = pd.DataFrame([
    {"split": "train", "rows": len(train), "months": f"{train['SaleMonth'].min()} to {train['SaleMonth'].max()}", "min_close_price": train["ClosePrice"].min(), "max_close_price": train["ClosePrice"].max()},
    {"split": "test", "rows": len(test), "months": f"{test['SaleMonth'].min()} to {test['SaleMonth'].max()}", "min_close_price": test["ClosePrice"].min(), "max_close_price": test["ClosePrice"].max()},
])
display(split_summary)
print(f"Diagnostic only: train-derived 0.5th/99.5th ClosePrice thresholds are ${train_target_lower:,.0f} to ${train_target_upper:,.0f}.")

# ## Prepare Model Matrix
# 
# The feature matrix contains 967 encoded predictors. No listing-price, days-on-market, or purchase-contract-date fields are present.

def make_sparse_matrix(frame, columns):
    array = frame[columns].to_numpy(dtype=np.float32, copy=True)
    blanks_filled = int(np.isnan(array).sum())
    np.nan_to_num(array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return sparse.csr_matrix(array), blanks_filled


def regression_metrics(y_true, y_pred):
    absolute_percentage_error = np.abs(y_true - y_pred) / y_true
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(absolute_percentage_error)),
        "mdape": float(np.median(absolute_percentage_error)),
    }

X_train, train_blanks_filled = make_sparse_matrix(train, feature_columns)
X_test, test_blanks_filled = make_sparse_matrix(test, feature_columns)
y_train = train["ClosePrice"].to_numpy(dtype=np.float64)
y_test = test["ClosePrice"].to_numpy(dtype=np.float64)

matrix_summary = pd.DataFrame([
    {"matrix": "X_train", "rows": X_train.shape[0], "columns": X_train.shape[1], "non_zero_values": X_train.nnz, "blanks_filled_as_zero": train_blanks_filled},
    {"matrix": "X_test", "rows": X_test.shape[0], "columns": X_test.shape[1], "non_zero_values": X_test.nnz, "blanks_filled_as_zero": test_blanks_filled},
])
display(matrix_summary)

# ## Train Linear Regression Baseline
# 
# Linear Regression is the Week 4 baseline because it is transparent and easy to compare against later nonlinear models.

MODEL_NAME = "Linear Regression"
fit_start = time.time()
model = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train)
fit_seconds = time.time() - fit_start

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_metrics = regression_metrics(y_train, train_predictions)
test_metrics = regression_metrics(y_test, test_predictions)

baseline_results = pd.DataFrame([{
    "model": MODEL_NAME,
    "experiment": "A - Kelvin KNN encoded features, full standardized test month",
    "data_source": str(DATA_PATH),
    "train_rows": len(train),
    "test_rows": len(test),
    "feature_count": len(feature_columns),
    "train_months": f"{train['SaleMonth'].min()} to {train['SaleMonth'].max()}",
    "test_month": str(test["SaleMonth"].mode().iat[0]),
    "train_r2": train_metrics["r2"],
    "test_r2": test_metrics["r2"],
    "test_mae": test_metrics["mae"],
    "test_rmse": test_metrics["rmse"],
    "test_median_absolute_error": test_metrics["median_absolute_error"],
    "test_mape": test_metrics["mape"],
    "test_mdape": test_metrics["mdape"],
    "fit_seconds": fit_seconds,
}])
display(baseline_results)

# ## Save Week 4 Deliverables
# 
# The result table is the Week 4 baseline table. I also save holdout predictions, top coefficients, and price-band diagnostics for review.

baseline_results_path = OUTPUT_DIR / "baseline_model_results.csv"
predictions_path = OUTPUT_DIR / "baseline_model_test_predictions.csv"
coefficients_path = OUTPUT_DIR / "baseline_model_top_coefficients.csv"
price_band_path = OUTPUT_DIR / "baseline_model_price_band_metrics.csv"

baseline_results.to_csv(baseline_results_path, index=False)

predictions = test[["CloseDate", "SaleMonth", "ClosePrice"]].copy()
predictions.insert(0, "source_row_index", test.index)
predictions["prediction"] = test_predictions
predictions["absolute_error"] = (predictions["ClosePrice"] - predictions["prediction"]).abs()
predictions["absolute_percentage_error"] = predictions["absolute_error"] / predictions["ClosePrice"]
predictions.to_csv(predictions_path, index=False)

coefficient_table = pd.DataFrame({"feature": feature_columns, "coefficient": model.coef_})
coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs()
coefficient_table = coefficient_table.sort_values("abs_coefficient", ascending=False).head(25).drop(columns="abs_coefficient")
coefficient_table.to_csv(coefficients_path, index=False)

band_frame = predictions.copy()
band_frame["price_band"] = pd.qcut(band_frame["ClosePrice"], q=5, duplicates="drop")
price_band_metrics = (
    band_frame.groupby("price_band", observed=True)
    .apply(lambda group: pd.Series({
        "rows": len(group),
        "mean_close_price": group["ClosePrice"].mean(),
        "test_r2": r2_score(group["ClosePrice"], group["prediction"]) if len(group) > 1 else np.nan,
        "mae": group["absolute_error"].mean(),
        "mape": group["absolute_percentage_error"].mean(),
        "mdape": group["absolute_percentage_error"].median(),
    }))
    .reset_index()
)
price_band_metrics["price_band"] = price_band_metrics["price_band"].astype(str)
price_band_metrics.to_csv(price_band_path, index=False)

print(f"Saved {baseline_results_path}")
print(f"Saved {predictions_path}")
print(f"Saved {coefficients_path}")
print(f"Saved {price_band_path}")
display(coefficient_table.head(15))
display(price_band_metrics)

# ## Summary Report
# 
# R2 is required for Week 4, and MAPE/MdAPE are included for team comparison. MdAPE is most useful as the typical percentage error because the target is highly skewed.

def dollars(value):
    return f"${value:,.0f}"


def pct(value):
    return f"{value:.1%}"

row = baseline_results.iloc[0]
report_lines = [
    "Week 4 Baseline Model Summary Report",
    "====================================",
    "",
    TEAM,
    "Notebook: 03_baseline_model.ipynb",
    "Task: Week 4 - Baseline Model",
    "",
    "Standardized evaluation controls",
    "--------------------------------",
    f"Encoded source: {DATA_PATH}",
    f"Readable pre-encoding source: {BEFORE_ENCODING_PATH}",
    f"Train months: {row['train_months']}",
    f"Test month: {row['test_month']}",
    f"Rows evaluated: {int(row['train_rows']):,} train, {int(row['test_rows']):,} test",
    f"Feature count: {int(row['feature_count']):,}",
    "Excluded leakage fields: ListPrice, OriginalListPrice, DaysOnMarket, PurchaseContractDate.",
    "CloseDate and SaleMonth were used only for split/reporting, not as model features.",
    "No test rows were removed for the headline evaluation.",
    "",
    "Baseline results",
    "----------------",
    baseline_results.to_string(index=False),
    "",
    "Interpretation",
    "--------------",
    f"The Linear Regression baseline explains {row['test_r2']:.3f} of variance on the full May 2026 test month.",
    f"The typical holdout percentage error is {pct(row['test_mdape'])} MdAPE; mean percentage error is {pct(row['test_mape'])} MAPE.",
    f"Average absolute dollar error is {dollars(row['test_mae'])}.",
    "The model is transparent, but high-end outliers and nonlinear location effects limit performance.",
    "",
    "Largest absolute coefficients",
    "-----------------------------",
    coefficient_table.head(15).to_string(index=False),
]
report_text = "\n".join(report_lines)
report_path = OUTPUT_DIR / "Week4_Baseline_Model_Summary_Report.txt"
report_path.write_text(report_text, encoding="utf-8")
print(report_text)
