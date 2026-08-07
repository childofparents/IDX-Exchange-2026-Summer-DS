# Auto-generated companion script for the notebook deliverable.
# Run from any folder inside the IDX Exchange project.


# ### IDX Exchange Team **ds55**
# ### Beini Lan
# # 04_model_comparison - Week 5 Additional Models
# 
# This notebook completes the Week 5 task from `Data Science v.4`: try Decision Tree and Random Forest regressors, compare their test R2 against the Week 4 baseline, and document model behavior.
# 
# The standardized evaluation methodology is fixed across interns: same chronological split, same latest full test month, no leakage, and shared R2, MAPE, and MdAPE metrics. The model choices and hyperparameters are this pipeline's experiments.

# ## Setup
# 
# This Week 5 iteration reuses the Kelvin KNN-imputed encoded dataset so the model comparison isolates model behavior more than preprocessing changes.

from pathlib import Path
import random
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

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
# The final train/test partition is chronological, rather than randomly sampled:
# May 2026 is always held out as the test month.  The seed is retained here as
# an explicit reproducibility control for any future random split experiment.
TRAIN_TEST_SPLIT_RANDOM_STATE = RANDOM_STATE
CROSS_VALIDATION_RANDOM_STATE = RANDOM_STATE
CROSS_VALIDATION_FOLDS = 5
# A single worker avoids machine-dependent parallel scheduling differences.
N_JOBS = 1
TRACE_COLUMNS = ["split", "CloseDate", "SaleMonth", "ClosePrice"]
FORBIDDEN_FEATURE_TERMS = ["ListPrice", "OriginalListPrice", "DaysOnMarket", "PurchaseContractDate"]

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


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

OUTPUT_DIR = PROJECT_ROOT / "W5 Additional Models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_MONTH = "2026-04"
print(f"Project root: {PROJECT_ROOT}")
print(f"Encoded modeling data: {DATA_PATH}")

# ## Load Data and Leakage Audit
# 
# The feature matrix excludes `CloseDate`, `SaleMonth`, and `ClosePrice`, along with any forbidden leakage fields if they appear. Absent sparse one-hot values are filled as zero when creating the model matrix.

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

# ## Standardized Split and Validation Month
# 
# The final test set remains the full latest month, May 2026. I use April 2026 only as an internal validation month for choosing between a small number of tree settings; it is then folded back into the final training set before evaluating May 2026.

development = train.loc[train["SaleMonth"].astype(str).ne(VALIDATION_MONTH)].copy()
validation = train.loc[train["SaleMonth"].astype(str).eq(VALIDATION_MONTH)].copy()
if validation.empty:
    raise ValueError(f"Validation month {VALIDATION_MONTH} not found in train split.")

split_summary = pd.DataFrame([
    {"split": "development", "rows": len(development), "months": f"{development['SaleMonth'].min()} to {development['SaleMonth'].max()}"},
    {"split": "validation", "rows": len(validation), "months": f"{validation['SaleMonth'].min()} to {validation['SaleMonth'].max()}"},
    {"split": "final_train", "rows": len(train), "months": f"{train['SaleMonth'].min()} to {train['SaleMonth'].max()}"},
    {"split": "test", "rows": len(test), "months": f"{test['SaleMonth'].min()} to {test['SaleMonth'].max()}"},
])
display(split_summary)
print(f"Diagnostic only: train-derived 0.5th/99.5th ClosePrice thresholds are ${train_target_lower:,.0f} to ${train_target_upper:,.0f}.")

# Expanding-window folds are chronological and never shuffled.  Therefore their
# membership is deterministic; CROSS_VALIDATION_RANDOM_STATE is recorded for
# consistency with the other reproducibility controls, but TimeSeriesSplit
# correctly does not accept or use a random_state parameter.
cv_ordered_development = development.sort_values(["CloseDate", "SaleMonth"]).reset_index(drop=True)
cv_splitter = TimeSeriesSplit(n_splits=CROSS_VALIDATION_FOLDS)
cv_fold_rows = []
for fold, (cv_train_indices, cv_validation_indices) in enumerate(cv_splitter.split(cv_ordered_development), start=1):
    cv_train = cv_ordered_development.iloc[cv_train_indices]
    cv_validation = cv_ordered_development.iloc[cv_validation_indices]
    cv_fold_rows.append({
        "fold": fold,
        "train_rows": len(cv_train),
        "validation_rows": len(cv_validation),
        "train_months": f"{cv_train['SaleMonth'].min()} to {cv_train['SaleMonth'].max()}",
        "validation_months": f"{cv_validation['SaleMonth'].min()} to {cv_validation['SaleMonth'].max()}",
        "shuffle": False,
        "random_state": CROSS_VALIDATION_RANDOM_STATE,
    })
cv_fold_plan = pd.DataFrame(cv_fold_rows)
cv_fold_plan.to_csv(OUTPUT_DIR / "model_comparison_cv_fold_plan.csv", index=False)
display(cv_fold_plan)

# ## Prepare Matrices and Metrics
# 
# The same 967 predictors are used for all three final model rows so the comparison table isolates model class and selected settings.

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

# ## Experiment Log: Validation Settings
# 
# I tried a small set of Decision Tree and Random Forest settings. The Random Forest with more trees, deeper trees, and `max_features=0.7` had the best validation MdAPE and R2, so it becomes the final Random Forest row.

X_development, development_blanks = make_sparse_matrix(development, feature_columns)
X_validation, validation_blanks = make_sparse_matrix(validation, feature_columns)
y_development = development["ClosePrice"].to_numpy(dtype=np.float64)
y_validation = validation["ClosePrice"].to_numpy(dtype=np.float64)

candidate_specs = [
    {"family": "Decision Tree", "model": "Decision Tree (depth 18, leaf 10)", "params": {"max_depth": 18, "min_samples_leaf": 10, "random_state": RANDOM_STATE}, "estimator": DecisionTreeRegressor(max_depth=18, min_samples_leaf=10, random_state=RANDOM_STATE)},
    {"family": "Decision Tree", "model": "Decision Tree (depth 24, leaf 10)", "params": {"max_depth": 24, "min_samples_leaf": 10, "random_state": RANDOM_STATE}, "estimator": DecisionTreeRegressor(max_depth=24, min_samples_leaf=10, random_state=RANDOM_STATE)},
    {"family": "Random Forest", "model": "Random Forest (40 trees, depth 24, leaf 10, max_features 0.4)", "params": {"n_estimators": 40, "max_depth": 24, "min_samples_leaf": 10, "max_features": 0.4, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}, "estimator": RandomForestRegressor(n_estimators=40, max_depth=24, min_samples_leaf=10, max_features=0.4, random_state=RANDOM_STATE, n_jobs=N_JOBS)},
    {"family": "Random Forest", "model": "Random Forest (60 trees, depth 30, leaf 10, max_features 0.7)", "params": {"n_estimators": 60, "max_depth": 30, "min_samples_leaf": 10, "max_features": 0.7, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}, "estimator": RandomForestRegressor(n_estimators=60, max_depth=30, min_samples_leaf=10, max_features=0.7, random_state=RANDOM_STATE, n_jobs=N_JOBS)},
]

validation_rows = []
for spec in candidate_specs:
    fit_start = time.time()
    spec["estimator"].fit(X_development, y_development)
    validation_predictions = spec["estimator"].predict(X_validation)
    metrics = regression_metrics(y_validation, validation_predictions)
    validation_rows.append({
        "family": spec["family"], "model": spec["model"], "validation_month": VALIDATION_MONTH,
        "development_rows": len(development), "validation_rows": len(validation),
        "validation_r2": metrics["r2"], "validation_mae": metrics["mae"], "validation_rmse": metrics["rmse"],
        "validation_mape": metrics["mape"], "validation_mdape": metrics["mdape"],
        "fit_seconds": time.time() - fit_start, "params": spec["params"],
    })

validation_results = pd.DataFrame(validation_rows)
selected_tree = validation_results.loc[validation_results["family"].eq("Decision Tree")].sort_values(["validation_mdape", "validation_r2"], ascending=[True, False]).iloc[0]
selected_forest = validation_results.loc[validation_results["family"].eq("Random Forest")].sort_values(["validation_mdape", "validation_r2"], ascending=[True, False]).iloc[0]
selected_params = {"Decision Tree": selected_tree["params"], "Random Forest": selected_forest["params"]}
validation_results.to_csv(OUTPUT_DIR / "model_comparison_validation_results.csv", index=False)
display(validation_results.drop(columns="params"))
print(f"Selected Decision Tree: {selected_tree['model']}")
print(f"Selected Random Forest: {selected_forest['model']}")

# ## Final Standardized Test Comparison
# 
# The three final rows are trained on all training months and evaluated on all May 2026 test rows. This is the table to use for Week 5 comparison.

final_specs = [
    {"family": "Linear Regression", "model": "Linear Regression (baseline)", "estimator": LinearRegression(n_jobs=N_JOBS), "params": {"n_jobs": N_JOBS}, "prediction_column": "linear_regression_prediction"},
    {"family": "Decision Tree", "model": selected_tree["model"], "estimator": DecisionTreeRegressor(**selected_params["Decision Tree"]), "params": selected_params["Decision Tree"], "prediction_column": "decision_tree_prediction"},
    {"family": "Random Forest", "model": selected_forest["model"], "estimator": RandomForestRegressor(**selected_params["Random Forest"]), "params": selected_params["Random Forest"], "prediction_column": "random_forest_prediction"},
]

comparison_rows = []
fitted_models = []
predictions = test[["CloseDate", "SaleMonth", "ClosePrice"]].copy()
predictions.insert(0, "source_row_index", test.index)

for spec in final_specs:
    fit_start = time.time()
    estimator = spec["estimator"]
    estimator.fit(X_train, y_train)
    fit_seconds = time.time() - fit_start
    train_predictions = estimator.predict(X_train)
    test_predictions = estimator.predict(X_test)
    train_metrics = regression_metrics(y_train, train_predictions)
    test_metrics = regression_metrics(y_test, test_predictions)
    predictions[spec["prediction_column"]] = test_predictions
    predictions[spec["prediction_column"].replace("prediction", "absolute_error")] = np.abs(y_test - test_predictions)
    predictions[spec["prediction_column"].replace("prediction", "absolute_percentage_error")] = np.abs(y_test - test_predictions) / y_test
    comparison_rows.append({
        "model": spec["model"], "family": spec["family"], "experiment": "A - Kelvin KNN encoded features, full standardized test month",
        "data_source": str(DATA_PATH), "train_rows": len(train), "test_rows": len(test), "feature_count": len(feature_columns),
        "train_months": f"{train['SaleMonth'].min()} to {train['SaleMonth'].max()}", "test_month": str(test["SaleMonth"].mode().iat[0]),
        "train_r2": train_metrics["r2"], "test_r2": test_metrics["r2"], "test_mae": test_metrics["mae"], "test_rmse": test_metrics["rmse"],
        "test_median_absolute_error": test_metrics["median_absolute_error"], "test_mape": test_metrics["mape"], "test_mdape": test_metrics["mdape"],
        "fit_seconds": fit_seconds, "selected_parameters": spec["params"],
    })
    fitted_models.append((spec, estimator))

comparison_results = pd.DataFrame(comparison_rows)
comparison_results.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "model_comparison_test_predictions.csv", index=False)
display(comparison_results[["model", "train_r2", "test_r2", "test_mae", "test_rmse", "test_mape", "test_mdape", "fit_seconds"]])

# ## Error Breakdown and Feature Signals
# 
# These diagnostics are not replacements for the standardized comparison table. They help explain where the models succeed or struggle.

price_band_rows = []
band_frame = predictions.copy()
band_frame["price_band"] = pd.qcut(band_frame["ClosePrice"], q=5, duplicates="drop")
for spec, _ in fitted_models:
    pred_col = spec["prediction_column"]
    ape_col = pred_col.replace("prediction", "absolute_percentage_error")
    err_col = pred_col.replace("prediction", "absolute_error")
    for price_band, group in band_frame.groupby("price_band", observed=True):
        price_band_rows.append({"model": spec["model"], "price_band": str(price_band), "rows": len(group), "mean_close_price": group["ClosePrice"].mean(), "test_r2": r2_score(group["ClosePrice"], group[pred_col]) if len(group) > 1 else np.nan, "mae": group[err_col].mean(), "mape": group[ape_col].mean(), "mdape": group[ape_col].median()})
price_band_metrics = pd.DataFrame(price_band_rows)
price_band_metrics.to_csv(OUTPUT_DIR / "model_comparison_price_band_metrics.csv", index=False)

importance_rows = []
for spec, estimator in fitted_models:
    if spec["family"] == "Linear Regression":
        values = np.asarray(estimator.coef_)
        ranking = np.argsort(np.abs(values))[::-1][:25]
        for rank, idx in enumerate(ranking, 1):
            importance_rows.append({"model": spec["model"], "rank": rank, "feature": feature_columns[idx], "importance_type": "coefficient", "value": float(values[idx])})
    else:
        values = np.asarray(estimator.feature_importances_)
        ranking = np.argsort(values)[::-1][:25]
        for rank, idx in enumerate(ranking, 1):
            importance_rows.append({"model": spec["model"], "rank": rank, "feature": feature_columns[idx], "importance_type": "feature_importance", "value": float(values[idx])})
feature_importance = pd.DataFrame(importance_rows)
feature_importance.to_csv(OUTPUT_DIR / "model_comparison_feature_importance.csv", index=False)
display(price_band_metrics.head(15))
display(feature_importance.groupby("model").head(8))

# ## Summary Report
# 
# The table below uses the same full May 2026 test set for all models. The Random Forest has the best R2 and MdAPE in this iteration.

def dollars(value):
    return f"${value:,.0f}"


def pct(value):
    return f"{value:.1%}"

display_table = comparison_results[["model", "train_r2", "test_r2", "test_mae", "test_rmse", "test_mape", "test_mdape", "fit_seconds"]].copy()
best_by_r2 = comparison_results.sort_values("test_r2", ascending=False).iloc[0]
best_by_mdape = comparison_results.sort_values("test_mdape", ascending=True).iloc[0]
report_lines = [
    "Week 5 Additional Models Summary Report", "======================================", "", TEAM,
    "Notebook: 04_model_comparison.ipynb", "Task: Week 5 - Additional Models", "",
    "Standardized evaluation controls", "--------------------------------",
    f"Encoded source: {DATA_PATH}", f"Train months: {train['SaleMonth'].min()} to {train['SaleMonth'].max()}",
    f"Validation month for model settings: {VALIDATION_MONTH}", f"Test month: {test['SaleMonth'].mode().iat[0]}",
    f"Feature count: {len(feature_columns):,}", f"Rows evaluated: {len(train):,} train, {len(test):,} test",
    "Excluded leakage fields: ListPrice, OriginalListPrice, DaysOnMarket, PurchaseContractDate.",
    "CloseDate and SaleMonth were used only for split/reporting, not as model features.",
    "No test rows were removed for the headline evaluation.", "",
    "Reproducibility controls", "------------------------",
    f"Global Python/NumPy/model seed: {RANDOM_STATE}.",
    f"Train/test split seed: {TRAIN_TEST_SPLIT_RANDOM_STATE} (not consumed: the split is chronological by month).",
    f"Cross-validation seed: {CROSS_VALIDATION_RANDOM_STATE}; {CROSS_VALIDATION_FOLDS} expanding-window folds, shuffle=False.",
    f"Model workers: {N_JOBS} (single worker for repeatable execution).",
    "Pinned package versions are recorded in requirements.txt.", "",
    "Cross-validation fold plan", "--------------------------", cv_fold_plan.to_string(index=False), "",
    "Validation settings check", "-------------------------", validation_results.drop(columns="params").to_string(index=False), "",
    "Final model comparison", "----------------------", display_table.to_string(index=False), "",
    "Interpretation", "--------------",
    f"Best test R2: {best_by_r2['model']} at {best_by_r2['test_r2']:.3f}.",
    f"Best typical percentage error: {best_by_mdape['model']} at {pct(best_by_mdape['test_mdape'])} MdAPE.",
    "Linear Regression is transparent but sensitive to extreme prices and linearity assumptions.",
    "The single Decision Tree improves MdAPE versus the linear baseline but performs worse on R2, showing weaker stability on high-dollar variance.",
    "The Random Forest gives the best Week 5 result because averaging regularized trees captures nonlinear interactions while reducing single-tree variance.",
    "Next iterations should document specific preprocessing and feature-engineering changes, then compare them against this experiment table.",
]
report_text = "\n".join(report_lines)
report_path = OUTPUT_DIR / "Week5_Additional_Models_Summary_Report.txt"
report_path.write_text(report_text, encoding="utf-8")
print(report_text)
