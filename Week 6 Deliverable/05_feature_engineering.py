#!/usr/bin/env python
# coding: utf-8

# ### IDX Exchange Team **ds55**
# ### Beini Lan
# # 05_feature_engineering - Week 6 Feature Engineering
# 
# This notebook updates the Week 4-5 workflow rather than starting a separate modeling approach. The model families, standardized May 2026 holdout, metrics, result-table layout, predictions, price-band diagnostics, and feature-importance outputs are carried forward. The material change is the preprocessing and feature set, followed by five-fold chronological cross-validation and re-training of:
# 
# 1. Linear Regression
# 2. Decision Tree Regressor
# 3. Random Forest Regressor

# ## Source grounding and design rules
# 
# This week's work is grounded in `AVM_Data_Science_Best_Practices v.1.pdf`, especially **"Three Feature Families Drive Most of the Predictive Power"**:
# 
# - **Intrinsic property characteristics:** retain physical facts and add condition- or utility-relevant relationships.
# - **Locational features:** turn latitude/longitude into a stable categorical spatial layer.
# - **Temporal features:** represent sale-time age and cyclical month without treating December and January as far apart.
# 
# The notebook also follows the document's related controls: exclude sale-process leakage, keep the final month untouched, fit preprocessing inside each cross-validation fold, retain missingness signals, report R2/MAPE/MdAPE/MAE together, and pin the environment.
# 
# `Data Science v.4.pdf`, Week 6, specifically requires bed/bath ratio, property age at time of sale, and a spatial join to California school-district boundaries. Following the mentor's clarified instruction, this work uses the **California School District Areas 2025-26** layer and filters it to unified school districts before the join:
# 
# https://data.ca.gov/dataset/california-school-district-areas-2025-26

# ## Setup
#
# Install the pinned Week 6 environment once with `python -m pip install -r requirements.txt`.
# If only the mapping dependency is missing, the mentor's minimum command is
# `python -m pip install geopandas`.

# In[1]:


from pathlib import Path
import ast
import gc
import json
import os
import random
import time
import urllib.request
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 160)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.float_format", "{:,.4f}".format)

TEAM = "Team ds55, Beini Lan"
RANDOM_STATE = 55
CROSS_VALIDATION_FOLDS = 5
N_JOBS = -1
TEST_MONTH = "2026-05"
EXPECTED_TEST_ROWS = 12_024
EXPERIMENT = (
    "B - Engineered intrinsic, cyclical-time, and CDE 2025-26 "
    "unified-school-district spatial features; full standardized test month"
)
FORBIDDEN_FEATURE_TERMS = [
    "ListPrice",
    "OriginalListPrice",
    "DaysOnMarket",
    "PurchaseContractDate",
]

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def candidate_project_roots():
    roots = []
    env_root = os.environ.get("IDX_PROJECT_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())

    starts = [
        Path.cwd(),
        Path("/content"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shareddrives"),
        Path("/kaggle/input"),
    ]
    for start in starts:
        if start.exists():
            roots.extend([start, *start.parents])

    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def find_project_root():
    required = Path("W3 Data Preprocessing/Kelvin/model_data_knn_before_encoding.csv")
    for root in candidate_project_roots():
        if (root / required).exists() and (root / "Data Science v.4.pdf").exists():
            return root

    for mount in [Path("/content/drive/MyDrive"), Path("/kaggle/input")]:
        if mount.exists():
            for match in mount.glob("**/W3 Data Preprocessing/Kelvin/model_data_knn_before_encoding.csv"):
                root = match.parents[3]
                if (root / "Data Science v.4.pdf").exists():
                    return root

    raise FileNotFoundError(
        "Could not locate the IDX Exchange project. Set IDX_PROJECT_ROOT or place "
        "the W3 Data Preprocessing folder beside Data Science v.4.pdf."
    )


PROJECT_ROOT = find_project_root()
INPUT_DATA = PROJECT_ROOT / "W3 Data Preprocessing" / "Kelvin" / "model_data_knn_before_encoding.csv"
OUTPUT_DIR = PROJECT_ROOT / "W6 Feature Engineering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCHOOL_DISTRICT_FILE = (
    OUTPUT_DIR / "resources" / "California_School_District_Areas_2025-26.geojson"
)
SCHOOL_DISTRICT_FILE.parent.mkdir(parents=True, exist_ok=True)
SCHOOL_DISTRICT_DOWNLOAD_URL = (
    "https://gis.data.ca.gov/api/download/v1/items/"
    "48870daecfe14c6ab376f6a673491914/geojson?layers=0"
)
WEEK5_RESULTS = PROJECT_ROOT / "W5 Additional Models" / "model_comparison_results.csv"

print("Week 6 environment ready.")


# ## Load the Week 3 readable modeling table
# 
# The Week 3 readable table is reused so this Week 6 update stays focused on the preprocessing and feature-engineering sections of the Week 4-5 workflow. Before engineering features, the notebook uses the retained `*_missing_ind` flags to replace Week 3 KNN-filled values with missing values again. This lets the Week 6 imputer learn only from each chronological training fold. The target, split labels, and trace dates are not model features.

# In[2]:


load_start = time.time()
df = pd.read_csv(INPUT_DATA, dtype={"PostalCode": "string"})
df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")
df["SaleMonth"] = df["CloseDate"].dt.to_period("M").astype(str)

restored_missing_counts = {}
for indicator_column in [
    column for column in df.columns if column.endswith("_missing_ind")
]:
    source_column = indicator_column.removesuffix("_missing_ind")
    if source_column in df.columns:
        missing_mask = pd.to_numeric(
            df[indicator_column],
            errors="coerce",
        ).fillna(0).eq(1)
        df.loc[missing_mask, source_column] = np.nan
        restored_missing_counts[source_column] = int(missing_mask.sum())

required_columns = {
    "split",
    "CloseDate",
    "SaleMonth",
    "ClosePrice",
    "Latitude",
    "Longitude",
    "LivingArea",
    "LotSizeSquareFeet",
    "YearBuilt",
    "BathroomsTotalInteger",
    "BedroomsTotal",
}
missing_required = sorted(required_columns.difference(df.columns))
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

base_feature_columns = [
    column
    for column in df.columns
    if column not in {"split", "CloseDate", "SaleMonth", "ClosePrice"}
]

load_summary = pd.DataFrame(
    [
        {"item": "rows", "value": f"{len(df):,}"},
        {"item": "base readable feature columns", "value": f"{len(base_feature_columns):,}"},
        {
            "item": "train months",
            "value": f"{df.loc[df['split'].eq('train'), 'SaleMonth'].min()} to "
            f"{df.loc[df['split'].eq('train'), 'SaleMonth'].max()}",
        },
        {
            "item": "test month",
            "value": df.loc[df["split"].eq("test"), "SaleMonth"].mode().iat[0],
        },
        {
            "item": "columns with pre-imputation missingness restored",
            "value": f"{len(restored_missing_counts):,}",
        },
        {"item": "load seconds", "value": f"{time.time() - load_start:.1f}"},
    ]
)
display(load_summary)


# ## Feature engineering

# ### 1. Bed/bath ratio

# In[3]:


bedrooms = pd.to_numeric(df["BedroomsTotal"], errors="coerce")
bathrooms = pd.to_numeric(df["BathroomsTotalInteger"], errors="coerce")

df["BedBathRatio"] = np.divide(
    bedrooms,
    bathrooms,
    out=np.full(len(df), np.nan, dtype=float),
    where=bathrooms.gt(0).to_numpy(),
)
df["BedBathRatio"] = df["BedBathRatio"].clip(lower=0, upper=10)


# **Rationale.** Bedrooms and bathrooms are useful separately, but their ratio represents functional balance: two five-bedroom homes can have very different utility when one has two bathrooms and the other has five. Division-by-zero becomes missing and the broad upper cap limits implausible ratios without using `ClosePrice`.

# ### 2. Property age at time of sale

# In[4]:


sale_year = df["CloseDate"].dt.year
year_built = pd.to_numeric(df["YearBuilt"], errors="coerce")
raw_property_age = sale_year - year_built

df["PropertyAgeAtSale"] = raw_property_age.where(raw_property_age.between(0, 200))
df["PropertyAgeAtSale_missing_ind"] = df["PropertyAgeAtSale"].isna().astype("int8")


# **Rationale.** `YearBuilt` is static, while age at the valuation/sale date reflects condition-relevant aging and changes as time passes. Negative or implausibly old ages are treated as missing; the missing indicator is retained and fold-fitted median imputation handles the numeric value later.

# ### 3. Cyclical sale-month features

# In[5]:


sale_month_number = df["CloseDate"].dt.month
df["SaleMonthSin"] = np.sin(2 * np.pi * sale_month_number / 12)
df["SaleMonthCos"] = np.cos(2 * np.pi * sale_month_number / 12)


# **Rationale.** The valuation date is known at prediction time, and California housing activity can be seasonal. Sine/cosine encoding captures that cycle while keeping December adjacent to January, unlike an ordinal month number. The raw `CloseDate` and `SaleMonth` remain trace fields and are excluded from the model.

# ### 4. Log transforms for skewed size measures

# In[6]:


df["LogLivingArea"] = np.log1p(
    pd.to_numeric(df["LivingArea"], errors="coerce").clip(lower=0)
)
df["LogLotSizeSquareFeet"] = np.log1p(
    pd.to_numeric(df["LotSizeSquareFeet"], errors="coerce").clip(lower=0)
)


# **Rationale.** Living area and lot size are strongly right-skewed, and an additional square foot usually has diminishing marginal value. `log1p` preserves ordering and zero handling while giving Linear Regression a more realistic nonlinear shape; the original size columns are retained for the tree models.

# ### 5. CDE unified school-district boundary mapping

# In[7]:


if not SCHOOL_DISTRICT_FILE.exists():
    temporary_download = SCHOOL_DISTRICT_FILE.with_suffix(".download")
    urllib.request.urlretrieve(SCHOOL_DISTRICT_DOWNLOAD_URL, temporary_download)
    temporary_download.replace(SCHOOL_DISTRICT_FILE)

districts = gpd.read_file(
    SCHOOL_DISTRICT_FILE,
    columns=["DistrictName", "DistrictType"],
).to_crs("EPSG:4326")
unified_districts = districts.loc[
    districts["DistrictType"].eq("Unified") & districts["DistrictName"].notna(),
    ["DistrictName", "geometry"],
].copy()
if unified_districts.empty:
    raise ValueError("No DistrictType='Unified' polygons were found in the boundary file.")

points = gpd.GeoDataFrame(
    {"source_row_index": df.index},
    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    crs="EPSG:4326",
)

point_district_matches = gpd.sjoin(
    points,
    unified_districts,
    how="left",
    predicate="within",
)

match_counts = (
    point_district_matches.groupby("source_row_index")["DistrictName"]
    .count()
    .reindex(df.index, fill_value=0)
)
if match_counts.gt(1).any():
    raise ValueError(
        "A property matched more than one Unified district polygon; inspect the "
        "2025-26 boundary layer before assigning DistrictName."
    )

district_name_by_row = (
    point_district_matches.groupby("source_row_index")["DistrictName"]
    .first()
    .reindex(df.index)
)
df["DistrictName"] = district_name_by_row
df["SchoolDistrictBoundaryMatched"] = df["DistrictName"].notna().astype("int8")
df["DistrictName"] = df["DistrictName"].fillna("No boundary match")

spatial_join_summary = pd.DataFrame(
    [
        {"item": "CDE 2025-26 polygons", "value": f"{len(districts):,}"},
        {
            "item": "Unified polygons used in spatial join",
            "value": f"{len(unified_districts):,}",
        },
        {
            "item": "properties matched to a Unified district",
            "value": f"{df['SchoolDistrictBoundaryMatched'].sum():,}",
        },
        {
            "item": "properties with no Unified district match",
            "value": f"{(1 - df['SchoolDistrictBoundaryMatched']).sum():,}",
        },
        {
            "item": "unique matched Unified DistrictName values",
            "value": f"{df.loc[df['SchoolDistrictBoundaryMatched'].eq(1), 'DistrictName'].nunique():,}",
        },
    ]
)
display(spatial_join_summary)


# **Rationale and algorithm.** This is deterministic **boundary-based spatial clustering**, not target-driven K-means and not a price aggregate. The official 2025-26 CDE layer is filtered to `DistrictType == "Unified"` before property points are joined with `gpd.sjoin(..., predicate="within")`. The resulting `DistrictName` is saved directly on every property row and is included in the downstream feature matrix; unmatched coordinates are retained as `No boundary match`, with a separate match flag. This follows the mentor's mapping instructions exactly and provides a target-independent location category without learning from `ClosePrice`.

# ### Old versus feature-engineered columns

# In[8]:


feature_update_table = pd.DataFrame(
    [
        {
            "feature family": "Intrinsic",
            "old/input columns": "BedroomsTotal + BathroomsTotalInteger",
            "feature-engineered columns": "BedBathRatio",
            "method": "Safe ratio with broad cap",
            "model role": "Added predictor",
        },
        {
            "feature family": "Intrinsic + temporal",
            "old/input columns": "YearBuilt + CloseDate",
            "feature-engineered columns": "PropertyAgeAtSale + missing indicator",
            "method": "Sale year minus year built; invalid ages set missing",
            "model role": "Added predictors",
        },
        {
            "feature family": "Temporal",
            "old/input columns": "CloseDate month",
            "feature-engineered columns": "SaleMonthSin + SaleMonthCos",
            "method": "12-month sine/cosine cycle",
            "model role": "Added predictors; raw dates remain trace-only",
        },
        {
            "feature family": "Intrinsic",
            "old/input columns": "LivingArea",
            "feature-engineered columns": "LogLivingArea",
            "method": "log1p transform",
            "model role": "Added predictor; original retained",
        },
        {
            "feature family": "Intrinsic",
            "old/input columns": "LotSizeSquareFeet",
            "feature-engineered columns": "LogLotSizeSquareFeet",
            "method": "log1p transform",
            "model role": "Added predictor; original retained",
        },
        {
            "feature family": "Locational",
            "old/input columns": "Latitude + Longitude + listing-entered HighSchoolDistrict",
            "feature-engineered columns": "DistrictName + boundary-match flag",
            "method": "CDE 2025-26 Unified-only point-in-polygon spatial join",
            "model role": "Added predictors and used throughout downstream modeling",
        },
    ]
)
feature_update_table.to_csv(
    OUTPUT_DIR / "feature_engineering_old_vs_new_columns.csv",
    index=False,
)
display(feature_update_table)


# ### First five rows of the updated dataset

# In[9]:


df.head()


# ### Save the updated feature set

# In[10]:


updated_feature_set_path = OUTPUT_DIR / "updated_feature_set.csv.gz"
df.to_csv(updated_feature_set_path, index=False, compression="gzip")

updated_feature_summary = pd.DataFrame(
    [
        {"item": "rows", "value": f"{len(df):,}"},
        {"item": "columns", "value": f"{df.shape[1]:,}"},
        {"item": "saved CSV", "value": updated_feature_set_path.name},
    ]
)
display(updated_feature_summary)


# ## Standardized evaluation controls

# In[11]:


TRACE_COLUMNS = ["split", "CloseDate", "SaleMonth", "ClosePrice"]

train = df.loc[df["split"].astype(str).eq("train")].copy()
test = df.loc[df["split"].astype(str).eq("test")].copy()

test_months = sorted(test["SaleMonth"].dropna().unique().tolist())
if test_months != [TEST_MONTH]:
    raise ValueError(f"Expected test month {[TEST_MONTH]}, found {test_months}")
if len(test) != EXPECTED_TEST_ROWS:
    raise ValueError(f"Expected {EXPECTED_TEST_ROWS:,} test rows, found {len(test):,}")

feature_columns = [
    column
    for column in df.columns
    if column not in TRACE_COLUMNS
]
leakage_columns = [
    column
    for column in feature_columns
    if any(term.lower() in column.lower() for term in FORBIDDEN_FEATURE_TERMS)
]
if leakage_columns:
    raise ValueError(f"Leakage columns found in feature matrix: {leakage_columns}")

X_train = train[feature_columns].copy()
X_test = test[feature_columns].copy()
y_train = train["ClosePrice"].to_numpy(dtype=np.float64)
y_test = test["ClosePrice"].to_numpy(dtype=np.float64)

if "PostalCode" in X_train.columns:
    X_train["PostalCode"] = X_train["PostalCode"].astype("string")
    X_test["PostalCode"] = X_test["PostalCode"].astype("string")

numeric_columns = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_columns = [
    column for column in feature_columns if column not in numeric_columns
]

for column in categorical_columns:
    X_train[column] = X_train[column].astype(object)
    X_test[column] = X_test[column].astype(object)

X_train = X_train.where(pd.notna(X_train), np.nan)
X_test = X_test.where(pd.notna(X_test), np.nan)

split_summary = pd.DataFrame(
    [
        {
            "split": "final_train",
            "rows": len(train),
            "months": f"{train['SaleMonth'].min()} to {train['SaleMonth'].max()}",
        },
        {
            "split": "test",
            "rows": len(test),
            "months": f"{test['SaleMonth'].min()} to {test['SaleMonth'].max()}",
        },
    ]
)
display(split_summary)
print(f"Readable model columns before fold-fitted encoding: {len(feature_columns):,}")
print(f"Forbidden feature matches: {len(leakage_columns)}")


# The final test set remains every Week 3 test row from May 2026. It is not used for feature selection, preprocessing fitting, hyperparameter tuning, or outlier-threshold decisions. `CloseDate`/`SaleMonth` are trace fields; only their target-independent engineered age and cyclical month representations enter the model.

# ## Fold-safe preprocessing and five expanding-month folds

# In[12]:


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=50,
                dtype=np.float32,
            ),
        ),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns),
    ],
    sparse_threshold=0.3,
)

ordered_months = sorted(train["SaleMonth"].dropna().unique().tolist())
if len(ordered_months) < CROSS_VALIDATION_FOLDS + 2:
    raise ValueError("Not enough training months for five expanding chronological folds.")

validation_month_blocks = np.array_split(
    np.array(ordered_months[2:], dtype=object),
    CROSS_VALIDATION_FOLDS,
)
cv_folds = []
cv_fold_rows = []

for fold_number, validation_months in enumerate(validation_month_blocks, start=1):
    validation_months = validation_months.tolist()
    first_validation_position = ordered_months.index(validation_months[0])
    fold_train_months = ordered_months[:first_validation_position]

    fold_train_indices = np.flatnonzero(
        train["SaleMonth"].isin(fold_train_months).to_numpy()
    )
    fold_validation_indices = np.flatnonzero(
        train["SaleMonth"].isin(validation_months).to_numpy()
    )
    cv_folds.append((fold_train_indices, fold_validation_indices))
    cv_fold_rows.append(
        {
            "fold": fold_number,
            "train_rows": len(fold_train_indices),
            "validation_rows": len(fold_validation_indices),
            "train_months": f"{fold_train_months[0]} to {fold_train_months[-1]}",
            "validation_months": f"{validation_months[0]} to {validation_months[-1]}",
            "shuffle": False,
            "random_state": RANDOM_STATE,
        }
    )

cv_fold_plan = pd.DataFrame(cv_fold_rows)
cv_fold_plan.to_csv(OUTPUT_DIR / "model_comparison_cv_fold_plan.csv", index=False)
display(cv_fold_plan)


# Each fold is an expanding window made of complete months, so a later month never helps encode or impute an earlier training fold. The preprocessor is cloned and fit inside each fold, then applied unchanged to that fold's validation months.

# ## Five-fold hyperparameter tuning

# In[13]:


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


candidate_specs = [
    {
        "family": "Linear Regression",
        "model": "Linear Regression (baseline)",
        "params": {"n_jobs": N_JOBS},
    },
    {
        "family": "Decision Tree",
        "model": "Decision Tree (depth 18, leaf 10)",
        "params": {
            "max_depth": 18,
            "min_samples_leaf": 10,
            "random_state": RANDOM_STATE,
        },
    },
    {
        "family": "Decision Tree",
        "model": "Decision Tree (depth 24, leaf 10)",
        "params": {
            "max_depth": 24,
            "min_samples_leaf": 10,
            "random_state": RANDOM_STATE,
        },
    },
    {
        "family": "Random Forest",
        "model": "Random Forest (40 trees, depth 24, leaf 10, max_features 0.4)",
        "params": {
            "n_estimators": 40,
            "max_depth": 24,
            "min_samples_leaf": 10,
            "max_features": 0.4,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
        },
    },
    {
        "family": "Random Forest",
        "model": "Random Forest (60 trees, depth 30, leaf 10, max_features 0.7)",
        "params": {
            "n_estimators": 60,
            "max_depth": 30,
            "min_samples_leaf": 10,
            "max_features": 0.7,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
        },
    },
]


def build_estimator(spec):
    if spec["family"] == "Linear Regression":
        return LinearRegression(**spec["params"])
    if spec["family"] == "Decision Tree":
        return DecisionTreeRegressor(**spec["params"])
    if spec["family"] == "Random Forest":
        return RandomForestRegressor(**spec["params"])
    raise ValueError(f"Unknown family: {spec['family']}")


cv_detail_rows = []
cv_start = time.time()

for fold_number, (fold_train_indices, fold_validation_indices) in enumerate(cv_folds, start=1):
    fold_preprocessor = clone(preprocessor)
    fold_X_train = fold_preprocessor.fit_transform(X_train.iloc[fold_train_indices])
    fold_X_validation = fold_preprocessor.transform(X_train.iloc[fold_validation_indices])
    fold_y_train = y_train[fold_train_indices]
    fold_y_validation = y_train[fold_validation_indices]

    for spec in candidate_specs:
        fit_start = time.time()
        estimator = build_estimator(spec)
        estimator.fit(fold_X_train, fold_y_train)
        fold_predictions = estimator.predict(fold_X_validation)
        metrics = regression_metrics(fold_y_validation, fold_predictions)
        cv_detail_rows.append(
            {
                "fold": fold_number,
                "family": spec["family"],
                "model": spec["model"],
                "train_rows": len(fold_train_indices),
                "validation_rows": len(fold_validation_indices),
                "encoded_feature_count": fold_X_train.shape[1],
                "validation_r2": metrics["r2"],
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_mape": metrics["mape"],
                "validation_mdape": metrics["mdape"],
                "fit_seconds": time.time() - fit_start,
                "params": json.dumps(spec["params"], sort_keys=True),
            }
        )

    del fold_preprocessor, fold_X_train, fold_X_validation
    gc.collect()
    print(f"Completed chronological CV fold {fold_number}/{CROSS_VALIDATION_FOLDS}.")

cv_detail = pd.DataFrame(cv_detail_rows)
cv_detail.to_csv(
    OUTPUT_DIR / "model_comparison_cv_detail_results.csv",
    index=False,
)

validation_results = (
    cv_detail.groupby(["family", "model", "params"], as_index=False)
    .agg(
        cv_folds=("fold", "nunique"),
        mean_validation_r2=("validation_r2", "mean"),
        std_validation_r2=("validation_r2", "std"),
        mean_validation_mae=("validation_mae", "mean"),
        mean_validation_rmse=("validation_rmse", "mean"),
        mean_validation_mape=("validation_mape", "mean"),
        mean_validation_mdape=("validation_mdape", "mean"),
        total_fit_seconds=("fit_seconds", "sum"),
    )
)
validation_results.to_csv(
    OUTPUT_DIR / "model_comparison_validation_results.csv",
    index=False,
)

selected_tree = (
    validation_results.loc[validation_results["family"].eq("Decision Tree")]
    .sort_values(
        ["mean_validation_mdape", "mean_validation_r2"],
        ascending=[True, False],
    )
    .iloc[0]
)
selected_forest = (
    validation_results.loc[validation_results["family"].eq("Random Forest")]
    .sort_values(
        ["mean_validation_mdape", "mean_validation_r2"],
        ascending=[True, False],
    )
    .iloc[0]
)

display(
    validation_results.drop(columns="params").sort_values(
        ["family", "mean_validation_mdape"]
    )
)
print(f"Selected Decision Tree: {selected_tree['model']}")
print(f"Selected Random Forest: {selected_forest['model']}")
print(f"Five-fold tuning seconds: {time.time() - cv_start:,.1f}")


# MdAPE is the primary selection metric because the price distribution is multi-scale and right-skewed; mean validation R2 is the tie-breaker. Linear Regression has no tree hyperparameters, but it is cross-validated in the same folds as a stability reference.

# ## Final standardized May 2026 comparison

# In[14]:


final_preprocessor = clone(preprocessor)
preprocess_start = time.time()
X_train_processed = final_preprocessor.fit_transform(X_train)
X_test_processed = final_preprocessor.transform(X_test)
preprocess_seconds = time.time() - preprocess_start
encoded_feature_names = final_preprocessor.get_feature_names_out()
encoded_feature_count = X_train_processed.shape[1]

selected_tree_params = json.loads(selected_tree["params"])
selected_forest_params = json.loads(selected_forest["params"])

final_specs = [
    {
        "family": "Linear Regression",
        "model": "Linear Regression (baseline)",
        "estimator": LinearRegression(n_jobs=N_JOBS),
        "params": {"n_jobs": N_JOBS},
        "prediction_column": "linear_regression_prediction",
    },
    {
        "family": "Decision Tree",
        "model": selected_tree["model"],
        "estimator": DecisionTreeRegressor(**selected_tree_params),
        "params": selected_tree_params,
        "prediction_column": "decision_tree_prediction",
    },
    {
        "family": "Random Forest",
        "model": selected_forest["model"],
        "estimator": RandomForestRegressor(**selected_forest_params),
        "params": selected_forest_params,
        "prediction_column": "random_forest_prediction",
    },
]

comparison_rows = []
fitted_models = []
predictions = test[["CloseDate", "SaleMonth", "ClosePrice"]].copy()
predictions.insert(0, "source_row_index", test.index)

for spec in final_specs:
    fit_start = time.time()
    estimator = spec["estimator"]
    estimator.fit(X_train_processed, y_train)
    fit_seconds = time.time() - fit_start

    train_predictions = estimator.predict(X_train_processed)
    test_predictions = estimator.predict(X_test_processed)
    train_metrics = regression_metrics(y_train, train_predictions)
    test_metrics = regression_metrics(y_test, test_predictions)

    predictions[spec["prediction_column"]] = test_predictions
    predictions[
        spec["prediction_column"].replace("prediction", "absolute_error")
    ] = np.abs(y_test - test_predictions)
    predictions[
        spec["prediction_column"].replace(
            "prediction", "absolute_percentage_error"
        )
    ] = np.abs(y_test - test_predictions) / y_test

    comparison_rows.append(
        {
            "model": spec["model"],
            "family": spec["family"],
            "experiment": EXPERIMENT,
            "train_rows": len(train),
            "test_rows": len(test),
            "feature_count": encoded_feature_count,
            "train_months": f"{train['SaleMonth'].min()} to {train['SaleMonth'].max()}",
            "test_month": TEST_MONTH,
            "train_r2": train_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_median_absolute_error": test_metrics["median_absolute_error"],
            "test_mape": test_metrics["mape"],
            "test_mdape": test_metrics["mdape"],
            "fit_seconds": fit_seconds,
            "selected_parameters": json.dumps(spec["params"], sort_keys=True),
        }
    )
    fitted_models.append((spec, estimator))

comparison_results = pd.DataFrame(comparison_rows)
comparison_results.to_csv(
    OUTPUT_DIR / "model_comparison_results.csv",
    index=False,
)
predictions.to_csv(
    OUTPUT_DIR / "model_comparison_test_predictions.csv",
    index=False,
)
pd.DataFrame({"feature": encoded_feature_names}).to_csv(
    OUTPUT_DIR / "model_comparison_encoded_feature_names.csv",
    index=False,
)

display(
    comparison_results[
        [
            "model",
            "train_r2",
            "test_r2",
            "test_mae",
            "test_rmse",
            "test_mape",
            "test_mdape",
            "fit_seconds",
        ]
    ]
)
print(f"Fold-fitted encoded feature count: {encoded_feature_count:,}")
print(f"Final preprocessing seconds: {preprocess_seconds:,.1f}")


# ## Error breakdown, feature signals, and Week 5 comparison

# In[15]:


price_band_rows = []
band_frame = predictions.copy()
band_frame["price_band"] = pd.qcut(
    band_frame["ClosePrice"],
    q=5,
    duplicates="drop",
)

for spec, _ in fitted_models:
    prediction_column = spec["prediction_column"]
    ape_column = prediction_column.replace(
        "prediction", "absolute_percentage_error"
    )
    error_column = prediction_column.replace("prediction", "absolute_error")

    for price_band, group in band_frame.groupby("price_band", observed=True):
        price_band_rows.append(
            {
                "model": spec["model"],
                "price_band": str(price_band),
                "rows": len(group),
                "mean_close_price": group["ClosePrice"].mean(),
                "test_r2": (
                    r2_score(group["ClosePrice"], group[prediction_column])
                    if len(group) > 1
                    else np.nan
                ),
                "mae": group[error_column].mean(),
                "mape": group[ape_column].mean(),
                "mdape": group[ape_column].median(),
            }
        )

price_band_metrics = pd.DataFrame(price_band_rows)
price_band_metrics.to_csv(
    OUTPUT_DIR / "model_comparison_price_band_metrics.csv",
    index=False,
)

importance_rows = []
for spec, estimator in fitted_models:
    if spec["family"] == "Linear Regression":
        values = np.asarray(estimator.coef_)
        ranking = np.argsort(np.abs(values))[::-1][:25]
        importance_type = "coefficient"
    else:
        values = np.asarray(estimator.feature_importances_)
        ranking = np.argsort(values)[::-1][:25]
        importance_type = "feature_importance"

    for rank, feature_index in enumerate(ranking, start=1):
        importance_rows.append(
            {
                "model": spec["model"],
                "rank": rank,
                "feature": encoded_feature_names[feature_index],
                "importance_type": importance_type,
                "value": float(values[feature_index]),
            }
        )

feature_importance = pd.DataFrame(importance_rows)
feature_importance.to_csv(
    OUTPUT_DIR / "model_comparison_feature_importance.csv",
    index=False,
)

performance_change = pd.DataFrame()
if WEEK5_RESULTS.exists():
    week5_results = pd.read_csv(WEEK5_RESULTS)
    week5_metrics = week5_results[
        ["family", "test_r2", "test_mape", "test_mdape"]
    ].rename(
        columns={
            "test_r2": "week5_test_r2",
            "test_mape": "week5_test_mape",
            "test_mdape": "week5_test_mdape",
        }
    )
    week6_metrics = comparison_results[
        ["family", "test_r2", "test_mape", "test_mdape"]
    ].rename(
        columns={
            "test_r2": "week6_test_r2",
            "test_mape": "week6_test_mape",
            "test_mdape": "week6_test_mdape",
        }
    )
    performance_change = week5_metrics.merge(week6_metrics, on="family", how="inner")
    performance_change["test_r2_change"] = (
        performance_change["week6_test_r2"]
        - performance_change["week5_test_r2"]
    )
    performance_change["test_mape_change"] = (
        performance_change["week6_test_mape"]
        - performance_change["week5_test_mape"]
    )
    performance_change["test_mdape_change"] = (
        performance_change["week6_test_mdape"]
        - performance_change["week5_test_mdape"]
    )
    performance_change.to_csv(
        OUTPUT_DIR / "week5_vs_week6_performance_change.csv",
        index=False,
    )

display(price_band_metrics.head(15))
display(feature_importance.groupby("model").head(8))
if not performance_change.empty:
    display(performance_change)


# ## Week 6 findings report

# In[16]:


def pct(value):
    return f"{value:.1%}"


display_table = comparison_results[
    [
        "model",
        "train_r2",
        "test_r2",
        "test_mae",
        "test_rmse",
        "test_mape",
        "test_mdape",
        "fit_seconds",
    ]
].copy()
best_by_r2 = comparison_results.sort_values("test_r2", ascending=False).iloc[0]
best_by_mdape = comparison_results.sort_values("test_mdape", ascending=True).iloc[0]

report_lines = [
    "Week 6 Feature Engineering Summary Report",
    "=========================================",
    "",
    TEAM,
    "Notebook: 05_feature_engineering.ipynb",
    "Task: Week 6 - Feature Engineering",
    "",
    "Standardized evaluation controls",
    "--------------------------------",
    f"Train months: {train['SaleMonth'].min()} to {train['SaleMonth'].max()}",
    f"Test month: {TEST_MONTH}",
    f"Rows evaluated: {len(train):,} train, {len(test):,} test",
    f"Fold-fitted encoded feature count: {encoded_feature_count:,}",
    "Excluded leakage fields: ListPrice, OriginalListPrice, DaysOnMarket, PurchaseContractDate.",
    "CloseDate and SaleMonth were used only for split/reporting and target-independent feature engineering.",
    "No test rows were removed for the headline evaluation.",
    "",
    "Feature engineering rationale",
    "-----------------------------",
    "- BedBathRatio: represents functional balance between sleeping capacity and bathroom utility; safe division and a broad cap control invalid ratios.",
    "- PropertyAgeAtSale: measures condition-relevant aging at the valuation date instead of treating YearBuilt as timeless.",
    "- SaleMonthSin/SaleMonthCos: capture cyclical seasonality while keeping December adjacent to January.",
    "- LogLivingArea: adds a diminishing-return representation of highly skewed living area while retaining the original field.",
    "- LogLotSizeSquareFeet: adds a diminishing-return representation of highly skewed lot size while retaining the original field.",
    "- DistrictName: comes from a target-independent point-in-polygon join after filtering the CDE 2025-26 layer to DistrictType = Unified.",
    "- SchoolDistrictBoundaryMatched: distinguishes matched Unified polygons from properties outside Unified-only coverage or with unusable coordinates.",
    "",
    "Reproducibility controls",
    "------------------------",
    f"Global Python/NumPy/model seed: {RANDOM_STATE}.",
    f"Cross-validation: {CROSS_VALIDATION_FOLDS} expanding complete-month folds, shuffle=False.",
    "All imputation, scaling, and categorical encoding were fitted inside each cross-validation fold.",
    "Week 3 missing-indicator flags were used to restore pre-imputation missingness before Week 6 feature engineering.",
    "Week 3 rule-based cleaning and feature caps remain inherited so this update changes preprocessing/feature engineering without redefining the historical row set.",
    f"Model workers: {N_JOBS}. Random Forest and split behavior remain fixed by random_state.",
    "Pinned package versions are recorded in requirements.txt.",
    "",
    "Spatial join quality",
    "--------------------",
    "Boundary source: https://data.ca.gov/dataset/california-school-district-areas-2025-26.",
    f"CDE polygons loaded: {len(districts):,}; Unified polygons used: {len(unified_districts):,}.",
    f"Properties matched to a Unified district: {df['SchoolDistrictBoundaryMatched'].sum():,} of {len(df):,}.",
    f"Properties retained as No boundary match: {(1 - df['SchoolDistrictBoundaryMatched']).sum():,}.",
    f"Unique matched Unified DistrictName values: {df.loc[df['SchoolDistrictBoundaryMatched'].eq(1), 'DistrictName'].nunique():,}.",
    "",
    "Cross-validation fold plan",
    "--------------------------",
    cv_fold_plan.to_string(index=False),
    "",
    "Five-fold tuning results",
    "------------------------",
    validation_results.drop(columns="params")
    .sort_values(["family", "mean_validation_mdape"])
    .to_string(index=False),
    "",
    "Final model comparison",
    "----------------------",
    display_table.to_string(index=False),
]

if not performance_change.empty:
    report_lines.extend(
        [
            "",
            "Week 5 to Week 6 test change",
            "----------------------------",
            performance_change.to_string(index=False),
        ]
    )

report_lines.extend(
    [
        "",
        "Interpretation",
        "--------------",
        f"Best test R2: {best_by_r2['model']} at {best_by_r2['test_r2']:.3f}.",
        f"Best typical percentage error: {best_by_mdape['model']} at {pct(best_by_mdape['test_mdape'])} MdAPE.",
        "The feature-engineered experiment adds intrinsic relationships, a cyclical valuation-time signal, and an authoritative school-district spatial category without using sale-process fields.",
        "Linear Regression benefits most when the engineered ratios/log transforms make nonlinear physical relationships easier to express.",
        "Decision Tree and Random Forest can use the joined Unified DistrictName and its interactions directly, while five-fold chronological tuning reduces dependence on a single favorable validation month.",
        "The school-district feature is a demand/location proxy, not a claim that district boundaries alone cause value; localized error checks should continue in later weeks.",
    ]
)

report_text = "\n".join(report_lines)
report_path = OUTPUT_DIR / "Week6_Feature_Engineering_Summary_Report.txt"
report_path.write_text(report_text, encoding="utf-8")
print(report_text)


# ## Completion checklist
# 
# - Updated Week 4-5 preprocessing/model-comparison structure saved as a new Week 6 notebook.
# - Required bed/bath ratio and property-age-at-sale features included with rationale.
# - CDE 2025-26 boundaries filtered to Unified districts, spatially joined, and saved as `DistrictName`.
# - Old versus feature-engineered columns displayed and saved.
# - `df.head()` displayed after feature engineering.
# - Updated feature-set CSV saved.
# - Five complete-month chronological folds used for tuning.
# - Linear Regression, Decision Tree, and Random Forest retrained on all training months.
# - Final results use the unchanged full May 2026 test set and Week 5 table layout, without a data-source path column.
# - Findings text report and pinned `requirements.txt` included in the W6 folder.
