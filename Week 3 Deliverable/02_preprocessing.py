from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "W3 Data Preprocessing"
CLEANED_CSV_DIR = OUTPUT_DIR / "Cleaned SFR CRMLSSold CSVs"

EXPECTED_MONTHS = [
    "202505",
    "202506",
    "202507",
    "202508",
    "202509",
    "202510",
    "202511",
    "202512",
    "202601",
    "202602",
    "202603",
    "202604",
    "202605",
]

REQUIRED_COLUMNS = [
    "ListingKey",
    "ListingId",
    "CloseDate",
    "ClosePrice",
    "PropertyType",
    "PropertySubType",
    "MlsStatus",
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
    "LotSizeAcres",
    "LotSizeArea",
    "DaysOnMarket",
    "City",
    "CountyOrParish",
    "PostalCode",
    "Latitude",
    "Longitude",
    "YearBuilt",
    "FireplacesTotal",
    "GarageSpaces",
    "ParkingTotal",
    "AssociationFee",
    "NewConstructionYN",
    "PoolPrivateYN",
    "ViewYN",
    "WaterfrontYN",
]

NUMERIC_FEATURES = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeUsedSqFt",
    "DaysOnMarket",
    "YearBuilt",
    "Latitude",
    "Longitude",
    "GarageSpaces",
    "ParkingTotal",
    "AssociationFee",
]

SKEWED_NUMERIC_FEATURES = {"LotSizeUsedSqFt", "DaysOnMarket", "AssociationFee"}
BOOLEAN_FEATURES = ["NewConstructionYN", "PoolPrivateYN", "ViewYN", "WaterfrontYN"]
CATEGORICAL_FEATURES = ["CountyOrParish", "PostalCode"]
LEAKAGE_COLUMNS = ["ListPrice", "OriginalListPrice"]
CHOSEN_TRAINING_WINDOW_MONTHS = 12


def find_crmls_files(expected_months: list[str] = EXPECTED_MONTHS) -> list[Path]:
    raw_dir = PROJECT_ROOT / "raw data"
    month_to_file = {}
    for file_path in sorted(raw_dir.glob("CRMLSSold*.csv")):
        month_match = re.search(r"(\d{6})", file_path.stem)
        if month_match and month_match.group(1) in expected_months:
            month_to_file[month_match.group(1)] = file_path

    missing = [month for month in expected_months if month not in month_to_file]
    if missing:
        raise FileNotFoundError(f"Missing expected CRMLS sold files for: {missing}")

    return [month_to_file[month] for month in expected_months]


def month_from_path(path: Path) -> str:
    month_match = re.search(r"(\d{6})", path.stem)
    if not month_match:
        raise ValueError(f"Could not parse source month from {path.name}")
    return month_match.group(1)


def read_month(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(REQUIRED_COLUMNS).difference(header))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    frame = pd.read_csv(
        path,
        usecols=REQUIRED_COLUMNS,
        dtype={
            "ListingKey": "string",
            "ListingId": "string",
            "PostalCode": "string",
            "City": "string",
            "CountyOrParish": "string",
            "PropertyType": "string",
            "PropertySubType": "string",
            "MlsStatus": "string",
        },
        low_memory=False,
    )
    frame["SourceFile"] = path.name
    frame["SourceMonth"] = month_from_path(path)
    return frame


def load_filtered_data() -> pd.DataFrame:
    raw = pd.concat([read_month(path) for path in find_crmls_files()], ignore_index=True)

    numeric_columns = [
        "ClosePrice",
        "LivingArea",
        "BedroomsTotal",
        "BathroomsTotalInteger",
        "LotSizeSquareFeet",
        "LotSizeAcres",
        "LotSizeArea",
        "DaysOnMarket",
        "Latitude",
        "Longitude",
        "YearBuilt",
        "FireplacesTotal",
        "GarageSpaces",
        "ParkingTotal",
        "AssociationFee",
    ]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw["CloseDate"] = pd.to_datetime(raw["CloseDate"], errors="coerce")
    raw["CloseMonth"] = raw["CloseDate"].dt.to_period("M").astype("string")
    raw["LotSizeUsedSqFt"] = raw["LotSizeSquareFeet"].where(
        raw["LotSizeSquareFeet"].gt(0),
        raw["LotSizeAcres"] * 43_560,
    )
    raw["PropertyFilterMatch"] = raw["PropertyType"].eq("Residential") & raw["PropertySubType"].eq(
        "SingleFamilyResidence"
    )

    return raw.loc[
        raw["PropertyFilterMatch"] & raw["ClosePrice"].notna() & raw["ClosePrice"].gt(0)
    ].copy()


def sanitize_feature_values(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.loc[cleaned["LivingArea"].le(0), "LivingArea"] = np.nan
    cleaned.loc[cleaned["BedroomsTotal"].le(0), "BedroomsTotal"] = np.nan
    cleaned.loc[cleaned["BathroomsTotalInteger"].le(0), "BathroomsTotalInteger"] = np.nan
    cleaned.loc[cleaned["LotSizeUsedSqFt"].le(0), "LotSizeUsedSqFt"] = np.nan
    cleaned.loc[cleaned["DaysOnMarket"].lt(0), "DaysOnMarket"] = np.nan
    cleaned.loc[~cleaned["Latitude"].between(32, 42.5), "Latitude"] = np.nan
    cleaned.loc[~cleaned["Longitude"].between(-125, -113), "Longitude"] = np.nan
    cleaned.loc[cleaned["GarageSpaces"].lt(0) | cleaned["GarageSpaces"].gt(20), "GarageSpaces"] = np.nan
    cleaned.loc[cleaned["ParkingTotal"].lt(0) | cleaned["ParkingTotal"].gt(30), "ParkingTotal"] = np.nan
    return cleaned


def build_window_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    available_months = sorted(frame["SourceMonth"].dropna().unique().tolist())
    test_month = available_months[-1]
    test = frame.loc[frame["SourceMonth"].eq(test_month)]
    rows = []

    for training_window_months in [3, 6, 9, 12]:
        if training_window_months > len(available_months) - 1:
            continue

        train_months = available_months[-1 - training_window_months : -1]
        train = frame.loc[frame["SourceMonth"].isin(train_months)]
        iqr_train = train["ClosePrice"].quantile(0.75) - train["ClosePrice"].quantile(0.25)
        iqr_test = test["ClosePrice"].quantile(0.75) - test["ClosePrice"].quantile(0.25)

        rows.append(
            {
                "training_window_months": training_window_months,
                "train_months": f"{train_months[0]}-{train_months[-1]}",
                "test_month": test_month,
                "train_rows": len(train),
                "test_rows": len(test),
                "median_price_pct_diff_vs_test": (
                    (train["ClosePrice"].median() - test["ClosePrice"].median())
                    / test["ClosePrice"].median()
                    * 100
                ),
                "iqr_pct_diff_vs_test": (iqr_train - iqr_test) / iqr_test * 100,
                "county_coverage_pct": test["CountyOrParish"].isin(set(train["CountyOrParish"].dropna())).mean()
                * 100,
                "postal_code_coverage_pct": test["PostalCode"].isin(set(train["PostalCode"].dropna())).mean()
                * 100,
                "avg_feature_missing_pct": train[
                    NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
                ].isna().mean().mean()
                * 100,
            }
        )

    train_months = available_months[-1 - CHOSEN_TRAINING_WINDOW_MONTHS : -1]
    return pd.DataFrame(rows), train_months, test_month


def bool_to_numeric(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(
        {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "1": 1,
            "0": 0,
        }
    )


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return slug or "missing"


def preprocess_for_modeling(frame: pd.DataFrame, train_months: list[str], test_month: str) -> tuple[pd.DataFrame, list[str]]:
    modeling = frame.loc[frame["SourceMonth"].isin([*train_months, test_month])].copy()
    modeling["Split"] = np.where(modeling["SourceMonth"].eq(test_month), "test", "train")
    train_mask = modeling["Split"].eq("train")
    output = modeling[["ListingKey", "ListingId", "CloseDate", "SourceMonth", "Split", "ClosePrice"]].copy()
    feature_columns = []

    for feature in NUMERIC_FEATURES:
        missing_col = f"{feature}_missing"
        output[missing_col] = modeling[feature].isna().astype("int8")
        feature_columns.append(missing_col)

        median_value = modeling.loc[train_mask, feature].median()
        imputed = modeling[feature].fillna(0 if pd.isna(median_value) else median_value)

        if feature in SKEWED_NUMERIC_FEATURES:
            transformed = np.log1p(imputed.clip(lower=0))
            feature_col = f"{feature}_log_z"
        else:
            transformed = imputed
            feature_col = f"{feature}_z"

        mean_value = transformed.loc[train_mask].mean()
        std_value = transformed.loc[train_mask].std(ddof=0)
        if pd.isna(std_value) or std_value == 0:
            std_value = 1
        output[feature_col] = ((transformed - mean_value) / std_value).round(6)
        feature_columns.append(feature_col)

    for feature in BOOLEAN_FEATURES:
        numeric_values = bool_to_numeric(modeling[feature])
        missing_col = f"{feature}_missing"
        encoded_col = f"{feature}_flag"
        output[missing_col] = numeric_values.isna().astype("int8")
        output[encoded_col] = numeric_values.fillna(0).astype("int8")
        feature_columns.extend([missing_col, encoded_col])

    county_values = modeling["CountyOrParish"].astype("string").str.strip()
    county_values = county_values.mask(county_values.eq(""))
    output["CountyOrParish_missing"] = county_values.isna().astype("int8")
    feature_columns.append("CountyOrParish_missing")

    for county in sorted(county_values.loc[train_mask].dropna().unique().tolist()):
        county_col = f"CountyOrParish__{slugify(county)}"
        output[county_col] = county_values.eq(county).astype("int8")
        feature_columns.append(county_col)

    postal_values = modeling["PostalCode"].astype("string").str.strip()
    postal_values = postal_values.mask(postal_values.eq(""))
    postal_train_frequency = postal_values.loc[train_mask].fillna("Missing").value_counts(normalize=True)
    output["PostalCode_missing"] = postal_values.isna().astype("int8")
    output["PostalCode_train_frequency"] = postal_values.fillna("Missing").map(postal_train_frequency).fillna(0).round(8)
    feature_columns.extend(["PostalCode_missing", "PostalCode_train_frequency"])

    output = output.sort_values(["SourceMonth", "ListingKey"], kind="mergesort").reset_index(drop=True)
    return output, feature_columns


def main() -> None:
    if set(LEAKAGE_COLUMNS).intersection(NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES):
        raise ValueError("Leakage columns should not be used as model features.")

    filtered = load_filtered_data()
    sanitized = sanitize_feature_values(filtered)
    diagnostics, train_months, test_month = build_window_diagnostics(sanitized)
    cleaned, feature_columns = preprocess_for_modeling(sanitized, train_months, test_month)

    CLEANED_CSV_DIR.mkdir(exist_ok=True)
    cleaned_csv_path = CLEANED_CSV_DIR / "cleaned_crmls_sfr_train_test.csv"
    cleaned.to_csv(cleaned_csv_path, index=False)

    summary = {
        "cleaned_csv": str(cleaned_csv_path),
        "rows": int(len(cleaned)),
        "columns": int(cleaned.shape[1]),
        "train_rows": int(cleaned["Split"].eq("train").sum()),
        "test_rows": int(cleaned["Split"].eq("test").sum()),
        "feature_columns": len(feature_columns),
        "missing_feature_values": int(cleaned[feature_columns].isna().sum().sum()),
        "non_numeric_feature_columns": [
            column for column in feature_columns if not pd.api.types.is_numeric_dtype(cleaned[column])
        ],
        "target_missing": int(cleaned["ClosePrice"].isna().sum()),
        "chosen_training_window_months": CHOSEN_TRAINING_WINDOW_MONTHS,
        "train_months": f"{train_months[0]}-{train_months[-1]}",
        "test_month": test_month,
        "window_diagnostics": diagnostics.round(4).to_dict(orient="records"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
