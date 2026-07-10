from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRACE_COLUMNS = ["ListingKey", "ListingId", "CloseDate", "SourceMonth", "Split", "ClosePrice"]
LEAKAGE_COLUMNS = ["ListPrice", "OriginalListPrice"]
MODEL_NAME = "Linear Regression (ordinary least squares)"


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "W3 Data Preprocessing").exists() and (candidate / "raw data").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root with W3 Data Preprocessing and raw data folders.")


def find_cleaned_csv(project_root: Path) -> Path:
    candidates = [
        project_root / "W3 Data Preprocessing" / "cleaned_crmls_sfr_train_test.csv",
        project_root / "W3 Data Preprocessing" / "Cleaned SFR CRMLSSold CSVs" / "cleaned_crmls_sfr_train_test.csv",
        project_root / "W3 Data Preprocessing" / "cleaned_crmls_sfr_train_test.csv.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the Week 3 cleaned train/test CSV.")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total_sum_squares = np.sum((y_true - y_true.mean()) ** 2)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    return float(1 - residual_sum_squares / total_sum_squares)


def add_intercept(matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(matrix.shape[0]), matrix])


def fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, int]:
    design_matrix = add_intercept(x_train)
    coefficients, _, rank, _ = np.linalg.lstsq(design_matrix, y_train, rcond=None)
    return coefficients, int(rank)


def predict_linear_regression(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ coefficients


def dollars(value: float) -> str:
    return f"${value:,.0f}"


def build_report(metrics: dict[str, object], top_coefficients: pd.DataFrame) -> str:
    formatted_coefficients = top_coefficients.copy()
    formatted_coefficients["coefficient"] = formatted_coefficients["coefficient"].map("{:,.4f}".format)

    return "\n".join(
        [
            "Week 4 Baseline Model Summary Report",
            "====================================",
            "",
            f"Model: {metrics['model']}",
            f"Training data: {metrics['train_rows']:,} rows from {metrics['train_months']}",
            f"Test data: {metrics['test_rows']:,} rows from {metrics['test_month']}",
            f"Predictor count: {metrics['feature_count']}",
            "",
            "Baseline performance",
            "--------------------",
            f"Test R2: {metrics['test_r2']:.4f}",
            f"Train R2: {metrics['train_r2']:.4f}",
            f"Test MAE: {dollars(float(metrics['test_mae']))}",
            f"Test RMSE: {dollars(float(metrics['test_rmse']))}",
            f"Test median absolute error: {dollars(float(metrics['test_median_absolute_error']))}",
            "",
            "Interpretation",
            "--------------",
            (
                "The first linear regression baseline explains "
                f"{float(metrics['test_r2']) * 100:.1f}% of ClosePrice variance on the holdout month. "
                "This is a useful starting line, but the error levels are still large for property-level "
                "price prediction."
            ),
            (
                "The Week 3 preprocessing kept listing-price leakage fields out of the modeling matrix. "
                "The next model should compare against this R2 while exploring stronger nonlinear models, "
                "target/outlier treatment, and richer location features."
            ),
            "",
            "Largest absolute coefficients",
            "-----------------------------",
            formatted_coefficients.to_string(index=False),
            "",
        ]
    )


def main() -> None:
    project_root = find_project_root(Path(__file__).resolve().parent)
    output_dir = project_root / "W4 Baseline Model"
    cleaned_csv_path = find_cleaned_csv(project_root)

    data = pd.read_csv(cleaned_csv_path)
    missing_required = sorted(set(TRACE_COLUMNS).difference(data.columns))
    if missing_required:
        raise ValueError(f"Missing required cleaned-data columns: {missing_required}")

    leakage_present = sorted(set(LEAKAGE_COLUMNS).intersection(data.columns))
    if leakage_present:
        raise ValueError(f"Leakage columns should not be present in the modeling data: {leakage_present}")

    feature_columns = [column for column in data.columns if column not in TRACE_COLUMNS]
    non_numeric_features = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric_features:
        raise ValueError(f"All Week 4 predictors must be numeric. Non-numeric columns: {non_numeric_features}")

    missing_feature_values = int(data[feature_columns].isna().sum().sum())
    if missing_feature_values:
        raise ValueError(f"Feature matrix contains {missing_feature_values:,} missing values.")

    train = data.loc[data["Split"].eq("train")].copy()
    test = data.loc[data["Split"].eq("test")].copy()
    if train.empty or test.empty:
        raise ValueError("The cleaned data must contain both train and test split rows.")

    x_train = train[feature_columns].to_numpy(dtype=np.float64)
    y_train = train["ClosePrice"].to_numpy(dtype=np.float64)
    x_test = test[feature_columns].to_numpy(dtype=np.float64)
    y_test = test["ClosePrice"].to_numpy(dtype=np.float64)

    coefficients, matrix_rank = fit_linear_regression(x_train, y_train)
    train_predictions = predict_linear_regression(x_train, coefficients)
    test_predictions = predict_linear_regression(x_test, coefficients)

    train_mean_predictions = np.full_like(y_test, y_train.mean(), dtype=np.float64)
    metrics = {
        "model": MODEL_NAME,
        "data_source": str(cleaned_csv_path),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_columns)),
        "train_months": f"{train['SourceMonth'].min()}-{train['SourceMonth'].max()}",
        "test_month": str(test["SourceMonth"].iloc[0]),
        "train_r2": r2_score(y_train, train_predictions),
        "test_r2": r2_score(y_test, test_predictions),
        "train_mean_baseline_test_r2": r2_score(y_test, train_mean_predictions),
        "test_mae": float(np.mean(np.abs(y_test - test_predictions))),
        "test_rmse": float(np.sqrt(np.mean((y_test - test_predictions) ** 2))),
        "test_median_absolute_error": float(np.median(np.abs(y_test - test_predictions))),
        "test_mean_actual_close_price": float(y_test.mean()),
        "test_median_actual_close_price": float(np.median(y_test)),
        "test_mean_predicted_close_price": float(test_predictions.mean()),
        "test_median_predicted_close_price": float(np.median(test_predictions)),
        "design_matrix_rank": matrix_rank,
        "design_matrix_columns_including_intercept": int(x_train.shape[1] + 1),
    }

    coefficients_table = pd.DataFrame(
        {
            "feature": ["intercept", *feature_columns],
            "coefficient": coefficients,
        }
    )
    top_coefficients = (
        coefficients_table.loc[coefficients_table["feature"].ne("intercept")]
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .head(15)
        .drop(columns="abs_coefficient")
        .reset_index(drop=True)
    )

    predictions = test[["ListingKey", "ListingId", "CloseDate", "SourceMonth", "ClosePrice"]].copy()
    predictions["PredictedClosePrice"] = test_predictions
    predictions["Residual"] = predictions["ClosePrice"] - predictions["PredictedClosePrice"]
    predictions["AbsoluteError"] = predictions["Residual"].abs()

    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "baseline_model_results.csv"
    coefficients_path = output_dir / "baseline_model_top_coefficients.csv"
    predictions_path = output_dir / "baseline_model_test_predictions.csv"
    report_path = output_dir / "Week4_Baseline_Model_Summary_Report.txt"

    pd.DataFrame([metrics]).to_csv(results_path, index=False)
    top_coefficients.to_csv(coefficients_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    report_path.write_text(build_report(metrics, top_coefficients), encoding="utf-8")

    print("Week 4 baseline complete.")
    print(f"Cleaned data: {cleaned_csv_path}")
    print(f"Results: {results_path}")
    print(f"Summary report: {report_path}")
    print(f"Test R2: {metrics['test_r2']:.4f}")


if __name__ == "__main__":
    main()
