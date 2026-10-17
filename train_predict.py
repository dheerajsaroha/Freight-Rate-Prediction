from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

TARGET = "posted_rate"
ID_COL = "load_id"

RANDOM_STATE = 42
N_SPLITS = 5
USE_CHRONOLOGICAL_CV = True

# Target transforms compared during CV; best by SELECTION_METRIC is used finally.
TARGET_TRANSFORMS = ["log1p", "identity"]
SELECTION_METRIC = "RMSE"  # "MAE", "RMSE", or "R2"

# Hyperparameter grid for optimization (lower MAE/RMSE, higher R2).
PARAM_GRID = {
    "max_iter": [400, 600],
    "learning_rate": [0.03, 0.05],
    "max_depth": [6, 8],
    "l2_regularization": [0.5, 1.0],
}
# min_samples_leaf fixed; keep grid small for runtime.
MIN_SAMPLES_LEAF = 10

NUMERIC_FEATURES = [
    "distance",
    "weight_clean",
    "market_index_clean",
    "quote_signal_clean",
    "log_distance",
    "log_weight",
    "dist_per_weight",
    "mi_times_qs",
    "month",
    "day_of_week",
    "is_weekend",
    "day_of_year",
    "doy_sin",
    "doy_cos",
    "weight_missing",
    "market_index_missing",
    "quote_signal_missing",
]

CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]


def find_file(candidates: list[Path], description: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n{checked}")


def find_input_files() -> tuple[Path, Path, Path]:
    train_path = find_file(
        [DATA / "train_test.csv", DATA / "train-test.csv"], "training dataset"
    )
    validation_path = find_file([DATA / "validation.csv"], "validation dataset")
    december_path = find_file(
        [DATA / "december_chart_inputs.csv", DATA / "december-chart-inputs.csv"],
        "December chart input dataset",
    )
    return train_path, validation_path, december_path


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def validate_required_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def validate_training_data(df: pd.DataFrame) -> None:
    validate_required_columns(
        df, [ID_COL, "date", "pickup", "delivery", "distance", "equipment", "weight", TARGET],
        "Training dataset",
    )
    if df.empty:
        raise ValueError("Training dataset is empty.")
    if df[ID_COL].isna().any():
        raise ValueError("Training data contains missing load_id values.")
    if df[ID_COL].duplicated().any():
        raise ValueError("Training data contains duplicate load_id values.")
    target = pd.to_numeric(df[TARGET], errors="coerce")
    if target.isna().any() or not np.isfinite(target).all():
        raise ValueError("Training target contains invalid numeric values.")
    if (target <= 0).any():
        raise ValueError("All posted_rate values must be positive for log1p training.")
    for col in ["distance", "weight", "market_index", "quote_signal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if (df["distance"].dropna() < 0).any():
        raise ValueError("Distance cannot be negative.")
    if df["distance"].notna().sum() == 0:
        raise ValueError("Distance is missing for every training row.")


def validate_inference_data(df: pd.DataFrame, dataset_name: str) -> None:
    required = ["date", "pickup", "delivery", "distance", "equipment", "weight"]
    if ID_COL in df.columns:
        required.insert(0, ID_COL)
    validate_required_columns(df, required, dataset_name)
    if df.empty:
        raise ValueError(f"{dataset_name} is empty.")
    if ID_COL in df.columns:
        if df[ID_COL].isna().any():
            raise ValueError(f"{dataset_name} contains missing load_id values.")
        if df[ID_COL].duplicated().any():
            raise ValueError(f"{dataset_name} contains duplicate load_id values.")
    for col in ["distance", "weight", "market_index", "quote_signal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if (df["distance"].dropna() < 0).any():
        raise ValueError(f"{dataset_name} contains negative distances.")


def prepare_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct features without learning statistics from the dataframe."""
    df = normalize_column_names(df)
    out = df.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Found {int(dates.isna().sum())} missing or invalid date values.")
    out["month"] = dates.dt.month
    out["day_of_week"] = dates.dt.dayofweek
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    out["day_of_year"] = dates.dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    for col in ["distance", "weight"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["market_index", "quote_signal"]:
        if col not in out.columns:
            out[col] = np.nan
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["weight_clean"] = out["weight"].where(out["weight"] > 0, np.nan)
    out["weight_missing"] = out["weight_clean"].isna().astype(int)
    out["market_index_clean"] = out["market_index"]
    out["quote_signal_clean"] = out["quote_signal"]
    out["market_index_missing"] = out["market_index_clean"].isna().astype(int)
    out["quote_signal_missing"] = out["quote_signal_clean"].isna().astype(int)
    out["log_distance"] = np.log1p(out["distance"].clip(lower=0))
    out["log_weight"] = np.log1p(out["weight_clean"].clip(lower=0))
    out["dist_per_weight"] = out["distance"] / (out["weight_clean"] + 1.0)
    out["mi_times_qs"] = out["market_index_clean"] * out["quote_signal_clean"]
    for col in CATEGORICAL_FEATURES:
        out[col] = (
            out[col].astype("string").fillna("Unknown").str.strip()
            .replace("", "Unknown").astype(str)
        )
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def fit_preprocessor(raw_train: pd.DataFrame) -> dict:
    """Fit imputations and categorical encoding on training rows only (fold-local)."""
    medians = {}
    for col in NUMERIC_FEATURES:
        values = pd.to_numeric(raw_train[col], errors="coerce")
        median = values.median()
        medians[col] = float(median) if not pd.isna(median) else 0.0
    encoder = make_encoder()
    encoder.fit(raw_train[CATEGORICAL_FEATURES])
    categorical_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return {
        "medians": medians,
        "encoder": encoder,
        "feature_columns": NUMERIC_FEATURES + categorical_names,
    }


def transform_features(raw_df: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    medians = preprocessor["medians"]
    encoder = preprocessor["encoder"]
    feature_columns = preprocessor["feature_columns"]
    numeric = raw_df[NUMERIC_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce").fillna(medians[col])
    encoded_array = encoder.transform(raw_df[CATEGORICAL_FEATURES])
    encoded_columns = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    encoded = pd.DataFrame(encoded_array, columns=encoded_columns, index=raw_df.index)
    result = pd.concat([numeric, encoded], axis=1).reindex(columns=feature_columns)
    result = result.astype(np.float32)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Feature matrix contains NaN or infinite values after preprocessing.")
    return result


def make_model(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE, **params
    )


def transform_target(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.log1p(y)
    if transform == "identity":
        return y.astype(float)
    raise ValueError(f"Unknown target transform: {transform}")


def inverse_transform_target(pred: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(pred)
    if transform == "identity":
        return pred
    raise ValueError(f"Unknown target transform: {transform}")


def pooled_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)),
    }


def cv_evaluate(
    raw_train: pd.DataFrame,
    target: np.ndarray,
    params: dict,
    transform: str,
) -> dict:
    """Chronological K-fold CV with fold-local preprocessing; returns pooled metrics + OOF."""
    if len(raw_train) < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} training rows for CV.")
    if USE_CHRONOLOGICAL_CV:
        date_order = pd.to_datetime(raw_train["date"], errors="coerce").argsort(kind="mergesort")
        raw_train = raw_train.iloc[date_order].reset_index(drop=True)
        target = target[date_order]
        cv = KFold(n_splits=N_SPLITS, shuffle=False)
        cv_label = "chronological contiguous KFold"
    else:
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_label = "shuffled K-fold"

    y_t = transform_target(target, transform)
    oof_predictions = np.full(len(raw_train), np.nan)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(raw_train), start=1):
        fold_train = raw_train.iloc[train_idx]
        fold_valid = raw_train.iloc[valid_idx]
        preprocessor = fit_preprocessor(fold_train)
        X_train = transform_features(fold_train, preprocessor)
        X_valid = transform_features(fold_valid, preprocessor)
        model = make_model(params)
        model.fit(X_train, y_t[train_idx])
        pred_t = model.predict(X_valid)
        pred = inverse_transform_target(pred_t, transform)
        pred = np.clip(pred, 1.0, None)
        oof_predictions[valid_idx] = pred
        actual = target[valid_idx]
        fold_rows.append({
            "transform": transform, "fold": fold, "rows": len(valid_idx),
            **pooled_metrics(actual, pred),
        })

    if np.isnan(oof_predictions).any():
        raise RuntimeError("Some rows did not receive out-of-fold predictions.")
    overall = pooled_metrics(target, oof_predictions)
    return {"overall": overall, "fold_rows": fold_rows, "oof": oof_predictions}


def param_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def tune_hyperparameters(raw_train: pd.DataFrame, target: np.ndarray, transform: str) -> tuple[dict, dict]:
    """Grid-search hyperparameters; select by SELECTION_METRIC (lower better unless R2)."""
    best_params = None
    best_metrics = None
    best_result = None
    all_rows = []

    combos = param_combinations(PARAM_GRID)
    for params in combos:
        result = cv_evaluate(raw_train, target, params, transform)
        overall = result["overall"]
        all_rows.extend(result["fold_rows"])
        score = overall[SELECTION_METRIC]
        is_better = (
            (best_metrics is None)
            or (SELECTION_METRIC == "R2" and overall["R2"] > best_metrics["R2"])
            or (SELECTION_METRIC != "R2" and score < best_metrics[SELECTION_METRIC])
        )
        if is_better:
            best_params = params
            best_metrics = overall
            best_result = result

    print("\nBest hyperparameters")
    print("--------------------")
    print(f"Params : {best_params}")
    print(f"MAE    : {best_metrics['MAE']:.2f}")
    print(f"RMSE   : {best_metrics['RMSE']:.2f}")
    print(f"R2     : {best_metrics['R2']:.4f}")
    return best_params, best_result


def validate_predictions(predictions: pd.DataFrame, expected_ids: pd.Series, output_path: Path) -> None:
    required = [ID_COL, "predicted_rate"]
    if predictions.columns.tolist() != required:
        raise ValueError(f"{output_path.name} must contain exactly these columns in this order: {required}")
    if len(predictions) != len(expected_ids):
        raise ValueError(f"{output_path.name}: expected {len(expected_ids)} rows, got {len(predictions)}.")
    ids = predictions[ID_COL].astype(str)
    expected = expected_ids.astype(str)
    if ids.duplicated().any():
        raise ValueError(f"{output_path.name} contains duplicate IDs.")
    if set(ids) != set(expected):
        missing = set(expected) - set(ids)
        extra = set(ids) - set(expected)
        raise ValueError(f"ID mismatch in {output_path.name}. Missing: {len(missing)}; extra: {len(extra)}.")
    values = pd.to_numeric(predictions["predicted_rate"], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{output_path.name} contains invalid predictions.")
    if (values <= 0).any():
        raise ValueError(f"{output_path.name} contains non-positive predictions.")


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    train_path, validation_path, december_path = find_input_files()

    train_df = normalize_column_names(pd.read_csv(train_path))
    validation_df = normalize_column_names(pd.read_csv(validation_path))
    december_df = normalize_column_names(pd.read_csv(december_path))

    validate_training_data(train_df)
    validate_inference_data(validation_df, "Validation dataset")
    validate_inference_data(december_df, "December dataset")

    print("\nDataset sizes")
    print("-------------")
    print(f"Training rows  : {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"December rows  : {len(december_df):,}")

    # Template ID order (optional).
    template_candidates = [BASE / "validation_predictions_template.csv", DATA / "validation_predictions_template.csv"]
    template_path = next((p for p in template_candidates if p.exists()), None)
    if template_path is not None:
        template = normalize_column_names(pd.read_csv(template_path))
        validate_required_columns(template, [ID_COL], "Prediction template")
        if template[ID_COL].duplicated().any():
            raise ValueError("Prediction template contains duplicate IDs.")
        if set(template[ID_COL].astype(str)) != set(validation_df[ID_COL].astype(str)):
            raise ValueError("Validation IDs do not match the template IDs.")
        id_order = template[ID_COL].astype(str).tolist()
        validation_df[ID_COL] = validation_df[ID_COL].astype(str)
        validation_df = validation_df.set_index(ID_COL).loc[id_order].reset_index()

    raw_train = prepare_raw_features(train_df)
    raw_validation = prepare_raw_features(validation_df)
    raw_december = prepare_raw_features(december_df)

    target = pd.to_numeric(train_df[TARGET], errors="raise").to_numpy()

    # ---- Stage 1: select target transform ----
    print("\nStage 1: target-transform comparison (chronological CV, fold-local preprocessing)")
    transform_results = {}
    for transform in TARGET_TRANSFORMS:
        base_params = {
            "max_iter": 400, "learning_rate": 0.05, "max_depth": 6, "l2_regularization": 1.0,
        }
        result = cv_evaluate(raw_train, target, base_params, transform)
        transform_results[transform] = result
        overall = result["overall"]
        print(
            f"[{transform}] MAE={overall['MAE']:.2f} RMSE={overall['RMSE']:.2f} "
            f"R2={overall['R2']:.4f}"
        )

    if SELECTION_METRIC == "R2":
        best_transform = max(transform_results, key=lambda t: transform_results[t]["overall"]["R2"])
    else:
        best_transform = min(
            transform_results,
            key=lambda t: transform_results[t]["overall"][SELECTION_METRIC],
        )
    print(f"Selected target transform: {best_transform}")

    # ---- Stage 2: hyperparameter tuning on selected transform ----
    best_params, best_result = tune_hyperparameters(raw_train, target, best_transform)

    # Save CV artifacts.
    pd.DataFrame(best_result["fold_rows"]).to_csv(BASE / "cv_fold_metrics.csv", index=False)
    oof_df = pd.DataFrame(
        {
            ID_COL: raw_train[ID_COL].astype(str).to_numpy(),
            "transform": best_transform,
            "actual_rate": target,
            "oof_predicted_rate": best_result["oof"],
            "absolute_error": np.abs(target - best_result["oof"]),
        }
    )
    oof_df.to_csv(BASE / "oof_predictions.csv", index=False)
    pd.DataFrame([{"transform": best_transform, **best_result["overall"], "params": str(best_params)}]).to_csv(
        BASE / "cv_pooled_metrics.csv", index=False
    )

    # ---- Stage 3: final model on all training data with best params ----
    print("\nTraining final model on all training rows...")
    final_preprocessor = fit_preprocessor(raw_train)
    X_train = transform_features(raw_train, final_preprocessor)
    X_validation = transform_features(raw_validation, final_preprocessor)
    X_december = transform_features(raw_december, final_preprocessor)

    final_model = make_model(best_params)
    final_model.fit(X_train, transform_target(target, best_transform))

    # Validation predictions.
    validation_pred = inverse_transform_target(final_model.predict(X_validation), best_transform)
    validation_pred = np.clip(validation_pred, 1.0, None)
    validation_output = pd.DataFrame(
        {ID_COL: validation_df[ID_COL].astype(str).to_numpy(), "predicted_rate": validation_pred}
    )
    validation_output_path = BASE / "validation_predictions.csv"
    validate_predictions(validation_output, validation_df[ID_COL], validation_output_path)
    validation_output.to_csv(validation_output_path, index=False)

    # December predictions (preserve original input columns).
    december_pred = inverse_transform_target(final_model.predict(X_december), best_transform)
    december_pred = np.clip(december_pred, 1.0, None)
    december_output = december_df.copy()
    december_output["predicted_rate"] = np.round(december_pred, 2)
    if not np.isfinite(december_output["predicted_rate"].to_numpy()).all():
        raise ValueError("December predictions contain invalid values.")
    if (december_output["predicted_rate"] <= 0).any():
        raise ValueError("December predictions must be positive.")

    # Prefer underscore filename; also sync hyphen filename.
    december_output_path = DATA / "december_chart_inputs.csv"
    december_output.to_csv(december_output_path, index=False)
    hyphen_path = DATA / "december-chart-inputs.csv"
    december_output.to_csv(hyphen_path, index=False)

    artifact = {
        "model": final_model,
        "preprocessor": final_preprocessor,
        "target_transform": best_transform,
        "best_params": best_params,
        "best_cv_metrics": best_result["overall"],
        "feature_columns": final_preprocessor["feature_columns"],
        "model_params": best_params,
        "chronological_cv": USE_CHRONOLOGICAL_CV,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
    }
    joblib.dump(artifact, BASE / "model.joblib")

    print("\nOutput checks passed.")
    print(f"Validation rows: {len(validation_output):,}; December rows: {len(december_output):,}")


if __name__ == "__main__":
    main()
