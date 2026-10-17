import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make training helpers importable so prediction preprocessing stays identical.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_predict import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ID_COL,
    TARGET,
    prepare_raw_features,
    transform_features,
    transform_target,
    inverse_transform_target,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODEL_PATH = BASE / "model.joblib"
TRAIN_PATH = DATA / "train-test.csv"
DECEMBER_PATH = DATA / "december_chart_inputs.csv"
CHART_PATH = BASE / "scorer_results" / "candidate_december.png"
VALIDATION_PATH = BASE / "validation_predictions.csv"
OOF_PATH = BASE / "oof_predictions.csv"
CV_FOLD_PATH = BASE / "cv_fold_metrics.csv"
CV_POOLED_PATH = BASE / "cv_pooled_metrics.csv"


@st.cache_resource
def load_artifact():
    import joblib

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    if TRAIN_PATH.exists():
        return pd.read_csv(TRAIN_PATH)
    return None


@st.cache_data
def load_oof():
    if OOF_PATH.exists():
        return pd.read_csv(OOF_PATH)
    return None


@st.cache_data
def load_cv_fold():
    if CV_FOLD_PATH.exists():
        return pd.read_csv(CV_FOLD_PATH)
    return None


artifact = load_artifact()
MODEL = artifact["model"]
PREPROCESSOR = artifact["preprocessor"]
TARGET_TRANSFORM = artifact.get("target_transform", "log1p")
BEST_PARAMS = artifact.get("best_params", {})
BEST_CV_METRICS = artifact.get("best_cv_metrics", {})
CHRONOLOGICAL_CV = artifact.get("chronological_cv", None)
N_SPLITS = artifact.get("n_splits", None)

train_df = load_training_data()
oof_df = load_oof()
cv_fold_df = load_cv_fold()


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run the exact training preprocessing on new data and predict."""
    raw = prepare_raw_features(df.copy())
    X = transform_features(raw, PREPROCESSOR)
    pred_t = MODEL.predict(X)
    pred = inverse_transform_target(pred_t, TARGET_TRANSFORM)
    pred = pd.Series(pred).clip(lower=1.0).to_numpy()
    out = df.copy()
    out["predicted_rate"] = pred
    return out


def plot_target_distribution(series):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(series, bins=60, color="#064A56", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Posted rate ($)")
    ax.set_ylabel("Number of loads")
    ax.set_title("Distribution of freight rates (training data)")
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_distance_vs_rate(df):
    import matplotlib.pyplot as plt

    sample = df.sample(n=min(3000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(sample["distance"], sample[TARGET], s=12, alpha=0.5, color="#064A56")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Posted rate ($)")
    ax.set_title("Rate vs. distance (sample of 3,000 loads)")
    ax.grid(True, color="#D9E2E4", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    return fig


def plot_equipment_vs_rate(df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    df.boxplot(column=TARGET, by="equipment", ax=ax,
               boxprops=dict(color="#064A56"),
               medianprops=dict(color="#FF6B35"),
               whiskerprops=dict(color="#064A56"),
               capprops=dict(color="#064A56"),
               flierprops=dict(marker="o", color="#064A56", alpha=0.4, markersize=3))
    ax.set_xlabel("Equipment type")
    ax.set_ylabel("Posted rate ($)")
    ax.set_title("Rate by equipment type")
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.6, alpha=0.7)
    plt.suptitle("")
    fig.tight_layout()
    return fig


def plot_monthly_trend(df):
    import matplotlib.pyplot as plt

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(monthly.index, monthly.values, marker="o", color="#064A56", linewidth=2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average posted rate ($)")
    ax.set_title("Average freight rate by month (training period)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, color="#D9E2E4", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    return fig


def plot_actual_vs_pred(oof_df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(oof_df["actual_rate"], oof_df["oof_predicted_rate"], s=14,
               alpha=0.45, color="#064A56", edgecolors="none")
    lo = min(oof_df["actual_rate"].min(), oof_df["oof_predicted_rate"].min())
    hi = max(oof_df["actual_rate"].max(), oof_df["oof_predicted_rate"].max())
    ax.plot([lo, hi], [lo, hi], color="#FF6B35", linestyle="--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual rate ($)")
    ax.set_ylabel("Predicted rate ($)")
    ax.set_title("Actual vs. predicted rate (out-of-fold)")
    ax.legend()
    ax.grid(True, color="#D9E2E4", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    return fig


def plot_residuals(oof_df):
    import matplotlib.pyplot as plt

    residuals = oof_df["actual_rate"] - oof_df["oof_predicted_rate"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(oof_df["oof_predicted_rate"], residuals, s=14, alpha=0.45, color="#064A56")
    ax.axhline(0, color="#FF6B35", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted rate ($)")
    ax.set_ylabel("Residual (actual − predicted, $)")
    ax.set_title("Residuals vs. predicted rate")
    ax.grid(True, color="#D9E2E4", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    return fig


def plot_cv_folds(cv_fold_df):
    import matplotlib.pyplot as plt

    plot_df = cv_fold_df[cv_fold_df["transform"] == TARGET_TRANSFORM].copy()
    folds = plot_df["fold"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(folds))
    ax.bar(x, plot_df["MAE"], color="#064A56", label="MAE", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(folds)
    ax.set_xlabel("CV fold")
    ax.set_ylabel("MAE ($)")
    ax.set_title(f"MAE by fold — {TARGET_TRANSFORM} target (chronological CV)")
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_cols):
    import matplotlib.pyplot as plt

    if not hasattr(model, "feature_importances_"):
        return None
    try:
        importances = model.feature_importances_
        if importances is None or len(importances) != len(feature_cols):
            return None
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp_df["feature"], imp_df["importance"], color="#064A56")
        ax.set_xlabel("Feature importance")
        ax.set_title("Top 12 features driving rate predictions")
        ax.grid(axis="x", color="#D9E2E4", linewidth=0.6, alpha=0.7)
        fig.tight_layout()
        return fig
    except Exception:  # noqa: BLE001
        return None


st.set_page_config(page_title="Freight Rate Predictor", page_icon="🚚", layout="wide")

st.title("🚚 Freight Rate Predictor")
st.caption(
    "HistGradientBoostingRegressor trained on 48,000 historical loads "
    "(Jan–Oct 2025). Predicts spot freight rates with fold-local preprocessing "
    "and chronological validation."
)

tab_single, tab_batch, tab_explore, tab_perf, tab_december, tab_info = st.tabs(
    ["Single load", "Batch (CSV)", "Data exploration", "Model performance",
     "December 2025 chart", "Model info"]
)

# ----------------------------------------------------------------------------
# Single load
# ----------------------------------------------------------------------------
with tab_single:
    st.subheader("Predict rate for one load")
    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            pickup = st.text_input("Pickup city", value="Lexington")
            delivery = st.text_input("Delivery city", value="Fort Wayne")
            distance = st.number_input("Distance (miles)", min_value=0.0, value=360.0, step=1.0)
        with c2:
            equipment = st.selectbox("Equipment", ["Dry Van", "Reefer", "Flatbed"], index=0)
            weight = st.number_input("Weight (lb)", min_value=0.0, value=32000.0, step=100.0)
            date = st.date_input("Pickup date")
        with c3:
            market_index = st.number_input("Market index", value=1.05, step=0.01, format="%.4f")
            quote_signal = st.number_input("Quote signal", value=2.05, step=0.01, format="%.4f")

        submitted = st.form_submit_button("Predict rate")

    if submitted:
        row = pd.DataFrame(
            [
                {
                    ID_COL: "SINGLE",
                    "pickup": pickup,
                    "delivery": delivery,
                    "distance": distance,
                    "equipment": equipment,
                    "weight": weight,
                    "date": pd.Timestamp(date),
                    "market_index": market_index,
                    "quote_signal": quote_signal,
                }
            ]
        )
        try:
            result = predict_dataframe(row)
            rate = float(result["predicted_rate"].iloc[0])
            st.success(f"Predicted rate: **${rate:,.2f}**")
            st.metric("Predicted rate ($)", f"{rate:,.2f}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")

# ----------------------------------------------------------------------------
# Batch prediction
# ----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch predict from CSV")
    st.write("Upload a CSV with columns: `pickup, delivery, distance, equipment, weight, date` "
             "(optional: `market_index`, `quote_signal`, `load_id`).")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            result = predict_dataframe(batch_df)
            st.success(f"Predicted {len(result)} loads.")
            st.dataframe(result.head(20), use_container_width=True, hide_index=True)
            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Batch prediction failed: {exc}")

# ----------------------------------------------------------------------------
# Data exploration
# ----------------------------------------------------------------------------
with tab_explore:
    st.subheader("Training data exploration")
    st.write("Visual summaries of the 48,000 historical loads used to train the model.")

    if train_df is None:
        st.info("Training data file not found.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total loads", f"{len(train_df):,}")
        c2.metric("Date range", f"{train_df['date'].min()} → {train_df['date'].max()}")
        c3.metric("Avg rate", f"${train_df[TARGET].mean():,.0f}")

        st.markdown("#### Rate distribution")
        try:
            st.pyplot(plot_target_distribution(train_df[TARGET]))
        except Exception:  # noqa: BLE001
            st.info("Rate distribution chart could not be rendered.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Rate vs. distance")
            try:
                st.pyplot(plot_distance_vs_rate(train_df))
            except Exception:  # noqa: BLE001
                st.info("Rate vs. distance chart could not be rendered.")
        with c2:
            st.markdown("#### Rate by equipment")
            try:
                st.pyplot(plot_equipment_vs_rate(train_df))
            except Exception:  # noqa: BLE001
                st.info("Rate by equipment chart could not be rendered.")

        st.markdown("#### Monthly rate trend")
        try:
            st.pyplot(plot_monthly_trend(train_df))
        except Exception:  # noqa: BLE001
            st.info("Monthly rate trend chart could not be rendered.")

# ----------------------------------------------------------------------------
# Model performance
# ----------------------------------------------------------------------------
with tab_perf:
    st.subheader("Model performance & diagnostics")
    st.write("How well the model predicts freight rates, measured with chronological cross-validation.")

    if BEST_CV_METRICS:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"${BEST_CV_METRICS.get('MAE', 0):,.2f}")
        c2.metric("RMSE", f"${BEST_CV_METRICS.get('RMSE', 0):,.2f}")
        c3.metric("R²", f"{BEST_CV_METRICS.get('R2', 0):.4f}")
        st.caption("Pooled out-of-fold metrics across 5 chronological folds.")

    st.markdown("#### MAE by CV fold")
    if cv_fold_df is not None:
        try:
            st.pyplot(plot_cv_folds(cv_fold_df))
        except Exception:  # noqa: BLE001
            st.info("MAE by fold chart could not be rendered.")
        st.caption("Each bar is one time-ordered validation block. Lower is better.")

    st.markdown("#### Actual vs. predicted rate")
    if oof_df is not None:
        try:
            st.pyplot(plot_actual_vs_pred(oof_df))
        except Exception:  # noqa: BLE001
            st.info("Actual vs. predicted chart could not be rendered.")
        st.caption("Points near the dashed line mean the prediction matches the actual rate.")

        st.markdown("#### Residuals vs. predicted rate")
        try:
            st.pyplot(plot_residuals(oof_df))
        except Exception:  # noqa: BLE001
            st.info("Residuals chart could not be rendered.")
        st.caption("Residuals scattered around zero with no clear pattern indicate a well-calibrated model.")
    else:
        st.info("Run `python train_predict.py` to generate OOF predictions for these plots.")

    st.markdown("#### Feature importance")
    fig_imp = plot_feature_importance(MODEL, PREPROCESSOR["feature_columns"])
    if fig_imp:
        try:
            st.pyplot(fig_imp)
        except Exception:  # noqa: BLE001
            st.info("Feature importance chart could not be rendered.")
        st.caption("Distance is the dominant driver of freight rate, as expected.")
    else:
        st.info("Feature importances not available from this model.")

# ----------------------------------------------------------------------------
# December chart
# ----------------------------------------------------------------------------
with tab_december:
    st.subheader("December 2025 fixed-lane predictions")
    st.write(
        "Fixed inputs: Lexington → Fort Wayne | 360 miles | Dry Van | 32,000 lb | only date changes."
    )
    if DECEMBER_PATH.exists():
        dec_df = pd.read_csv(DECEMBER_PATH)
        st.dataframe(dec_df, use_container_width=True, hide_index=True)
        if CHART_PATH.exists():
            try:
                st.image(str(CHART_PATH), caption="Candidate: December 2025 Predicted Load Rate")
            except Exception:  # noqa: BLE001
                st.info("December chart image could not be rendered.")
        else:
            st.info("Run `python score.py` to generate the chart.")
    else:
        st.info("Run `python train_predict.py` to generate December predictions.")

# ----------------------------------------------------------------------------
# Model info
# ----------------------------------------------------------------------------
with tab_info:
    st.subheader("Model & pipeline summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows", f"{len(train_df):,}" if train_df is not None else "—")
    c2.metric("Validation rows", "12,000")
    c3.metric("Features", len(PREPROCESSOR["feature_columns"]))

    st.write("**Model:**", type(MODEL).__name__)
    st.write("**Target transform:**", TARGET_TRANSFORM)
    st.write("**Chronological CV:**", CHRONOLOGICAL_CV)
    st.write("**CV folds:**", N_SPLITS)
    st.write("**Best hyperparameters:**")
    st.code(str(BEST_PARAMS))
    st.write("**Best CV metrics:**")
    st.code(str(BEST_CV_METRICS))
    with st.expander("Full feature column list"):
        st.code("\n".join(PREPROCESSOR["feature_columns"]))

    st.divider()
    st.write("**Output files**")
    st.write("- `validation_predictions.csv` — 12,000 validation load predictions")
    st.write("- `data/december_chart_inputs.csv` — 31 December fixed-route predictions")
    st.write("- `scorer_results/candidate_december.png` — December rate chart")
    st.write("- `model.joblib` — trained model + preprocessor")
    st.write("- `cv_fold_metrics.csv`, `cv_pooled_metrics.csv`, `oof_predictions.csv` — CV audit trail")
