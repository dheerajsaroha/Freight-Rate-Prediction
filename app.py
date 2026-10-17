import os
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
MODEL_PATH = BASE / "model.joblib"
DECEMBER_PATH = BASE / "data" / "december_chart_inputs.csv"
CHART_PATH = BASE / "scorer_results" / "candidate_december.png"
VALIDATION_PATH = BASE / "validation_predictions.csv"


@st.cache_resource
def load_artifact():
    import joblib

    artifact = joblib.load(MODEL_PATH)
    return artifact


artifact = load_artifact()
MODEL = artifact["model"]
PREPROCESSOR = artifact["preprocessor"]
TARGET_TRANSFORM = artifact.get("target_transform", "log1p")
BEST_PARAMS = artifact.get("best_params", {})
BEST_CV_METRICS = artifact.get("best_cv_metrics", {})
CHRONOLOGICAL_CV = artifact.get("chronological_cv", None)
N_SPLITS = artifact.get("n_splits", None)


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


st.set_page_config(page_title="Freight Rate Predictor", page_icon="🚚", layout="wide")

st.title("🚚 Freight Rate Predictor")
st.caption(
    "HistGradientBoostingRegressor trained on 48,000 historical loads "
    "(Jan–Oct 2025). Predicts spot freight rates with fold-local preprocessing "
    "and chronological validation."
)

tab_single, tab_batch, tab_december, tab_info = st.tabs(
    ["Single load", "Batch (CSV)", "December 2025 chart", "Model info"]
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
            st.dataframe(result.head(20), use_container_width=True)
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
            st.image(str(CHART_PATH), caption="Candidate: December 2025 Predicted Load Rate")
        else:
            st.info("Run `python score.py` to generate the chart.")
    else:
        st.info("Run `python train_predict.py` to generate December predictions.")

# ----------------------------------------------------------------------------
# Model info
# ----------------------------------------------------------------------------
with tab_info:
    st.subheader("Model & validation summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{BEST_CV_METRICS.get('MAE', float('nan')):,.2f}" if BEST_CV_METRICS else "—")
    c2.metric("RMSE", f"{BEST_CV_METRICS.get('RMSE', float('nan')):,.2f}" if BEST_CV_METRICS else "—")
    c3.metric("R²", f"{BEST_CV_METRICS.get('R2', float('nan')):.4f}" if BEST_CV_METRICS else "—")
    st.write("**Target transform:**", TARGET_TRANSFORM)
    st.write("**Chronological CV:**", CHRONOLOGICAL_CV)
    st.write("**CV folds:**", N_SPLITS)
    st.write("**Best params:**", BEST_PARAMS)
    st.write("**Feature columns:**", len(PREPROCESSOR["feature_columns"]))
    with st.expander("Feature column list"):
        st.code("\n".join(PREPROCESSOR["feature_columns"]))
