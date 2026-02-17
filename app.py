import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dengue AI Dashboard", layout="wide")

comp = pd.read_csv("data/MODEL_COMPARISON_TABLE_Calibrated.csv")
test = pd.read_csv("data/COMPARE_Test_2022_Calibrated_AllModels.csv", parse_dates=["DATE"])
fcst = pd.read_csv("data/FORECAST_2023_2025_Calibrated_AllModels.csv", parse_dates=["DATE"])
calp = pd.read_csv("data/CALIBRATION_PARAMS_2021.csv")

st.title("AI-Driven Dengue Prediction Dashboard (Nueva Vizcaya)")

st.subheader("Model Ranking (Test Year: 2022 Observed)")
st.dataframe(comp, use_container_width=True)

best_model = comp.sort_values("MAE_2022").iloc[0]["Model"]
st.write(f"**Best model on 2022 observed:** {best_model}")

st.subheader("Calibration Parameters (fit on 2021)")
st.dataframe(calp, use_container_width=True)

model_choice = st.selectbox(
    "Choose model to visualize",
    ["LSTM", "XGBoost", "RandomForest", "Ensemble"]
)

use_cal = st.checkbox("Use calibrated predictions", value=True)

col_map_test = {
    "XGBoost": ("XGB_PRED", "XGB_PRED_CAL"),
    "RandomForest": ("RF_PRED", "RF_PRED_CAL"),
    "LSTM": ("LSTM_PRED", "LSTM_PRED_CAL"),
    "Ensemble": ("ENSEMBLE_PRED", "ENSEMBLE_PRED_CAL"),
}
col_map_fc = col_map_test

raw_col, cal_col = col_map_test[model_choice]
pred_col = cal_col if use_cal and cal_col in test.columns else raw_col

st.subheader("2022 Observed: Actual vs Predicted")
plot_df = test[["DATE", "ACTUAL_CASES", pred_col]].rename(columns={pred_col: "PREDICTED"}).set_index("DATE")
st.line_chart(plot_df)

pred_col_fc = cal_col if use_cal and cal_col in fcst.columns else raw_col
st.subheader("Forecast Demo (2023–2025)")
plot_fc = fcst[["DATE", pred_col_fc]].rename(columns={pred_col_fc: "FORECASTED_CASES"}).set_index("DATE")
st.line_chart(plot_fc)

st.caption("Note: 2023–2025 are forecast estimates only and are not evaluated against observed ground truth.")
