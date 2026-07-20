from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


from utils.data_loader import (
    get_battery_files,
    load_battery_data
)

from utils.model_loader import load_models


# ---------------------------------------------------------
# Page settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Prediction",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------
def find_column(dataframe, possible_names):
    for name in possible_names:
        if name in dataframe.columns:
            return name

    return None


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🤖 AI Prediction Analysis")

st.caption(
    "Evaluate next-step voltage and temperature predictions "
    "across a complete discharge cycle"
)


# ---------------------------------------------------------
# Load battery files
# ---------------------------------------------------------
battery_files = get_battery_files()

if not battery_files:
    st.error("No processed battery datasets were found.")
    st.stop()


# ---------------------------------------------------------
# Load trained models
# ---------------------------------------------------------
try:
    voltage_model, temperature_model, feature_list = load_models()

except Exception as error:
    st.error("The trained models could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
selected_battery = st.sidebar.selectbox(
    "Select battery",
    options=list(battery_files.keys())
)

battery_df = load_battery_data(
    battery_files[selected_battery]
).copy()


# ---------------------------------------------------------
# Detect required columns
# ---------------------------------------------------------
cycle_column = find_column(
    battery_df,
    [
        "discharge_cycle",
        "cycle",
        "cycle_number",
        "operation_index"
    ]
)

time_column = find_column(
    battery_df,
    [
        "time_s",
        "Time_s",
        "time"
    ]
)

next_voltage_column = find_column(
    battery_df,
    [
        "next_voltage_V",
        "next_voltage"
    ]
)

next_temperature_column = find_column(
    battery_df,
    [
        "next_temperature_C",
        "next_temperature"
    ]
)


if (
    cycle_column is None
    or time_column is None
    or next_voltage_column is None
    or next_temperature_column is None
):
    st.error(
        "Required cycle, time, voltage-target or "
        "temperature-target columns are missing."
    )

    st.write("Available columns:")
    st.write(list(battery_df.columns))

    st.stop()


# ---------------------------------------------------------
# Confirm trained-model features exist
# ---------------------------------------------------------
missing_features = [
    feature
    for feature in feature_list
    if feature not in battery_df.columns
]

if missing_features:
    st.error(
        "Some trained-model features are missing "
        "from this battery dataset."
    )

    st.write(missing_features)
    st.stop()


# ---------------------------------------------------------
# Select a valid cycle
# ---------------------------------------------------------
cycle_counts = (
    battery_df
    .groupby(cycle_column)
    .size()
)

valid_cycles = sorted(
    cycle_counts[
        cycle_counts >= 2
    ].index.tolist()
)


if not valid_cycles:
    st.error(
        "No discharge cycles contain enough rows "
        "for prediction."
    )
    st.stop()


selected_cycle = st.sidebar.selectbox(
    "Select discharge cycle",
    options=valid_cycles
)


cycle_df = (
    battery_df[
        battery_df[cycle_column] == selected_cycle
    ]
    .copy()
    .sort_values(time_column)
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------
X_cycle = (
    cycle_df[feature_list]
    .replace([np.inf, -np.inf], np.nan)
)


try:
    cycle_df["predicted_next_voltage_V"] = (
        voltage_model.predict(X_cycle)
    )

    cycle_df["predicted_next_temperature_C"] = (
        temperature_model.predict(X_cycle)
    )

except Exception as error:
    st.error(
        "Prediction failed for the selected cycle."
    )

    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Prepare evaluation dataset
# ---------------------------------------------------------
evaluation_df = cycle_df.dropna(
    subset=[
        next_voltage_column,
        next_temperature_column,
        "predicted_next_voltage_V",
        "predicted_next_temperature_C"
    ]
).copy()


if evaluation_df.empty:
    st.warning(
        "No valid rows are available for evaluation."
    )
    st.stop()


# ---------------------------------------------------------
# Actual and predicted values
# ---------------------------------------------------------
voltage_actual = evaluation_df[
    next_voltage_column
]

voltage_predicted = evaluation_df[
    "predicted_next_voltage_V"
]

temperature_actual = evaluation_df[
    next_temperature_column
]

temperature_predicted = evaluation_df[
    "predicted_next_temperature_C"
]


# ---------------------------------------------------------
# Calculate voltage metrics
# ---------------------------------------------------------
voltage_mae = mean_absolute_error(
    voltage_actual,
    voltage_predicted
)

voltage_rmse = np.sqrt(
    mean_squared_error(
        voltage_actual,
        voltage_predicted
    )
)

voltage_r2 = r2_score(
    voltage_actual,
    voltage_predicted
)


# ---------------------------------------------------------
# Calculate temperature metrics
# ---------------------------------------------------------
temperature_mae = mean_absolute_error(
    temperature_actual,
    temperature_predicted
)

temperature_rmse = np.sqrt(
    mean_squared_error(
        temperature_actual,
        temperature_predicted
    )
)

temperature_r2 = r2_score(
    temperature_actual,
    temperature_predicted
)


# ---------------------------------------------------------
# Page title
# ---------------------------------------------------------
st.subheader(
    f"Battery {selected_battery} — "
    f"Discharge Cycle {selected_cycle}"
)


# ---------------------------------------------------------
# Voltage metrics
# ---------------------------------------------------------
st.markdown("### Voltage Prediction Performance")

v1, v2, v3 = st.columns(3)

with v1:
    st.metric(
        "Voltage MAE",
        f"{voltage_mae:.6f} V"
    )

with v2:
    st.metric(
        "Voltage RMSE",
        f"{voltage_rmse:.6f} V"
    )

with v3:
    st.metric(
        "Voltage R²",
        f"{voltage_r2:.4f}"
    )


# ---------------------------------------------------------
# Temperature metrics
# ---------------------------------------------------------
st.markdown("### Temperature Prediction Performance")

t1, t2, t3 = st.columns(3)

with t1:
    st.metric(
        "Temperature MAE",
        f"{temperature_mae:.4f} °C"
    )

with t2:
    st.metric(
        "Temperature RMSE",
        f"{temperature_rmse:.4f} °C"
    )

with t3:
    st.metric(
        "Temperature R²",
        f"{temperature_r2:.4f}"
    )


# ---------------------------------------------------------
# Voltage chart
# ---------------------------------------------------------
voltage_figure = go.Figure()

voltage_figure.add_trace(
    go.Scatter(
        x=evaluation_df[time_column],
        y=voltage_actual,
        mode="lines",
        name="Actual next voltage"
    )
)

voltage_figure.add_trace(
    go.Scatter(
        x=evaluation_df[time_column],
        y=voltage_predicted,
        mode="lines",
        name="Predicted next voltage",
        line=dict(dash="dash")
    )
)

voltage_figure.update_layout(
    title="Actual vs Predicted Next Voltage",
    xaxis_title="Current Time (s)",
    yaxis_title="Next Voltage (V)",
    hovermode="x unified"
)

st.plotly_chart(
    voltage_figure,
    use_container_width=True
)


# ---------------------------------------------------------
# Temperature chart
# ---------------------------------------------------------
temperature_figure = go.Figure()

temperature_figure.add_trace(
    go.Scatter(
        x=evaluation_df[time_column],
        y=temperature_actual,
        mode="lines",
        name="Actual next temperature"
    )
)

temperature_figure.add_trace(
    go.Scatter(
        x=evaluation_df[time_column],
        y=temperature_predicted,
        mode="lines",
        name="Predicted next temperature",
        line=dict(dash="dash")
    )
)

temperature_figure.update_layout(
    title="Actual vs Predicted Next Temperature",
    xaxis_title="Current Time (s)",
    yaxis_title="Next Temperature (°C)",
    hovermode="x unified"
)

st.plotly_chart(
    temperature_figure,
    use_container_width=True
)


# ---------------------------------------------------------
# Calculate prediction errors
# ---------------------------------------------------------
evaluation_df["voltage_error_V"] = (
    evaluation_df["predicted_next_voltage_V"]
    - evaluation_df[next_voltage_column]
)

evaluation_df["temperature_error_C"] = (
    evaluation_df["predicted_next_temperature_C"]
    - evaluation_df[next_temperature_column]
)


# ---------------------------------------------------------
# Error charts
# ---------------------------------------------------------
left_chart, right_chart = st.columns(2)


with left_chart:
    voltage_error_figure = go.Figure()

    voltage_error_figure.add_trace(
        go.Scatter(
            x=evaluation_df[time_column],
            y=evaluation_df["voltage_error_V"],
            mode="lines",
            name="Voltage error"
        )
    )

    voltage_error_figure.add_hline(
        y=0,
        line_dash="dash"
    )

    voltage_error_figure.update_layout(
        title="Voltage Prediction Error",
        xaxis_title="Current Time (s)",
        yaxis_title="Error (V)"
    )

    st.plotly_chart(
        voltage_error_figure,
        use_container_width=True
    )


with right_chart:
    temperature_error_figure = go.Figure()

    temperature_error_figure.add_trace(
        go.Scatter(
            x=evaluation_df[time_column],
            y=evaluation_df["temperature_error_C"],
            mode="lines",
            name="Temperature error"
        )
    )

    temperature_error_figure.add_hline(
        y=0,
        line_dash="dash"
    )

    temperature_error_figure.update_layout(
        title="Temperature Prediction Error",
        xaxis_title="Current Time (s)",
        yaxis_title="Error (°C)"
    )

    st.plotly_chart(
        temperature_error_figure,
        use_container_width=True
    )


# ---------------------------------------------------------
# Model assessment
# ---------------------------------------------------------
st.subheader("Model Assessment")

if (
    voltage_r2 >= 0.95
    and temperature_r2 >= 0.95
):
    st.success(
        "Both models show excellent agreement with "
        "the measured next-step values."
    )

elif (
    voltage_r2 >= 0.80
    and temperature_r2 >= 0.80
):
    st.info(
        "The models show good predictive performance, "
        "although some local errors remain."
    )

else:
    st.warning(
        "One or both models show reduced performance "
        "for this battery-cycle combination."
    )


# ---------------------------------------------------------
# Results table
# ---------------------------------------------------------
output_columns = [
    time_column,
    next_voltage_column,
    "predicted_next_voltage_V",
    "voltage_error_V",
    next_temperature_column,
    "predicted_next_temperature_C",
    "temperature_error_C"
]


with st.expander("View prediction results"):
    st.dataframe(
        evaluation_df[output_columns],
        use_container_width=True
    )


# ---------------------------------------------------------
# Download button
# ---------------------------------------------------------
prediction_csv = evaluation_df[
    output_columns
].to_csv(index=False).encode("utf-8")


st.download_button(
    label="Download prediction results",
    data=prediction_csv,
    file_name=(
        f"{selected_battery}_cycle_"
        f"{selected_cycle}_predictions.csv"
    ),
    mime="text/csv"
)