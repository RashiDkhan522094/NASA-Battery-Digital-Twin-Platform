from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import IsolationForest


# =========================================================
# Project path
# =========================================================
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


from utils.data_loader import (
    get_battery_files,
    load_battery_data,
)


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🚨",
    layout="wide",
)


# =========================================================
# Styling
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(239, 68, 68, 0.09),
                transparent 30%
            ),
            #08111f;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 16px;
        padding: 15px;
    }

    div[data-testid="stPlotlyChart"] {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Helper functions
# =========================================================
def find_column(dataframe, possible_names):
    for name in possible_names:
        if name in dataframe.columns:
            return name
    return None


def safe_zscore(series):
    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    mean_value = numeric_series.mean()
    std_value = numeric_series.std()

    if pd.isna(std_value) or std_value == 0:
        return pd.Series(
            np.zeros(len(numeric_series)),
            index=numeric_series.index,
        )

    return (
        numeric_series - mean_value
    ) / std_value


def classify_severity(row):
    score = 0

    if row["thermal_anomaly"]:
        score += 2

    if row["voltage_drop_anomaly"]:
        score += 2

    if row["current_anomaly"]:
        score += 1

    if row["model_anomaly"]:
        score += 2

    if score >= 5:
        return "Critical"

    if score >= 2:
        return "Warning"

    return "Normal"


# =========================================================
# Header
# =========================================================
st.title("🚨 AI Anomaly Detection")

st.caption(
    "Detect abnormal voltage, temperature, current and "
    "multivariate battery behaviour"
)


# =========================================================
# Load battery files
# =========================================================
battery_files = get_battery_files()

if not battery_files:
    st.error(
        "No processed battery datasets were found."
    )
    st.stop()


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Anomaly Controls")

selected_battery = st.sidebar.selectbox(
    "Select battery",
    options=list(battery_files.keys()),
)

battery_df = load_battery_data(
    battery_files[selected_battery]
).copy()


# =========================================================
# Detect columns
# =========================================================
cycle_column = find_column(
    battery_df,
    [
        "discharge_cycle",
        "cycle",
        "cycle_number",
        "operation_index",
    ],
)

time_column = find_column(
    battery_df,
    [
        "time_s",
        "Time_s",
        "time",
    ],
)

voltage_column = find_column(
    battery_df,
    [
        "voltage_measured_V",
        "Voltage_measured",
        "voltage_V",
    ],
)

temperature_column = find_column(
    battery_df,
    [
        "temperature_measured_C",
        "Temperature_measured",
        "temperature_C",
    ],
)

current_column = find_column(
    battery_df,
    [
        "current_measured_A",
        "Current_measured",
        "current_A",
    ],
)

capacity_column = find_column(
    battery_df,
    [
        "capacity_Ah",
        "capacity",
        "Capacity",
    ],
)

voltage_change_column = find_column(
    battery_df,
    [
        "voltage_change_V",
        "delta_voltage_V",
    ],
)

temperature_change_column = find_column(
    battery_df,
    [
        "temperature_change_C",
        "delta_temperature_C",
    ],
)


required_columns = {
    "cycle": cycle_column,
    "time": time_column,
    "voltage": voltage_column,
    "temperature": temperature_column,
}

missing_required = [
    label
    for label, column in required_columns.items()
    if column is None
]

if missing_required:
    st.error(
        "Required columns are missing: "
        + ", ".join(missing_required)
    )

    st.write("Available columns:")
    st.write(list(battery_df.columns))
    st.stop()


# =========================================================
# Select cycle
# =========================================================
cycle_counts = (
    battery_df
    .groupby(cycle_column)
    .size()
)

valid_cycles = sorted(
    cycle_counts[
        cycle_counts >= 10
    ].index.tolist()
)

if not valid_cycles:
    st.error(
        "No discharge cycles contain enough rows "
        "for anomaly detection."
    )
    st.stop()


selected_cycle = st.sidebar.selectbox(
    "Select discharge cycle",
    options=valid_cycles,
)

contamination = st.sidebar.slider(
    "Expected anomaly fraction",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
)

temperature_limit = st.sidebar.slider(
    "Temperature warning limit (°C)",
    min_value=30.0,
    max_value=60.0,
    value=45.0,
    step=1.0,
)

voltage_limit = st.sidebar.slider(
    "Low-voltage limit (V)",
    min_value=2.5,
    max_value=3.5,
    value=3.0,
    step=0.1,
)


cycle_df = (
    battery_df[
        battery_df[cycle_column] == selected_cycle
    ]
    .copy()
    .sort_values(time_column)
    .reset_index(drop=True)
)


# =========================================================
# Engineer anomaly features
# =========================================================
cycle_df["voltage_zscore"] = safe_zscore(
    cycle_df[voltage_column]
)

cycle_df["temperature_zscore"] = safe_zscore(
    cycle_df[temperature_column]
)

if current_column is not None:
    cycle_df["current_zscore"] = safe_zscore(
        cycle_df[current_column]
    )
else:
    cycle_df["current_zscore"] = 0.0


if voltage_change_column is None:
    cycle_df["calculated_voltage_change_V"] = (
        cycle_df[voltage_column]
        .diff()
        .fillna(0)
    )
    voltage_change_column = (
        "calculated_voltage_change_V"
    )


if temperature_change_column is None:
    cycle_df[
        "calculated_temperature_change_C"
    ] = (
        cycle_df[temperature_column]
        .diff()
        .fillna(0)
    )
    temperature_change_column = (
        "calculated_temperature_change_C"
    )


cycle_df["thermal_anomaly"] = (
    (cycle_df[temperature_column] >= temperature_limit)
    | (cycle_df["temperature_zscore"].abs() >= 3.0)
    | (
        cycle_df[temperature_change_column]
        >= cycle_df[temperature_change_column]
        .quantile(0.99)
    )
)


cycle_df["voltage_drop_anomaly"] = (
    (cycle_df[voltage_column] <= voltage_limit)
    | (
        cycle_df[voltage_change_column]
        <= cycle_df[voltage_change_column]
        .quantile(0.01)
    )
    | (cycle_df["voltage_zscore"].abs() >= 3.0)
)


cycle_df["current_anomaly"] = (
    cycle_df["current_zscore"].abs() >= 3.0
)


# =========================================================
# Isolation Forest
# =========================================================
model_feature_columns = [
    voltage_column,
    temperature_column,
    voltage_change_column,
    temperature_change_column,
]

if current_column is not None:
    model_feature_columns.append(
        current_column
    )

if capacity_column is not None:
    model_feature_columns.append(
        capacity_column
    )


model_input = (
    cycle_df[model_feature_columns]
    .replace([np.inf, -np.inf], np.nan)
    .copy()
)

model_input = model_input.fillna(
    model_input.median(numeric_only=True)
)


try:
    isolation_model = IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    cycle_df["isolation_label"] = (
        isolation_model.fit_predict(
            model_input
        )
    )

    cycle_df["anomaly_score"] = (
        -isolation_model.score_samples(
            model_input
        )
    )

    cycle_df["model_anomaly"] = (
        cycle_df["isolation_label"] == -1
    )

except Exception as error:
    st.error(
        "Isolation Forest anomaly detection failed."
    )
    st.exception(error)
    st.stop()


# =========================================================
# Combined anomaly flag
# =========================================================
cycle_df["is_anomaly"] = (
    cycle_df["thermal_anomaly"]
    | cycle_df["voltage_drop_anomaly"]
    | cycle_df["current_anomaly"]
    | cycle_df["model_anomaly"]
)

cycle_df["severity"] = cycle_df.apply(
    classify_severity,
    axis=1,
)


# =========================================================
# Summary metrics
# =========================================================
total_rows = len(cycle_df)

anomaly_count = int(
    cycle_df["is_anomaly"].sum()
)

anomaly_rate = (
    anomaly_count
    / total_rows
    * 100
)

critical_count = int(
    (
        cycle_df["severity"] == "Critical"
    ).sum()
)

warning_count = int(
    (
        cycle_df["severity"] == "Warning"
    ).sum()
)


st.subheader(
    f"Battery {selected_battery} — "
    f"Discharge Cycle {selected_cycle}"
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Detected Anomalies",
        f"{anomaly_count}",
    )

with metric_2:
    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%",
    )

with metric_3:
    st.metric(
        "Warning Events",
        f"{warning_count}",
    )

with metric_4:
    st.metric(
        "Critical Events",
        f"{critical_count}",
    )


# =========================================================
# Overall health alert
# =========================================================
st.subheader("Mission Alert Status")

if critical_count > 0:
    st.error(
        f"{critical_count} critical anomaly events were detected. "
        "Immediate inspection is recommended."
    )

elif warning_count > 0:
    st.warning(
        f"{warning_count} warning events were detected. "
        "Continue close monitoring."
    )

elif anomaly_count > 0:
    st.info(
        "Minor statistical anomalies were detected, "
        "but no critical operating fault was identified."
    )

else:
    st.success(
        "No abnormal battery behaviour was detected "
        "for the selected cycle."
    )


# =========================================================
# Voltage anomaly chart
# =========================================================
voltage_figure = go.Figure()

voltage_figure.add_trace(
    go.Scatter(
        x=cycle_df[time_column],
        y=cycle_df[voltage_column],
        mode="lines",
        name="Measured voltage",
    )
)

voltage_anomalies = cycle_df[
    cycle_df["voltage_drop_anomaly"]
    | cycle_df["model_anomaly"]
]

voltage_figure.add_trace(
    go.Scatter(
        x=voltage_anomalies[time_column],
        y=voltage_anomalies[voltage_column],
        mode="markers",
        name="Detected anomaly",
        marker=dict(
            size=9,
            symbol="x",
        ),
    )
)

voltage_figure.add_hline(
    y=voltage_limit,
    line_dash="dash",
    annotation_text="Low-voltage limit",
)

voltage_figure.update_layout(
    title="Voltage Anomaly Detection",
    xaxis_title="Time (s)",
    yaxis_title="Voltage (V)",
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={
        "color": "white"
    },
)

st.plotly_chart(
    voltage_figure,
    use_container_width=True,
)


# =========================================================
# Temperature anomaly chart
# =========================================================
temperature_figure = go.Figure()

temperature_figure.add_trace(
    go.Scatter(
        x=cycle_df[time_column],
        y=cycle_df[temperature_column],
        mode="lines",
        name="Measured temperature",
    )
)

temperature_anomalies = cycle_df[
    cycle_df["thermal_anomaly"]
    | cycle_df["model_anomaly"]
]

temperature_figure.add_trace(
    go.Scatter(
        x=temperature_anomalies[time_column],
        y=temperature_anomalies[
            temperature_column
        ],
        mode="markers",
        name="Detected anomaly",
        marker=dict(
            size=9,
            symbol="x",
        ),
    )
)

temperature_figure.add_hline(
    y=temperature_limit,
    line_dash="dash",
    annotation_text="Temperature limit",
)

temperature_figure.update_layout(
    title="Temperature Anomaly Detection",
    xaxis_title="Time (s)",
    yaxis_title="Temperature (°C)",
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={
        "color": "white"
    },
)

st.plotly_chart(
    temperature_figure,
    use_container_width=True,
)


# =========================================================
# Anomaly score
# =========================================================
score_figure = go.Figure()

score_figure.add_trace(
    go.Scatter(
        x=cycle_df[time_column],
        y=cycle_df["anomaly_score"],
        mode="lines",
        name="Isolation Forest score",
    )
)

score_threshold = cycle_df.loc[
    cycle_df["model_anomaly"],
    "anomaly_score",
].min()

if not pd.isna(score_threshold):
    score_figure.add_hline(
        y=score_threshold,
        line_dash="dash",
        annotation_text="Model anomaly threshold",
    )

score_figure.update_layout(
    title="Multivariate Anomaly Score",
    xaxis_title="Time (s)",
    yaxis_title="Anomaly Score",
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={
        "color": "white"
    },
)

st.plotly_chart(
    score_figure,
    use_container_width=True,
)


# =========================================================
# Fault diagnosis
# =========================================================
st.subheader("Fault Diagnosis")

diagnosis_messages = []

if cycle_df["thermal_anomaly"].any():
    diagnosis_messages.append(
        "Thermal anomaly detected: possible overheating, "
        "rapid temperature rise or sensor instability."
    )

if cycle_df["voltage_drop_anomaly"].any():
    diagnosis_messages.append(
        "Voltage anomaly detected: possible deep discharge, "
        "rapid voltage collapse or degraded cell behaviour."
    )

if cycle_df["current_anomaly"].any():
    diagnosis_messages.append(
        "Current anomaly detected: possible load spike, "
        "measurement fault or unstable discharge demand."
    )

if cycle_df["model_anomaly"].any():
    diagnosis_messages.append(
        "Multivariate anomaly detected: the combined battery "
        "state differs from typical behaviour in this cycle."
    )


if diagnosis_messages:
    for message in diagnosis_messages:
        st.warning(message)

else:
    st.success(
        "No significant thermal, electrical or "
        "multivariate fault was identified."
    )


# =========================================================
# Anomaly event log
# =========================================================
anomaly_log = cycle_df[
    cycle_df["is_anomaly"]
].copy()

display_columns = [
    time_column,
    voltage_column,
    temperature_column,
]

if current_column is not None:
    display_columns.append(
        current_column
    )

display_columns.extend(
    [
        "anomaly_score",
        "thermal_anomaly",
        "voltage_drop_anomaly",
        "current_anomaly",
        "model_anomaly",
        "severity",
    ]
)


with st.expander("View anomaly event log"):
    if anomaly_log.empty:
        st.info(
            "No anomaly events were recorded."
        )
    else:
        st.dataframe(
            anomaly_log[
                display_columns
            ],
            use_container_width=True,
        )


# =========================================================
# Download results
# =========================================================
anomaly_csv = cycle_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download anomaly detection results",
    data=anomaly_csv,
    file_name=(
        f"{selected_battery}_cycle_"
        f"{selected_cycle}_anomalies.csv"
    ),
    mime="text/csv",
)