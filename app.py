from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.ui_polish import (
    apply_global_polish,
    render_startup_screen,
    render_footer,
)


# =========================================================
# Project path
# =========================================================
PROJECT_FOLDER = Path(__file__).resolve().parent

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


from utils.data_loader import get_battery_files

from utils.preprocessing import preprocess_battery_data

from utils.model_loader import (
    load_models,
    load_rul_model,
)

from utils.prediction import (
    predict_next_state,
    predict_rul,
)

from utils.helpers import battery_status


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="NASA Battery Digital Twin",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_polish()
render_startup_screen()

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
                rgba(37, 99, 235, 0.12),
                transparent 32%
            ),
            #08111f;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
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

    .main-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .main-subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Helper functions
# =========================================================
def create_gauge(
    value,
    minimum,
    maximum,
    title,
    suffix="",
    threshold=None,
):
    gauge = {
        "axis": {
            "range": [minimum, maximum],
        },
        "bar": {
            "thickness": 0.28,
        },
        "bgcolor": "rgba(0,0,0,0)",
        "borderwidth": 0,
        "steps": [
            {
                "range": [
                    minimum,
                    minimum + (maximum - minimum) * 0.33,
                ],
                "color": "rgba(239, 68, 68, 0.20)",
            },
            {
                "range": [
                    minimum + (maximum - minimum) * 0.33,
                    minimum + (maximum - minimum) * 0.66,
                ],
                "color": "rgba(245, 158, 11, 0.18)",
            },
            {
                "range": [
                    minimum + (maximum - minimum) * 0.66,
                    maximum,
                ],
                "color": "rgba(34, 197, 94, 0.18)",
            },
        ],
    }

    if threshold is not None:
        gauge["threshold"] = {
            "line": {
                "width": 3,
            },
            "thickness": 0.75,
            "value": threshold,
        }

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": suffix,
                "font": {
                    "size": 26,
                },
            },
            title={
                "text": title,
                "font": {
                    "size": 18,
                },
            },
            gauge=gauge,
        )
    )

    figure.update_layout(
        height=300,
        margin=dict(
            l=35,
            r=35,
            t=60,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "white",
        },
    )

    return figure


# =========================================================
# Header
# =========================================================
st.markdown(
    '<div class="main-title">🚀 NASA Battery Digital Twin</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-subtitle">
    Mission-control interface for battery health monitoring,
    AI prediction and digital-twin simulation
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Load battery files and models
# =========================================================
battery_files = get_battery_files()

if not battery_files:
    st.error(
        "No processed battery files were found in the "
        "`processed_datasets` folder."
    )
    st.stop()


try:
    (
        voltage_model,
        temperature_model,
        feature_list,
    ) = load_models()

    (
        rul_model,
        rul_feature_columns,
    ) = load_rul_model()

except Exception as error:
    st.error(
        "The trained models could not be loaded."
    )
    st.exception(error)
    st.stop()


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.title("Mission Controls")

selected_battery = st.sidebar.selectbox(
    "Battery",
    options=list(battery_files.keys()),
)

try:
    processed_data = preprocess_battery_data(
        battery_files[selected_battery]
    )

except Exception as error:
    st.error(
        "The selected battery dataset could not be processed."
    )
    st.exception(error)
    st.stop()


battery_df = processed_data["battery_df"].copy()

columns = processed_data["columns"]

cycle_column = columns["cycle"]
time_column = columns["time"]
voltage_column = columns["voltage"]
temperature_column = columns["temperature"]
current_column = columns["current"]
capacity_column = columns["capacity"]

available_cycles = processed_data["available_cycles"]

cycle_capacity_df = processed_data[
    "cycle_capacity_df"
].copy()


# =========================================================
# Select discharge cycle
# =========================================================

if not available_cycles:
    st.error(
        "No valid discharge cycles were found."
    )
    st.stop()


selected_cycle = st.sidebar.selectbox(
    "Discharge cycle",
    options=available_cycles,
)

cycle_df = (
    battery_df[
        battery_df[cycle_column] == selected_cycle
    ]
    .copy()
    .sort_values(time_column)
    .reset_index(drop=True)
)

if len(cycle_df) < 2:
    st.warning(
        "The selected cycle does not contain enough rows."
    )
    st.stop()


selected_index = st.sidebar.slider(
    "Mission time step",
    min_value=0,
    max_value=len(cycle_df) - 2,
    value=min(20, len(cycle_df) - 2),
    step=1,
)

current_row = cycle_df.iloc[selected_index]
next_row = cycle_df.iloc[selected_index + 1]


# =========================================================
# Current measurements
# =========================================================
current_voltage = float(
    current_row[voltage_column]
)

current_temperature = float(
    current_row[temperature_column]
)

current_time = float(
    current_row[time_column]
)

next_time = float(
    next_row[time_column]
)

if current_column is not None:
    current_value = float(
        current_row[current_column]
    )
else:
    current_value = 0.0


# =========================================================
# SOH calculation using early-life reference capacity
# =========================================================
if cycle_capacity_df.empty:
    st.error(
        "No physically valid cycle-level capacity data were found."
    )
    st.stop()


# Use the maximum capacity recorded within the first
# 20 valid discharge cycles as the early-life reference.
reference_window = cycle_capacity_df.head(
    min(20, len(cycle_capacity_df))
)

reference_capacity = float(
    reference_window[capacity_column].max()
)

if (
    not np.isfinite(reference_capacity)
    or reference_capacity <= 0
):
    st.error(
        "A valid early-life reference capacity "
        "could not be calculated."
    )
    st.stop()


# Capacity of the currently selected discharge cycle
selected_cycle_capacity_values = pd.to_numeric(
    cycle_df[capacity_column],
    errors="coerce",
).dropna()

selected_cycle_capacity_values = (
    selected_cycle_capacity_values[
        selected_cycle_capacity_values > 0
    ]
)

if selected_cycle_capacity_values.empty:
    st.error(
        "The selected cycle has no valid capacity value."
    )
    st.stop()

current_capacity = float(
    selected_cycle_capacity_values.median()
)


# Calculate SOH relative to the early-life reference
current_soh = (
    current_capacity
    / reference_capacity
    * 100
)

# Small values above 100% may occur from measurement noise
health_percentage = float(
    np.clip(
        current_soh,
        0,
        100,
    )
)
# =========================================================
# AI next-state prediction
# =========================================================
try:
    (
        predicted_voltage,
        predicted_temperature,
    ) = predict_next_state(
        current_row,
        voltage_model,
        temperature_model,
        feature_list,
    )

except Exception as error:
    st.error(
        "The voltage and temperature models "
        "could not generate a prediction."
    )
    st.exception(error)
    st.stop()


# =========================================================
# Battery operating status
# =========================================================
voltage_state, temperature_state, _ = battery_status(
    current_voltage,
    current_temperature,
    current_capacity,
)


if health_percentage >= 90:
    health_state = "Healthy"

elif health_percentage >= 80:
    health_state = "Warning"

elif health_percentage >= 70:
    health_state = "Degraded"

else:
    health_state = "Critical"


# =========================================================
# AI-based RUL prediction
# =========================================================
try:
    estimated_rul = predict_rul(
        cycle_df=cycle_df,
        cycle_column=cycle_column,
        capacity_column=capacity_column,
        temperature_column=temperature_column,
        voltage_column=voltage_column,
        current_column=current_column,
        time_column=time_column,
        rul_model=rul_model,
        rul_feature_columns=rul_feature_columns,
    )

except Exception as error:
    st.warning(
        "The ML-based RUL model could not generate a prediction."
    )
    estimated_rul = np.nan


if health_percentage <= 80:
    rul_text = "EOL"
    rul_gauge_value = 0.0

elif np.isnan(estimated_rul):
    rul_text = "Unknown"
    rul_gauge_value = 0.0

else:
    rul_text = f"{estimated_rul:.0f} cycles"

    rul_gauge_value = float(
        np.clip(
            estimated_rul,
            0,
            500,
        )
    )


# =========================================================
# Mission summary
# =========================================================
st.subheader(
    f"Mission Status — Battery {selected_battery}, "
    f"Cycle {selected_cycle}"
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Current Voltage",
        f"{current_voltage:.4f} V",
    )

with metric_2:
    st.metric(
        "Predicted Next Voltage",
        f"{predicted_voltage:.4f} V",
        delta=(
            f"{predicted_voltage - current_voltage:+.4f} V"
        ),
    )

with metric_3:
    st.metric(
        "Current Temperature",
        f"{current_temperature:.2f} °C",
    )

with metric_4:
    st.metric(
        "Predicted Temperature",
        f"{predicted_temperature:.2f} °C",
        delta=(
            f"{predicted_temperature - current_temperature:+.2f} °C"
        ),
    )


# =========================================================
# Battery state, telemetry and alerts
# =========================================================
left_panel, center_panel, right_panel = st.columns(
    [1.0, 1.35, 1.0]
)


with left_panel:
    st.markdown("### 🔋 Battery State")

    st.progress(
        int(
            np.clip(
                health_percentage,
                0,
                100,
            )
        )
    )

    st.metric(
        "State of Health",
        f"{health_percentage:.1f}%",
    )

    st.metric(
        "Capacity",
        f"{current_capacity:.4f} Ah",
    )

    st.caption(
        f"Reference capacity: "
        f"{reference_capacity:.4f} Ah"
    )


with center_panel:
    st.markdown("### 📡 Mission Telemetry")

    telemetry_1, telemetry_2 = st.columns(2)

    with telemetry_1:
        st.metric(
            "Current",
            f"{current_value:.4f} A",
        )

        st.caption(
            "Measured discharge current"
        )

    with telemetry_2:
        st.metric(
            "Mission Time",
            f"{current_time:.1f} s",
        )

        st.caption(
            f"Prediction horizon: "
            f"{next_time - current_time:.1f} s"
        )

    st.metric(
        "AI-Predicted RUL",
        rul_text,
    )

    st.caption(
        "Preliminary ML estimate based on "
        "cycle-level operating features."
    )


with right_panel:
    st.markdown("### 🚨 System Alerts")

    if temperature_state == "Critical":
        st.error(
            "Critical thermal condition detected. "
            "Stop discharge and inspect the cooling system."
        )

    elif voltage_state == "Critical":
        st.error(
            "Critical voltage condition detected. "
            "Stop discharge to prevent deep-discharge damage."
        )

    elif health_percentage <= 70:
        st.error(
            "Critical battery degradation detected. "
            "Replacement is strongly recommended."
        )

    elif health_percentage <= 80:
        st.error(
            "The battery has reached the conventional "
            "80% end-of-life threshold."
        )

    elif (
        temperature_state == "Warning"
        or voltage_state == "Warning"
        or health_state in [
            "Warning",
            "Degraded",
        ]
    ):
        st.warning(
            "Maintenance warning. Continue close monitoring "
            "and schedule an inspection."
        )

    else:
        st.success(
            "Nominal operation. "
            "No immediate corrective action is required."
        )


# =========================================================
# Digital twin gauges
# =========================================================
st.subheader("Digital Twin Gauges")

gauge_1, gauge_2 = st.columns(2)

with gauge_1:
    st.plotly_chart(
        create_gauge(
            current_voltage,
            2.5,
            4.3,
            "Voltage",
            " V",
            threshold=3.0,
        ),
        use_container_width=True,
        key="voltage_gauge",
    )

with gauge_2:
    st.plotly_chart(
        create_gauge(
            current_temperature,
            0,
            60,
            "Temperature",
            " °C",
            threshold=45,
        ),
        use_container_width=True,
        key="temperature_gauge",
    )


gauge_3, gauge_4 = st.columns(2)

with gauge_3:
    st.plotly_chart(
        create_gauge(
            health_percentage,
            0,
            100,
            "State of Health",
            "%",
            threshold=80,
        ),
        use_container_width=True,
        key="soh_gauge",
    )

with gauge_4:
    st.plotly_chart(
        create_gauge(
            rul_gauge_value,
            0,
            500,
            "AI-Predicted RUL",
            "",
            threshold=50,
        ),
        use_container_width=True,
        key="rul_gauge",
    )

# =========================================================
# Battery degradation trend
# =========================================================
st.subheader("Battery Degradation Trend")

cycle_capacity_df["SOH_percent"] = (
    cycle_capacity_df[capacity_column]
    / reference_capacity
    * 100
)

cycle_capacity_df["SOH_percent"] = np.clip(
    cycle_capacity_df["SOH_percent"],
    0,
    100,
)

degradation_figure = go.Figure()

degradation_figure.add_trace(
    go.Scatter(
        x=cycle_capacity_df[cycle_column],
        y=cycle_capacity_df["SOH_percent"],
        mode="lines+markers",
        name="SOH",
    )
)

degradation_figure.add_hline(
    y=80,
    line_dash="dash",
    annotation_text="80% End-of-Life Threshold",
)

degradation_figure.add_vline(
    x=selected_cycle,
    line_dash="dot",
    annotation_text="Selected cycle",
)

degradation_figure.update_layout(
    title=f"SOH Degradation — Battery {selected_battery}",
    xaxis_title="Discharge Cycle",
    yaxis_title="State of Health (%)",
    yaxis=dict(
        range=[0, 105],
    ),
    height=450,
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={
        "color": "white",
        "size": 14,
    },
)

st.plotly_chart(
    degradation_figure,
    use_container_width=True,
    key="degradation_trend",
)
# =========================================================
# Subsystem status
# =========================================================
st.subheader("Subsystem Status")

status_1, status_2, status_3 = st.columns(3)


with status_1:
    if voltage_state == "Normal":
        st.success(
            f"Voltage: {voltage_state}"
        )

    elif voltage_state == "Warning":
        st.warning(
            f"Voltage: {voltage_state}"
        )

    else:
        st.error(
            f"Voltage: {voltage_state}"
        )


with status_2:
    if temperature_state == "Normal":
        st.success(
            f"Temperature: {temperature_state}"
        )

    elif temperature_state == "Warning":
        st.warning(
            f"Temperature: {temperature_state}"
        )

    else:
        st.error(
            f"Temperature: {temperature_state}"
        )


with status_3:
    if health_state == "Healthy":
        st.success(
            f"Battery Health: {health_state}"
        )

    elif health_state in [
        "Warning",
        "Degraded",
    ]:
        st.warning(
            f"Battery Health: {health_state}"
        )

    else:
        st.error(
            f"Battery Health: {health_state}"
        )


# =========================================================
# Live telemetry charts
# =========================================================
st.subheader("Live Battery Telemetry")


voltage_figure = go.Figure()

voltage_figure.add_trace(
    go.Scatter(
        x=cycle_df[time_column],
        y=cycle_df[voltage_column],
        mode="lines",
        name="Measured voltage",
    )
)

voltage_figure.add_trace(
    go.Scatter(
        x=[
            current_time,
            next_time,
        ],
        y=[
            current_voltage,
            predicted_voltage,
        ],
        mode="lines+markers",
        name="AI next-step prediction",
        line=dict(
            dash="dash",
        ),
    )
)

voltage_figure.add_vline(
    x=current_time,
    line_dash="dot",
    annotation_text="Current mission time",
)

voltage_figure.update_layout(
    title="Voltage Profile",
    xaxis_title="Time (s)",
    yaxis_title="Voltage (V)",
    height=500,
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={
        "color": "white",
        "size": 14,
    },
)


temperature_figure = go.Figure()

temperature_figure.add_trace(
    go.Scatter(
        x=cycle_df[time_column],
        y=cycle_df[temperature_column],
        mode="lines",
        name="Measured temperature",
    )
)

temperature_figure.add_trace(
    go.Scatter(
        x=[
            current_time,
            next_time,
        ],
        y=[
            current_temperature,
            predicted_temperature,
        ],
        mode="lines+markers",
        name="AI next-step prediction",
        line=dict(
            dash="dash",
        ),
    )
)

temperature_figure.add_vline(
    x=current_time,
    line_dash="dot",
    annotation_text="Current mission time",
)

temperature_figure.update_layout(
    title="Temperature Profile",
    xaxis_title="Time (s)",
    yaxis_title="Temperature (°C)",
    height=500,
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={
        "color": "white",
        "size": 14,
    },
)


chart_1, chart_2 = st.columns(2)

with chart_1:
    st.plotly_chart(
        voltage_figure,
        use_container_width=True,
    )

with chart_2:
    st.plotly_chart(
        temperature_figure,
        use_container_width=True,
    )


# =========================================================
# AI recommendation
# =========================================================
st.subheader("AI Mission Recommendation")

if temperature_state == "Critical":
    st.error(
        "Critical battery temperature detected. "
        "Terminate discharge and inspect thermal management."
    )

elif voltage_state == "Critical":
    st.error(
        "Critical low voltage detected. "
        "Terminate discharge to prevent battery damage."
    )

elif health_percentage <= 70:
    st.error(
        "Battery health is critically degraded. "
        "Replacement is strongly recommended."
    )

elif health_percentage <= 80:
    st.warning(
        "The battery has reached the conventional "
        "end-of-life threshold. Replacement planning "
        "is recommended."
    )

elif health_percentage < 90:
    st.warning(
        "The battery shows measurable degradation. "
        "Continue close monitoring and plan maintenance."
    )

else:
    st.success(
        "The digital twin reports nominal battery operation. "
        "No immediate intervention is required."
    )


# =========================================================
# RUL transparency information
# =========================================================
with st.expander(
    "About the AI-based RUL prediction"
):
    st.write(
        "The RUL value is generated by a "
        "HistGradientBoosting regression pipeline trained on "
        "cycle-level features from multiple NASA batteries."
    )

    st.write(
        "The current unseen-battery evaluation produced "
        "approximately 18 cycles MAE, 25 cycles RMSE and "
        "an R² of about 0.35."
    )

    st.write(
        "The target represents cycles remaining until the "
        "final recorded cycle in the available dataset. "
        "It should be treated as a preliminary data-driven "
        "estimate rather than a certified failure prediction."
    )


# =========================================================
# Data inspection
# =========================================================
with st.expander(
    "View current mission data"
):
    st.dataframe(
        cycle_df,
        use_container_width=True,
    )
render_footer()