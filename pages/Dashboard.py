from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


# ---------------------------------------------------------
# Import project utilities
# ---------------------------------------------------------
from utils.data_loader import get_battery_files, load_battery_data
from utils.model_loader import load_models
from utils.prediction import predict_next_state
from utils.helpers import battery_status
from utils.charts import voltage_chart, temperature_chart


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="NASA Battery Digital Twin",
    page_icon="🔋",
    layout="wide"
)


# ---------------------------------------------------------
# Custom style
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #9ca3af;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    .status-normal {
        padding: 8px 14px;
        border-radius: 8px;
        background-color: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.50);
        text-align: center;
        font-weight: 700;
    }

    .status-warning {
        padding: 8px 14px;
        border-radius: 8px;
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.50);
        text-align: center;
        font-weight: 700;
    }

    .status-critical {
        padding: 8px 14px;
        border-radius: 8px;
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.50);
        text-align: center;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.20);
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def find_column(dataframe, possible_names):
    for name in possible_names:
        if name in dataframe.columns:
            return name

    return None


def show_status(label, value):
    value_lower = str(value).lower()

    if value_lower in ["normal", "healthy"]:
        css_class = "status-normal"
    elif value_lower in ["warning", "degraded"]:
        css_class = "status-warning"
    else:
        css_class = "status-critical"

    st.markdown(
        f"""
        <div class="{css_class}">
            {label}: {value}
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<p class="main-title">🚀 NASA Battery Digital Twin</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitle">
    AI-powered battery monitoring, next-step prediction and health assessment
    </p>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Load battery files
# ---------------------------------------------------------
battery_files = get_battery_files()

if not battery_files:
    st.error(
        "No processed battery datasets were found inside "
        "`processed_datasets`."
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Digital Twin Controls")

selected_battery = st.sidebar.selectbox(
    "Select battery",
    options=list(battery_files.keys())
)

battery_df = load_battery_data(
    battery_files[selected_battery]
).copy()


# ---------------------------------------------------------
# Detect column names
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

voltage_column = find_column(
    battery_df,
    [
        "voltage_measured_V",
        "Voltage_measured",
        "voltage_V"
    ]
)

temperature_column = find_column(
    battery_df,
    [
        "temperature_measured_C",
        "Temperature_measured",
        "temperature_C"
    ]
)

current_column = find_column(
    battery_df,
    [
        "current_measured_A",
        "Current_measured",
        "current_A"
    ]
)

capacity_column = find_column(
    battery_df,
    [
        "capacity_Ah",
        "capacity",
        "Capacity"
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


required_columns = {
    "cycle": cycle_column,
    "time": time_column,
    "voltage": voltage_column,
    "temperature": temperature_column
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


# ---------------------------------------------------------
# Select discharge cycle
# ---------------------------------------------------------
available_cycles = sorted(
    battery_df[cycle_column]
    .dropna()
    .unique()
    .tolist()
)

selected_cycle = st.sidebar.selectbox(
    "Select discharge cycle",
    options=available_cycles
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
        "This cycle does not contain enough rows for prediction."
    )
    st.stop()


# ---------------------------------------------------------
# Select current time step
# ---------------------------------------------------------
maximum_index = len(cycle_df) - 2

selected_index = st.sidebar.slider(
    "Select current time step",
    min_value=0,
    max_value=maximum_index,
    value=min(10, maximum_index),
    step=1
)


current_row = cycle_df.iloc[selected_index]
next_row = cycle_df.iloc[selected_index + 1]

current_time = float(current_row[time_column])
next_time = float(next_row[time_column])

current_voltage = float(current_row[voltage_column])
actual_next_voltage = float(next_row[voltage_column])

current_temperature = float(
    current_row[temperature_column]
)

actual_next_temperature = float(
    next_row[temperature_column]
)


# ---------------------------------------------------------
# Load models and generate predictions
# ---------------------------------------------------------
try:
    voltage_model, temperature_model, feature_list = load_models()

    predicted_voltage, predicted_temperature = predict_next_state(
        current_row,
        voltage_model,
        temperature_model,
        feature_list
    )

except Exception as error:
    st.error("The trained models could not generate a prediction.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Capacity and status
# ---------------------------------------------------------
if capacity_column is not None:
    capacity_value = float(current_row[capacity_column])
else:
    capacity_value = np.nan


voltage_state, temperature_state, health_state = battery_status(
    current_voltage,
    current_temperature,
    capacity_value if not np.isnan(capacity_value) else 1.4
)


# ---------------------------------------------------------
# Main metrics
# ---------------------------------------------------------
st.subheader(
    f"Battery {selected_battery} — Discharge Cycle {selected_cycle}"
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Current Voltage",
        f"{current_voltage:.4f} V"
    )

with metric_2:
    st.metric(
        "Predicted Next Voltage",
        f"{predicted_voltage:.4f} V",
        delta=f"{predicted_voltage - current_voltage:+.4f} V"
    )

with metric_3:
    st.metric(
        "Current Temperature",
        f"{current_temperature:.2f} °C"
    )

with metric_4:
    st.metric(
        "Predicted Next Temperature",
        f"{predicted_temperature:.2f} °C",
        delta=f"{predicted_temperature - current_temperature:+.2f} °C"
    )


metric_5, metric_6, metric_7, metric_8 = st.columns(4)

with metric_5:
    if current_column is not None:
        current_value = float(current_row[current_column])
        st.metric(
            "Current",
            f"{current_value:.3f} A"
        )
    else:
        st.metric("Current", "Not available")

with metric_6:
    if not np.isnan(capacity_value):
        st.metric(
            "Capacity",
            f"{capacity_value:.4f} Ah"
        )
    else:
        st.metric("Capacity", "Not available")

with metric_7:
    st.metric(
        "Current Time",
        f"{current_time:.1f} s"
    )

with metric_8:
    st.metric(
        "Prediction Horizon",
        f"{next_time - current_time:.1f} s"
    )


# ---------------------------------------------------------
# System status
# ---------------------------------------------------------
st.subheader("System Status")

status_1, status_2, status_3 = st.columns(3)

with status_1:
    show_status("Voltage", voltage_state)

with status_2:
    show_status("Temperature", temperature_state)

with status_3:
    show_status("Battery Health", health_state)


# ---------------------------------------------------------
# Prediction errors
# ---------------------------------------------------------
voltage_error = predicted_voltage - actual_next_voltage
temperature_error = (
    predicted_temperature - actual_next_temperature
)

st.subheader("AI Prediction Performance")

error_1, error_2, error_3, error_4 = st.columns(4)

with error_1:
    st.metric(
        "Actual Next Voltage",
        f"{actual_next_voltage:.4f} V"
    )

with error_2:
    st.metric(
        "Voltage Error",
        f"{voltage_error:.5f} V"
    )

with error_3:
    st.metric(
        "Actual Next Temperature",
        f"{actual_next_temperature:.2f} °C"
    )

with error_4:
    st.metric(
        "Temperature Error",
        f"{temperature_error:.4f} °C"
    )


# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------
chart_df = cycle_df.rename(
    columns={
        time_column: "time_s",
        voltage_column: "voltage_measured_V",
        temperature_column: "temperature_measured_C"
    }
)

left_chart, right_chart = st.columns(2)

with left_chart:
    try:
        voltage_figure = voltage_chart(
            chart_df,
            current_time,
            predicted_voltage,
            actual_next_voltage
        )

        st.plotly_chart(
            voltage_figure,
            use_container_width=True
        )

    except Exception:
        st.line_chart(
            chart_df.set_index("time_s")[
                "voltage_measured_V"
            ]
        )


with right_chart:
    try:
        temperature_figure = temperature_chart(
            chart_df,
            current_time,
            predicted_temperature,
            actual_next_temperature
        )

        st.plotly_chart(
            temperature_figure,
            use_container_width=True
        )

    except Exception:
        st.line_chart(
            chart_df.set_index("time_s")[
                "temperature_measured_C"
            ]
        )


# ---------------------------------------------------------
# Current profile
# ---------------------------------------------------------
if current_column is not None:
    st.subheader("Discharge Current Profile")

    current_chart_df = cycle_df[
        [time_column, current_column]
    ].copy()

    current_chart_df = current_chart_df.rename(
        columns={
            time_column: "Time (s)",
            current_column: "Current (A)"
        }
    )

    st.line_chart(
        current_chart_df.set_index("Time (s)")
    )


# ---------------------------------------------------------
# AI recommendation
# ---------------------------------------------------------
st.subheader("AI Recommendation")

if temperature_state == "Critical":
    st.error(
        "Critical temperature detected. Stop the discharge process "
        "and inspect the battery cooling and thermal-management system."
    )

elif voltage_state == "Critical":
    st.error(
        "Critical voltage level detected. End the discharge cycle "
        "to prevent deep-discharge damage."
    )

elif (
    temperature_state == "Warning"
    or voltage_state == "Warning"
    or health_state == "Degraded"
):
    st.warning(
        "Battery degradation or an operating warning has been detected. "
        "Continue monitoring and consider maintenance inspection."
    )

else:
    st.success(
        "The battery is operating within the current safety limits. "
        "No immediate corrective action is required."
    )


# ---------------------------------------------------------
# Data preview
# ---------------------------------------------------------
with st.expander("View selected cycle data"):
    st.dataframe(
        cycle_df,
        use_container_width=True
    )