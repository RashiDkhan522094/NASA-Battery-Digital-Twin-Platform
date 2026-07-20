from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


from utils.data_loader import get_battery_files, load_battery_data


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Battery Health",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def find_column(dataframe, possible_names):
    for name in possible_names:
        if name in dataframe.columns:
            return name
    return None


def classify_health(soh):
    if soh >= 80:
        return "Healthy"
    elif soh >= 70:
        return "Degraded"
    return "End of Life"


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🩺 Battery Health Analysis")

st.caption(
    "Capacity degradation, state of health and remaining useful life estimation"
)


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------
battery_files = get_battery_files()

if not battery_files:
    st.error("No processed battery datasets were found.")
    st.stop()


selected_battery = st.sidebar.selectbox(
    "Select battery",
    options=list(battery_files.keys())
)

battery_df = load_battery_data(
    battery_files[selected_battery]
).copy()


# ---------------------------------------------------------
# Detect columns
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

capacity_column = find_column(
    battery_df,
    [
        "capacity_Ah",
        "capacity",
        "Capacity"
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

voltage_column = find_column(
    battery_df,
    [
        "voltage_measured_V",
        "Voltage_measured",
        "voltage_V"
    ]
)


if cycle_column is None or capacity_column is None:
    st.error(
        "Cycle or capacity column could not be found in this dataset."
    )
    st.write("Available columns:")
    st.write(list(battery_df.columns))
    st.stop()


# ---------------------------------------------------------
# Create cycle-level health table
# ---------------------------------------------------------
aggregation = {
    capacity_column: "first"
}

if temperature_column is not None:
    aggregation[temperature_column] = "max"

if voltage_column is not None:
    aggregation[voltage_column] = "min"


health_df = (
    battery_df
    .groupby(cycle_column, as_index=False)
    .agg(aggregation)
    .sort_values(cycle_column)
    .reset_index(drop=True)
)


health_df = health_df.rename(
    columns={
        cycle_column: "Cycle",
        capacity_column: "Capacity_Ah"
    }
)


initial_capacity = float(
    health_df["Capacity_Ah"].iloc[0]
)

health_df["SOH_percent"] = (
    health_df["Capacity_Ah"]
    / initial_capacity
    * 100
)


eol_threshold_capacity = initial_capacity * 0.80

health_df["Health_Status"] = health_df[
    "SOH_percent"
].apply(classify_health)


# ---------------------------------------------------------
# Current battery health
# ---------------------------------------------------------
latest_row = health_df.iloc[-1]

latest_cycle = int(latest_row["Cycle"])
latest_capacity = float(latest_row["Capacity_Ah"])
latest_soh = float(latest_row["SOH_percent"])
latest_status = latest_row["Health_Status"]


# ---------------------------------------------------------
# Estimate RUL using linear degradation
# ---------------------------------------------------------
valid_health_df = health_df.dropna(
    subset=["Cycle", "Capacity_Ah"]
)

if len(valid_health_df) >= 2:
    slope, intercept = np.polyfit(
        valid_health_df["Cycle"],
        valid_health_df["Capacity_Ah"],
        1
    )

    if slope < 0:
        predicted_eol_cycle = (
            eol_threshold_capacity - intercept
        ) / slope

        estimated_rul = max(
            0,
            predicted_eol_cycle - latest_cycle
        )
    else:
        predicted_eol_cycle = np.nan
        estimated_rul = np.nan
else:
    slope = np.nan
    intercept = np.nan
    predicted_eol_cycle = np.nan
    estimated_rul = np.nan


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Initial Capacity",
        f"{initial_capacity:.4f} Ah"
    )

with metric_2:
    st.metric(
        "Current Capacity",
        f"{latest_capacity:.4f} Ah",
        delta=f"{latest_capacity - initial_capacity:.4f} Ah"
    )

with metric_3:
    st.metric(
        "State of Health",
        f"{latest_soh:.2f}%"
    )

with metric_4:

    if latest_soh <= 80:
        st.metric(
            "Estimated RUL",
            "EOL"
        )

    elif np.isnan(estimated_rul):
        st.metric(
            "Estimated RUL",
            "Unknown"
        )

    else:
        st.metric(
            "Estimated RUL",
            f"{estimated_rul:.0f} cycles"
        )

st.subheader(f"Battery Status: {latest_status}")


# ---------------------------------------------------------
# Capacity degradation chart
# ---------------------------------------------------------
capacity_figure = go.Figure()

capacity_figure.add_trace(
    go.Scatter(
        x=health_df["Cycle"],
        y=health_df["Capacity_Ah"],
        mode="lines+markers",
        name="Measured capacity"
    )
)

capacity_figure.add_hline(
    y=eol_threshold_capacity,
    line_dash="dash",
    annotation_text="80% End-of-Life Threshold"
)

if not np.isnan(predicted_eol_cycle):
    capacity_figure.add_vline(
        x=predicted_eol_cycle,
        line_dash="dot",
        annotation_text="Predicted EOL"
    )

capacity_figure.update_layout(
    title=f"Capacity Degradation — Battery {selected_battery}",
    xaxis_title="Discharge Cycle",
    yaxis_title="Capacity (Ah)",
    hovermode="x unified"
)

st.plotly_chart(
    capacity_figure,
    use_container_width=True
)


# ---------------------------------------------------------
# SOH chart
# ---------------------------------------------------------
soh_figure = go.Figure()

soh_figure.add_trace(
    go.Scatter(
        x=health_df["Cycle"],
        y=health_df["SOH_percent"],
        mode="lines+markers",
        name="SOH"
    )
)

soh_figure.add_hline(
    y=80,
    line_dash="dash",
    annotation_text="80% SOH Threshold"
)

soh_figure.update_layout(
    title="State of Health Across Battery Life",
    xaxis_title="Discharge Cycle",
    yaxis_title="SOH (%)",
    hovermode="x unified"
)

st.plotly_chart(
    soh_figure,
    use_container_width=True
)


# ---------------------------------------------------------
# Degradation summary
# ---------------------------------------------------------
capacity_loss = initial_capacity - latest_capacity
capacity_loss_percent = 100 - latest_soh

st.subheader("Degradation Summary")

summary_1, summary_2, summary_3 = st.columns(3)

with summary_1:
    st.metric(
        "Capacity Loss",
        f"{capacity_loss:.4f} Ah"
    )

with summary_2:
    st.metric(
        "Capacity Loss Percentage",
        f"{capacity_loss_percent:.2f}%"
    )

with summary_3:
    st.metric(
        "Latest Recorded Cycle",
        f"{latest_cycle}"
    )


# ---------------------------------------------------------
# Health recommendation
# ---------------------------------------------------------
st.subheader("Maintenance Recommendation")

if latest_soh >= 90:
    st.success(
        "The battery is in strong condition. Continue normal operation "
        "and routine monitoring."
    )

elif latest_soh >= 80:
    st.info(
        "The battery remains operational, but measurable degradation "
        "has occurred. Continue monitoring capacity and temperature."
    )

elif latest_soh >= 70:
    st.warning(
        "The battery has fallen below the conventional 80% SOH "
        "threshold. Schedule maintenance or replacement planning."
    )

else:
    st.error(
        "The battery is approaching or has reached end-of-life. "
        "Replacement is strongly recommended."
    )


# ---------------------------------------------------------
# Table
# ---------------------------------------------------------
with st.expander("View cycle-level health data"):
    st.dataframe(
        health_df,
        use_container_width=True
    )