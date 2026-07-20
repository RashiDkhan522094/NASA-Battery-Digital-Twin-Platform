from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


PROJECT_FOLDER = Path(__file__).resolve().parents[1]

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


from utils.mission_control import (
    classify_operating_state,
    cycle_capacity_table,
    cycle_soh_percent,
    detect_columns,
    list_battery_files,
    load_battery_dataset,
    mission_confidence,
    reference_capacity,
    validate_required_columns,
)


st.set_page_config(
    page_title="Live Mission Control",
    page_icon="🛰️",
    layout="wide",
)


PROCESSED_FOLDER = (
    PROJECT_FOLDER
    / "processed_datasets"
)


st.markdown(
    """
    <style>
    .mission-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .mission-subtitle {
        color: #94a3b8;
        margin-bottom: 1.3rem;
    }

    .status-card {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 1.0rem 1.1rem;
        background: rgba(15, 23, 42, 0.55);
    }

    .mission-log {
        font-family: Consolas, monospace;
        font-size: 0.92rem;
        line-height: 1.55;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        padding: 0.9rem;
        background: rgba(2, 6, 23, 0.68);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="mission-title">🛰️ Live NASA Battery Mission Control</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="mission-subtitle">'
    'Live replay of measured battery telemetry with next-step '
    'forecasting, health assessment and automatic alerts.'
    '</div>',
    unsafe_allow_html=True,
)


battery_files = list_battery_files(
    PROCESSED_FOLDER
)

if not battery_files:
    st.error(
        "No processed battery datasets were found in "
        f"{PROCESSED_FOLDER}."
    )
    st.stop()


with st.sidebar:
    st.header("Mission Controls")

    selected_battery = st.selectbox(
        "Battery",
        list(battery_files.keys()),
    )


battery_df = load_battery_dataset(
    battery_files[selected_battery]
)

columns = detect_columns(
    battery_df
)

validate_required_columns(
    columns
)

cycle_column = columns["cycle"]
time_column = columns["time"]
voltage_column = columns["voltage"]
temperature_column = columns["temperature"]
current_column = columns["current"]
capacity_column = columns["capacity"]
next_voltage_column = columns["next_voltage"]
next_temperature_column = columns["next_temperature"]


available_cycles = (
    pd.to_numeric(
        battery_df[cycle_column],
        errors="coerce",
    )
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

with st.sidebar:
    selected_cycle = st.selectbox(
        "Discharge cycle",
        available_cycles,
    )


cycle_df = (
    battery_df[
        pd.to_numeric(
            battery_df[cycle_column],
            errors="coerce",
        )
        == float(selected_cycle)
    ]
    .copy()
    .sort_values(time_column)
    .reset_index(drop=True)
)

if len(cycle_df) < 2:
    st.warning(
        "The selected cycle contains fewer than two telemetry samples."
    )
    st.stop()


scenario_key = (
    selected_battery,
    float(selected_cycle),
)

if (
    st.session_state.get(
        "mission_scenario_key"
    )
    != scenario_key
):
    st.session_state[
        "mission_scenario_key"
    ] = scenario_key

    st.session_state[
        "mission_step"
    ] = 0

    st.session_state[
        "mission_playing"
    ] = False


max_step = len(cycle_df) - 1

with st.sidebar:
    control_1, control_2, control_3 = st.columns(3)

    with control_1:
        if st.button(
            "▶",
            help="Play live telemetry",
            use_container_width=True,
        ):
            st.session_state[
                "mission_playing"
            ] = True

    with control_2:
        if st.button(
            "⏸",
            help="Pause live telemetry",
            use_container_width=True,
        ):
            st.session_state[
                "mission_playing"
            ] = False

    with control_3:
        if st.button(
            "↺",
            help="Reset mission",
            use_container_width=True,
        ):
            st.session_state[
                "mission_playing"
            ] = False

            st.session_state[
                "mission_step"
            ] = 0

    selected_step = st.slider(
        "Mission time step",
        min_value=0,
        max_value=max_step,
        value=min(
            int(
                st.session_state.get(
                    "mission_step",
                    0,
                )
            ),
            max_step,
        ),
        step=1,
    )

    st.session_state[
        "mission_step"
    ] = selected_step

    replay_speed = st.slider(
        "Replay interval (seconds)",
        min_value=0.2,
        max_value=2.0,
        value=0.8,
        step=0.1,
    )


row = cycle_df.iloc[selected_step]

voltage = float(
    pd.to_numeric(
        row[voltage_column],
        errors="coerce",
    )
)

temperature = float(
    pd.to_numeric(
        row[temperature_column],
        errors="coerce",
    )
)

current = float(
    pd.to_numeric(
        row[current_column],
        errors="coerce",
    )
)

mission_time = float(
    pd.to_numeric(
        row[time_column],
        errors="coerce",
    )
)

capacity = float(
    pd.to_numeric(
        row[capacity_column],
        errors="coerce",
    )
)


capacity_df = cycle_capacity_table(
    battery_df,
    cycle_column,
    capacity_column,
)

reference_capacity_value = reference_capacity(
    capacity_df,
    capacity_column,
)

soh = cycle_soh_percent(
    capacity,
    reference_capacity_value,
)


if (
    next_voltage_column is not None
    and pd.notna(row[next_voltage_column])
):
    predicted_voltage = float(
        row[next_voltage_column]
    )
else:
    predicted_voltage = float(
        cycle_df.iloc[
            min(selected_step + 1, max_step)
        ][voltage_column]
    )


if (
    next_temperature_column is not None
    and pd.notna(row[next_temperature_column])
):
    predicted_temperature = float(
        row[next_temperature_column]
    )
else:
    predicted_temperature = float(
        cycle_df.iloc[
            min(selected_step + 1, max_step)
        ][temperature_column]
    )


actual_next_row = cycle_df.iloc[
    min(selected_step + 1, max_step)
]

actual_next_voltage = float(
    actual_next_row[voltage_column]
)

actual_next_temperature = float(
    actual_next_row[temperature_column]
)

voltage_error = (
    predicted_voltage
    - actual_next_voltage
)

temperature_error = (
    predicted_temperature
    - actual_next_temperature
)

confidence = mission_confidence(
    voltage_error,
    temperature_error,
)

operating_state, recommendation = (
    classify_operating_state(
        voltage=voltage,
        temperature=temperature,
        soh=soh,
    )
)


if operating_state == "CRITICAL":
    status_function = st.error
elif operating_state == "HIGH RISK":
    status_function = st.error
elif operating_state == "WARNING":
    status_function = st.warning
else:
    status_function = st.success


header_col_1, header_col_2 = st.columns(
    [1.0, 0.38]
)

with header_col_1:
    st.subheader(
        f"Mission Status — Battery {selected_battery}, "
        f"Cycle {selected_cycle}"
    )

with header_col_2:
    st.metric(
        "Replay Progress",
        f"{selected_step}/{max_step}",
    )


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Current Voltage",
        f"{voltage:.4f} V",
    )

with metric_2:
    st.metric(
        "AI Next Voltage",
        f"{predicted_voltage:.4f} V",
        delta=f"{voltage_error:+.4f} V",
    )

with metric_3:
    st.metric(
        "Current Temperature",
        f"{temperature:.2f} °C",
    )

with metric_4:
    st.metric(
        "AI Next Temperature",
        f"{predicted_temperature:.2f} °C",
        delta=f"{temperature_error:+.2f} °C",
    )


left_panel, middle_panel, right_panel = st.columns(
    [0.9, 1.0, 0.9]
)

with left_panel:
    st.markdown("### 🔋 Battery State")

    st.progress(
        int(
            np.clip(
                soh,
                0,
                100,
            )
        )
    )

    st.metric(
        "State of Health",
        f"{soh:.1f}%",
    )

    st.metric(
        "Capacity",
        f"{capacity:.4f} Ah",
    )

    st.caption(
        f"Reference capacity: "
        f"{reference_capacity_value:.4f} Ah"
    )

with middle_panel:
    st.markdown("### 📡 Mission Telemetry")

    telemetry_1, telemetry_2 = st.columns(2)

    with telemetry_1:
        st.metric(
            "Current",
            f"{current:.4f} A",
        )

    with telemetry_2:
        st.metric(
            "Mission Time",
            f"{mission_time:.1f} s",
        )

    st.metric(
        "AI Confidence",
        f"{confidence:.0f}%",
    )

    st.caption(
        "Confidence is derived from next-step voltage and "
        "temperature prediction errors."
    )

with right_panel:
    st.markdown("### 🚨 System Status")

    status_function(
        f"{operating_state}: {recommendation}"
    )

    if voltage < 3.0:
        st.error(
            "Critical voltage threshold crossed."
        )
    elif voltage < 3.2:
        st.warning(
            "Voltage is approaching the lower operating limit."
        )
    else:
        st.success(
            "Voltage subsystem nominal."
        )

    if temperature >= 45:
        st.error(
            "Thermal warning threshold crossed."
        )
    elif temperature >= 40:
        st.warning(
            "Temperature is elevated."
        )
    else:
        st.success(
            "Thermal subsystem nominal."
        )


st.subheader("Digital Twin Gauges")

gauge_figure = make_subplots(
    rows=1,
    cols=4,
    specs=[
        [{"type": "indicator"}] * 4
    ],
)

gauge_specs = [
    (
        voltage,
        "Voltage",
        " V",
        [2.5, 4.3],
    ),
    (
        temperature,
        "Temperature",
        " °C",
        [0, 60],
    ),
    (
        soh,
        "SOH",
        "%",
        [0, 100],
    ),
    (
        confidence,
        "AI Confidence",
        "%",
        [0, 100],
    ),
]

for index, (
    value,
    title,
    suffix,
    axis_range,
) in enumerate(gauge_specs, start=1):
    gauge_figure.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix},
            title={"text": title},
            gauge={
                "axis": {"range": axis_range},
                "bar": {"thickness": 0.35},
            },
        ),
        row=1,
        col=index,
    )

gauge_figure.update_layout(
    height=300,
    margin=dict(
        l=20,
        r=20,
        t=55,
        b=10,
    ),
)

st.plotly_chart(
    gauge_figure,
    use_container_width=True,
)


st.subheader("Live Telemetry")

chart_col_1, chart_col_2 = st.columns(2)

visible_df = cycle_df.iloc[
    : selected_step + 1
]

with chart_col_1:
    voltage_figure = go.Figure()

    voltage_figure.add_trace(
        go.Scatter(
            x=visible_df[time_column],
            y=visible_df[voltage_column],
            mode="lines",
            name="Measured voltage",
        )
    )

    voltage_figure.add_trace(
        go.Scatter(
            x=[mission_time],
            y=[voltage],
            mode="markers",
            name="Current mission point",
            marker={"size": 13},
        )
    )

    voltage_figure.add_trace(
        go.Scatter(
            x=[
                float(
                    actual_next_row[
                        time_column
                    ]
                )
            ],
            y=[predicted_voltage],
            mode="markers",
            name="AI next-step forecast",
            marker={
                "size": 12,
                "symbol": "diamond",
            },
        )
    )

    voltage_figure.update_layout(
        title="Voltage Profile",
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
        height=420,
    )

    st.plotly_chart(
        voltage_figure,
        use_container_width=True,
    )

with chart_col_2:
    temperature_figure = go.Figure()

    temperature_figure.add_trace(
        go.Scatter(
            x=visible_df[time_column],
            y=visible_df[temperature_column],
            mode="lines",
            name="Measured temperature",
        )
    )

    temperature_figure.add_trace(
        go.Scatter(
            x=[mission_time],
            y=[temperature],
            mode="markers",
            name="Current mission point",
            marker={"size": 13},
        )
    )

    temperature_figure.add_trace(
        go.Scatter(
            x=[
                float(
                    actual_next_row[
                        time_column
                    ]
                )
            ],
            y=[predicted_temperature],
            mode="markers",
            name="AI next-step forecast",
            marker={
                "size": 12,
                "symbol": "diamond",
            },
        )
    )

    temperature_figure.update_layout(
        title="Temperature Profile",
        xaxis_title="Time (s)",
        yaxis_title="Temperature (°C)",
        height=420,
    )

    st.plotly_chart(
        temperature_figure,
        use_container_width=True,
    )


st.subheader("Mission Event Log")

events = [
    (
        f"{mission_time:8.1f} s | "
        f"Telemetry sample {selected_step}/{max_step}"
    ),
    (
        f"{mission_time:8.1f} s | "
        f"Voltage {voltage:.4f} V"
    ),
    (
        f"{mission_time:8.1f} s | "
        f"Temperature {temperature:.2f} °C"
    ),
    (
        f"{mission_time:8.1f} s | "
        f"SOH {soh:.1f}%"
    ),
    (
        f"{mission_time:8.1f} s | "
        f"System state: {operating_state}"
    ),
    (
        f"{mission_time:8.1f} s | "
        f"AI confidence: {confidence:.0f}%"
    ),
]

st.markdown(
    '<div class="mission-log">'
    + "<br>".join(events)
    + "</div>",
    unsafe_allow_html=True,
)


st.subheader("AI Mission Recommendation")

status_function(
    recommendation
)


# Automatic replay:
# each rerun advances one sample and then triggers the next rerun.
if st.session_state.get(
    "mission_playing",
    False,
):
    if selected_step < max_step:
        time.sleep(replay_speed)

        st.session_state[
            "mission_step"
        ] = selected_step + 1

        st.rerun()
    else:
        st.session_state[
            "mission_playing"
        ] = False

        st.success(
            "Mission replay completed."
        )
