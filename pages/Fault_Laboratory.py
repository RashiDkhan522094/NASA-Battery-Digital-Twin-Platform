from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from utils.incident_storage import save_incident_bundle


PROJECT_FOLDER = Path(__file__).resolve().parents[1]

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


from utils.data_loader import (
    get_battery_files,
    load_battery_data,
)

from utils.model_loader import load_models

from utils.prediction import predict_next_state

from utils.simulation_engine import (
    build_recursive_simulation,
    data_informed_bounds,
)

from utils.fault_engine import (
    FAULT_LIBRARY,
    available_faults,
    calculate_fault_risk,
    inject_fault,
)

from utils.maintenance_engine import (
    generate_maintenance_recommendation,
)

from utils.incident_report import (
    build_event_log,
    create_incident_id,
    generate_incident_pdf,
)


st.set_page_config(
    page_title="Fault Injection Laboratory",
    page_icon="🛠️",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(239,68,68,0.10),
                transparent 30%
            ),
            #08111f;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(148,163,184,0.18);
    }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.95);
        border: 1px solid rgba(148,163,184,0.20);
        border-radius: 16px;
        padding: 15px;
    }

    div[data-testid="stPlotlyChart"] {
        background: rgba(15,23,42,0.55);
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 16px;
        padding: 6px;
    }

    .fault-title {
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .fault-subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_column(dataframe, possible_names):
    for name in possible_names:
        if name in dataframe.columns:
            return name
    return None


def calculate_reference_capacity(
    battery_df,
    cycle_column,
    capacity_column,
):
    cycle_capacity_df = (
        battery_df
        .groupby(
            cycle_column,
            as_index=False,
        )[capacity_column]
        .median()
        .sort_values(cycle_column)
        .reset_index(drop=True)
    )

    cycle_capacity_df[capacity_column] = pd.to_numeric(
        cycle_capacity_df[capacity_column],
        errors="coerce",
    )

    cycle_capacity_df = cycle_capacity_df.dropna(
        subset=[capacity_column]
    )

    cycle_capacity_df = cycle_capacity_df[
        cycle_capacity_df[capacity_column] > 0
    ].copy()

    if cycle_capacity_df.empty:
        return np.nan

    reference_window = cycle_capacity_df.head(
        min(20, len(cycle_capacity_df))
    )

    return float(
        reference_window[capacity_column].max()
    )


def render_fault_battery(
    soh,
    voltage,
    temperature,
    risk_level,
    fault_type,
):
    if risk_level == "Critical":
        color = "#ef4444"
        glow = "rgba(239,68,68,0.65)"
    elif risk_level == "High":
        color = "#f97316"
        glow = "rgba(249,115,22,0.60)"
    elif risk_level == "Moderate":
        color = "#facc15"
        glow = "rgba(250,204,21,0.55)"
    else:
        color = "#22c55e"
        glow = "rgba(34,197,94,0.55)"

    fill_level = float(np.clip(soh, 4, 100))

    html = f"""
    <html>
    <head>
    <style>
    body {{
        margin: 0;
        background: transparent;
        color: white;
        font-family: Arial, sans-serif;
    }}

    .card {{
        height: 475px;
        border-radius: 22px;
        border: 1px solid rgba(148,163,184,0.20);
        background:
            radial-gradient(
                circle at center,
                {glow},
                rgba(15,23,42,0.97) 52%
            );
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }}

    .terminal {{
        width: 78px;
        height: 24px;
        background: linear-gradient(180deg,#f8fafc,#94a3b8);
        border-radius: 10px 10px 2px 2px;
    }}

    .shell {{
        position: relative;
        width: 190px;
        height: 310px;
        border: 8px solid #dbeafe;
        border-radius: 28px;
        box-sizing: border-box;
        background: rgba(2,6,23,0.92);
        box-shadow:
            0 0 30px {glow},
            inset 0 0 20px rgba(255,255,255,0.06);
        overflow: hidden;
    }}

    .fill {{
        position: absolute;
        left: 11px;
        right: 11px;
        bottom: 11px;
        height: {fill_level}%;
        max-height: calc(100% - 22px);
        border-radius: 14px;
        background:
            linear-gradient(
                180deg,
                rgba(255,255,255,0.30),
                {color}
            );
        box-shadow: 0 0 28px {glow};
        animation: pulse 1.6s ease-in-out infinite;
    }}

    .percentage {{
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 38px;
        font-weight: 900;
        z-index: 4;
        text-shadow: 0 2px 14px rgba(0,0,0,0.70);
    }}

    .fault-wave {{
        position: absolute;
        left: -25%;
        width: 150%;
        height: 6px;
        background: white;
        box-shadow: 0 0 18px white;
        animation: scan 1.4s linear infinite;
        opacity: 0.8;
        z-index: 5;
    }}

    @keyframes pulse {{
        0%,100% {{ filter: brightness(0.90); }}
        50% {{ filter: brightness(1.22); }}
    }}

    @keyframes scan {{
        0% {{ bottom: 12%; opacity: 0; }}
        15% {{ opacity: 1; }}
        85% {{ opacity: 1; }}
        100% {{ bottom: 88%; opacity: 0; }}
    }}

    .risk {{
        margin-top: 22px;
        font-size: 27px;
        font-weight: 900;
        color: {color};
        letter-spacing: 1.5px;
    }}

    .fault {{
        margin-top: 8px;
        color: #cbd5e1;
        font-size: 16px;
    }}

    .telemetry {{
        margin-top: 14px;
        display: flex;
        gap: 12px;
    }}

    .pill {{
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(15,23,42,0.76);
        border: 1px solid rgba(148,163,184,0.20);
    }}
    </style>
    </head>

    <body>
        <div class="card">
            <div class="terminal"></div>
            <div class="shell">
                <div class="fill"></div>
                <div class="fault-wave"></div>
                <div class="percentage">{soh:.1f}%</div>
            </div>

            <div class="risk">{risk_level.upper()} RISK</div>
            <div class="fault">{fault_type}</div>

            <div class="telemetry">
                <span class="pill">V: {voltage:.2f}</span>
                <span class="pill">T: {temperature:.1f} °C</span>
            </div>
        </div>
    </body>
    </html>
    """

    components.html(
        html,
        height=500,
        scrolling=False,
    )



def create_automatic_fault_replay(
    fault_df,
    fault_type,
):
    """
    Create a professional automatic replay dashboard with compact
    gauge titles, larger gauges, clean tick spacing, synchronized
    telemetry markers and animated fault progression.
    """
    frame_count = len(fault_df)

    if frame_count == 0:
        return go.Figure()

    time_values = fault_df["time_s"].to_numpy(dtype=float)
    voltage_values = fault_df["true_voltage_V"].to_numpy(dtype=float)
    temperature_values = fault_df[
        "true_temperature_C"
    ].to_numpy(dtype=float)
    soh_values = fault_df[
        "fault_soh_percent"
    ].to_numpy(dtype=float)

    risk_scores = []
    risk_levels = []

    for index in range(frame_count):
        risk_result = calculate_fault_risk(
            final_voltage=float(voltage_values[index]),
            max_temperature=float(
                np.max(temperature_values[: index + 1])
            ),
            final_soh=float(soh_values[index]),
        )
        risk_scores.append(risk_result["risk_score"])
        risk_levels.append(risk_result["risk_level"])

    def risk_color(level):
        if level == "Critical":
            return "#ef4444"
        if level == "High":
            return "#f97316"
        if level == "Moderate":
            return "#facc15"
        return "#22c55e"

    def risk_background(level):
        if level == "Critical":
            return "rgba(127,29,29,0.18)"
        if level == "High":
            return "rgba(154,52,18,0.15)"
        if level == "Moderate":
            return "rgba(113,63,18,0.13)"
        return "rgba(20,83,45,0.12)"

    initial_color = risk_color(risk_levels[0])

    figure = make_subplots(
        rows=3,
        cols=4,
        specs=[
            [
                {"type": "indicator"},
                {"type": "indicator"},
                {"type": "indicator"},
                {"type": "indicator"},
            ],
            [
                {"type": "xy", "colspan": 2},
                None,
                {"type": "xy", "colspan": 2},
                None,
            ],
            [
                {"type": "xy", "colspan": 4},
                None,
                None,
                None,
            ],
        ],
        row_heights=[0.41, 0.45, 0.14],
        vertical_spacing=0.065,
        horizontal_spacing=0.07,
    )

    gauge_domains = [
        {"x": [0.015, 0.225], "y": [0.665, 0.925]},
        {"x": [0.270, 0.480], "y": [0.665, 0.925]},
        {"x": [0.525, 0.735], "y": [0.665, 0.925]},
        {"x": [0.780, 0.990], "y": [0.665, 0.925]},
    ]

    def add_gauge(
        value,
        title,
        suffix,
        axis_range,
        tickvals,
        ticktext,
        steps,
        threshold_value,
        domain,
        value_format,
    ):
        gauge = {
            "shape": "angular",
            "axis": {
                "range": axis_range,
                "tickmode": "array",
                "tickvals": tickvals,
                "ticktext": ticktext,
                "tickfont": {"size": 10, "color": "#cbd5e1"},
                "tickwidth": 1,
                "tickcolor": "#64748b",
                "ticklen": 4,
            },
            "bar": {
                "color": initial_color,
                "thickness": 0.15,
            },
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps,
        }

        if threshold_value is not None:
            gauge["threshold"] = {
                "line": {
                    "color": "#f8fafc",
                    "width": 3,
                },
                "thickness": 0.72,
                "value": threshold_value,
            }

        figure.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=float(value),
                domain=domain,
                title={
                    "text": f"<b>{title}</b>",
                    "font": {
                        "size": 14,
                        "color": "#f8fafc",
                    },
                },
                number={
                    "suffix": suffix,
                    "valueformat": value_format,
                    "font": {
                        "size": 22,
                        "color": "#f8fafc",
                    },
                },
                gauge=gauge,
            )
        )

    add_gauge(
        voltage_values[0],
        "Voltage",
        " V",
        [2.0, 4.3],
        [2.0, 3.0, 4.3],
        ["2.0", "3.0", "4.3"],
        [
            {"range": [2.0, 3.0], "color": "rgba(239,68,68,0.30)"},
            {"range": [3.0, 3.4], "color": "rgba(250,204,21,0.26)"},
            {"range": [3.4, 4.3], "color": "rgba(34,197,94,0.28)"},
        ],
        3.0,
        gauge_domains[0],
        ".3f",
    )

    add_gauge(
        temperature_values[0],
        "Temperature",
        " °C",
        [0, 70],
        [0, 35, 70],
        ["0", "35", "70"],
        [
            {"range": [0, 40], "color": "rgba(34,197,94,0.28)"},
            {"range": [40, 45], "color": "rgba(250,204,21,0.26)"},
            {"range": [45, 70], "color": "rgba(239,68,68,0.30)"},
        ],
        45,
        gauge_domains[1],
        ".1f",
    )

    add_gauge(
        soh_values[0],
        "State of Health",
        "%",
        [0, 100],
        [0, 50, 100],
        ["0", "50", "100"],
        [
            {"range": [0, 70], "color": "rgba(239,68,68,0.30)"},
            {"range": [70, 90], "color": "rgba(250,204,21,0.26)"},
            {"range": [90, 100], "color": "rgba(34,197,94,0.28)"},
        ],
        80,
        gauge_domains[2],
        ".1f",
    )

    add_gauge(
        risk_scores[0],
        "Risk Score",
        "/100",
        [0, 100],
        [0, 50, 100],
        ["0", "50", "100"],
        [
            {"range": [0, 20], "color": "rgba(34,197,94,0.28)"},
            {"range": [20, 45], "color": "rgba(250,204,21,0.26)"},
            {"range": [45, 70], "color": "rgba(249,115,22,0.28)"},
            {"range": [70, 100], "color": "rgba(239,68,68,0.30)"},
        ],
        None,
        gauge_domains[3],
        ".0f",
    )

    figure.add_trace(
        go.Scatter(
            x=time_values,
            y=voltage_values,
            mode="lines",
            name="Voltage",
            line={"width": 3},
            hovertemplate="%{x:.1f} s<br>%{y:.3f} V<extra></extra>",
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=[float(time_values[0])],
            y=[float(voltage_values[0])],
            mode="markers",
            name="Voltage playback",
            marker={"size": 15, "color": initial_color},
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=time_values,
            y=temperature_values,
            mode="lines",
            name="Temperature",
            line={"width": 3},
            hovertemplate="%{x:.1f} s<br>%{y:.2f} °C<extra></extra>",
        ),
        row=2,
        col=3,
    )

    figure.add_trace(
        go.Scatter(
            x=[float(time_values[0])],
            y=[float(temperature_values[0])],
            mode="markers",
            name="Temperature playback",
            marker={"size": 15, "color": initial_color},
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    figure.add_trace(
        go.Bar(
            x=[0.0],
            y=["Fault progression"],
            orientation="h",
            marker={"color": initial_color},
            text=["0%"],
            textposition="inside",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    figure.add_annotation(
        x=0.225,
        y=0.592,
        xref="paper",
        yref="paper",
        text="<b>Voltage Response</b>",
        showarrow=False,
        font={"size": 13, "color": "#f8fafc"},
    )

    figure.add_annotation(
        x=0.775,
        y=0.592,
        xref="paper",
        yref="paper",
        text="<b>Temperature Response</b>",
        showarrow=False,
        font={"size": 13, "color": "#f8fafc"},
    )

    frames = []

    for index in range(frame_count):
        frame_color = risk_color(risk_levels[index])
        progress_percent = (
            100.0 * index / max(1, frame_count - 1)
        )

        frames.append(
            go.Frame(
                name=str(index),
                data=[
                    go.Indicator(
                        value=float(voltage_values[index]),
                        gauge={"bar": {"color": frame_color}},
                    ),
                    go.Indicator(
                        value=float(temperature_values[index]),
                        gauge={"bar": {"color": frame_color}},
                    ),
                    go.Indicator(
                        value=float(soh_values[index]),
                        gauge={"bar": {"color": frame_color}},
                    ),
                    go.Indicator(
                        value=float(risk_scores[index]),
                        gauge={"bar": {"color": frame_color}},
                    ),
                    go.Scatter(
                        x=time_values,
                        y=voltage_values,
                    ),
                    go.Scatter(
                        x=[float(time_values[index])],
                        y=[float(voltage_values[index])],
                        marker={
                            "size": 15,
                            "color": frame_color,
                        },
                    ),
                    go.Scatter(
                        x=time_values,
                        y=temperature_values,
                    ),
                    go.Scatter(
                        x=[float(time_values[index])],
                        y=[float(temperature_values[index])],
                        marker={
                            "size": 15,
                            "color": frame_color,
                        },
                    ),
                    go.Bar(
                        x=[progress_percent],
                        y=["Fault progression"],
                        marker={"color": frame_color},
                        text=[f"{progress_percent:.0f}%"],
                    ),
                ],
                traces=list(range(9)),
                layout=go.Layout(
                    paper_bgcolor=risk_background(
                        risk_levels[index]
                    ),
                    title={
                        "text": (
                            f"{fault_type} — Step "
                            f"{index}/{frame_count - 1} | "
                            f"Risk: {risk_levels[index]} "
                            f"({risk_scores[index]}/100)"
                        ),
                        "x": 0.5,
                        "xanchor": "center",
                        "font": {
                            "size": 19,
                            "color": frame_color,
                        },
                    },
                ),
            )
        )

    figure.frames = frames

    slider_steps = [
        {
            "args": [
                [str(index)],
                {
                    "frame": {
                        "duration": 0,
                        "redraw": True,
                    },
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": str(index),
            "method": "animate",
        }
        for index in range(frame_count)
    ]

    figure.update_layout(
        title={
            "text": (
                f"{fault_type} — Step 0/"
                f"{frame_count - 1} | "
                f"Risk: {risk_levels[0]} "
                f"({risk_scores[0]}/100)"
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.992,
            "font": {
                "size": 19,
                "color": initial_color,
            },
        },
        height=900,
        margin={"l": 55, "r": 55, "t": 72, "b": 118},
        paper_bgcolor=risk_background(risk_levels[0]),
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "size": 13},
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.13,
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": 350,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 100
                                },
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "mode": "immediate",
                                "transition": {
                                    "duration": 0
                                },
                            },
                        ],
                    },
                    {
                        "label": "↺ Reset",
                        "method": "animate",
                        "args": [
                            ["0"],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": True,
                                },
                                "mode": "immediate",
                                "transition": {
                                    "duration": 0
                                },
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Playback step: ",
                    "font": {"color": "white"},
                },
                "pad": {"t": 48},
                "steps": slider_steps,
            }
        ],
    )

    figure.update_xaxes(
        title_text="Simulation Time (s)",
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text="Voltage (V)",
        row=2,
        col=1,
    )
    figure.update_xaxes(
        title_text="Simulation Time (s)",
        row=2,
        col=3,
    )
    figure.update_yaxes(
        title_text="Temperature (°C)",
        row=2,
        col=3,
    )
    figure.update_xaxes(
        range=[0, 100],
        showgrid=False,
        title_text="Fault development (%)",
        row=3,
        col=1,
    )
    figure.update_yaxes(
        showticklabels=False,
        row=3,
        col=1,
    )

    return figure


st.markdown(
    '<div class="fault-title">🛠️ Fault Injection Laboratory</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="fault-subtitle">
    Inject controlled faults into the AI digital twin,
    assess system risk and generate predictive-maintenance actions.
    </div>
    """,
    unsafe_allow_html=True,
)


battery_files = get_battery_files()

if not battery_files:
    st.error(
        "No processed battery files were found."
    )
    st.stop()


try:
    (
        voltage_model,
        temperature_model,
        feature_list,
    ) = load_models()

except Exception as error:
    st.error(
        "The voltage and temperature models could not be loaded."
    )
    st.exception(error)
    st.stop()


st.sidebar.title("Fault Controls")

selected_battery = st.sidebar.selectbox(
    "Battery",
    options=list(battery_files.keys()),
)

battery_df = load_battery_data(
    battery_files[selected_battery]
).copy()


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


required_columns = {
    "cycle": cycle_column,
    "time": time_column,
    "voltage": voltage_column,
    "temperature": temperature_column,
    "capacity": capacity_column,
}

missing_columns = [
    name
    for name, column in required_columns.items()
    if column is None
]

if missing_columns:
    st.error(
        "Required columns are missing: "
        + ", ".join(missing_columns)
    )
    st.stop()


numeric_columns = [
    cycle_column,
    time_column,
    voltage_column,
    temperature_column,
    capacity_column,
]

if current_column is not None:
    numeric_columns.append(current_column)

for column in numeric_columns:
    battery_df[column] = pd.to_numeric(
        battery_df[column],
        errors="coerce",
    )

battery_df = battery_df.dropna(
    subset=[
        cycle_column,
        time_column,
        voltage_column,
        temperature_column,
        capacity_column,
    ]
).copy()


available_cycles = sorted(
    battery_df[cycle_column]
    .dropna()
    .unique()
    .tolist()
)

selected_cycle = st.sidebar.selectbox(
    "Starting discharge cycle",
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


start_index = st.sidebar.slider(
    "Starting time step",
    min_value=0,
    max_value=len(cycle_df) - 2,
    value=min(20, len(cycle_df) - 2),
    step=1,
)

baseline_row = cycle_df.iloc[start_index].copy()

baseline_current = (
    float(baseline_row[current_column])
    if current_column is not None
    else -2.0
)

baseline_voltage = float(
    baseline_row[voltage_column]
)

baseline_temperature = float(
    baseline_row[temperature_column]
)

capacity_values = pd.to_numeric(
    cycle_df[capacity_column],
    errors="coerce",
).dropna()

capacity_values = capacity_values[
    capacity_values > 0
]

if capacity_values.empty:
    st.error(
        "The selected cycle has no valid capacity value."
    )
    st.stop()

current_capacity = float(
    capacity_values.median()
)

reference_capacity = calculate_reference_capacity(
    battery_df,
    cycle_column,
    capacity_column,
)

if (
    not np.isfinite(reference_capacity)
    or reference_capacity <= 0
):
    st.error(
        "A valid reference capacity could not be calculated."
    )
    st.stop()

baseline_soh = float(
    np.clip(
        current_capacity
        / reference_capacity
        * 100,
        0,
        100,
    )
)


time_differences = (
    cycle_df[time_column]
    .diff()
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

time_differences = time_differences[
    time_differences > 0
]

default_time_step = (
    float(time_differences.median())
    if not time_differences.empty
    else 10.0
)


st.sidebar.markdown("---")
st.sidebar.subheader("Fault Scenario")

fault_type = st.sidebar.selectbox(
    "Fault type",
    options=available_faults(),
)

fault_description = (
    FAULT_LIBRARY[fault_type].description
)

st.sidebar.caption(
    fault_description
)

severity_percent = st.sidebar.slider(
    "Fault severity (%)",
    min_value=0,
    max_value=100,
    value=int(
        FAULT_LIBRARY[
            fault_type
        ].default_severity * 100
    ),
    step=5,
)

simulation_steps = st.sidebar.slider(
    "Simulation steps",
    min_value=10,
    max_value=100,
    value=40,
    step=5,
)

time_step_s = st.sidebar.slider(
    "Time step (s)",
    min_value=1.0,
    max_value=60.0,
    value=float(
        np.clip(
            default_time_step,
            1.0,
            60.0,
        )
    ),
    step=1.0,
)

smoothing = st.sidebar.slider(
    "Prediction stability",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
)

run_fault = st.sidebar.button(
    "⚠️ Inject Fault",
    type="primary",
    use_container_width=True,
)


scenario_signature = (
    str(selected_battery),
    str(selected_cycle),
    int(start_index),
    str(fault_type),
    int(severity_percent),
    int(simulation_steps),
    float(time_step_s),
    float(smoothing),
)

stored_signature = st.session_state.get(
    "fault_scenario_signature"
)

stored_fault_df = st.session_state.get(
    "fault_simulation_df"
)


if run_fault:
    voltage_bounds, temperature_bounds = (
        data_informed_bounds(
            battery_df,
            voltage_column,
            temperature_column,
        )
    )

    try:
        baseline_simulation_df = (
            build_recursive_simulation(
                baseline_row=baseline_row,
                voltage_model=voltage_model,
                temperature_model=temperature_model,
                feature_list=feature_list,
                predict_next_state=predict_next_state,
                time_column=time_column,
                voltage_column=voltage_column,
                temperature_column=temperature_column,
                current_column=current_column,
                applied_current=baseline_current,
                simulation_steps=simulation_steps,
                time_step_s=time_step_s,
                smoothing=smoothing,
                voltage_bounds=voltage_bounds,
                temperature_bounds=temperature_bounds,
            )
        )

    except Exception as error:
        st.error(
            "The baseline recursive simulation failed."
        )
        st.exception(error)
        st.stop()

    severity = severity_percent / 100.0

    try:
        fault_df = inject_fault(
            simulation_df=baseline_simulation_df,
            fault_type=fault_type,
            severity=severity,
            baseline_current=baseline_current,
            baseline_soh=baseline_soh,
        )

    except Exception as error:
        st.error(
            "The selected fault could not be injected."
        )
        st.exception(error)
        st.stop()

    # Keep the completed simulation available during later
    # Streamlit reruns, including when Save to Incident History
    # or a download button is clicked.
    st.session_state[
        "fault_scenario_signature"
    ] = scenario_signature

    st.session_state[
        "fault_simulation_df"
    ] = fault_df.copy(deep=True)

elif (
    stored_fault_df is not None
    and stored_signature == scenario_signature
):
    # Reuse the last completed simulation. This prevents the page
    # from returning to the baseline preview when another button
    # triggers a Streamlit rerun.
    fault_df = stored_fault_df.copy(deep=True)

else:
    st.info(
        "Choose a fault, severity and simulation duration, "
        "then press **Inject Fault**."
    )

    preview_1, preview_2 = st.columns(
        [1.0, 1.35]
    )

    with preview_1:
        render_fault_battery(
            baseline_soh,
            baseline_voltage,
            baseline_temperature,
            "Low",
            "Normal preview",
        )

    with preview_2:
        st.markdown("### Baseline Operating Point")

        a, b = st.columns(2)

        with a:
            st.metric(
                "Voltage",
                f"{baseline_voltage:.3f} V",
            )

            st.metric(
                "SOH",
                f"{baseline_soh:.1f}%",
            )

        with b:
            st.metric(
                "Temperature",
                f"{baseline_temperature:.1f} °C",
            )

            st.metric(
                "Current",
                f"{baseline_current:.3f} A",
            )

        st.caption(
            "The fault engine will modify this baseline "
            "trajectory using a documented severity-dependent rule."
        )

    st.stop()


final_voltage = float(
    fault_df["true_voltage_V"].iloc[-1]
)

final_temperature = float(
    fault_df["true_temperature_C"].iloc[-1]
)

maximum_temperature = float(
    fault_df["true_temperature_C"].max()
)

final_soh = float(
    fault_df["fault_soh_percent"].iloc[-1]
)

risk_result = calculate_fault_risk(
    final_voltage=final_voltage,
    max_temperature=maximum_temperature,
    final_soh=final_soh,
)

risk_level = risk_result["risk_level"]
risk_score = risk_result["risk_score"]


estimated_rul = max(
    0.0,
    (final_soh - 70.0) * 4.0
)

maintenance = (
    generate_maintenance_recommendation(
        fault_type=fault_type,
        risk_level=risk_level,
        final_voltage=final_voltage,
        max_temperature=maximum_temperature,
        final_soh=final_soh,
        estimated_rul=estimated_rul,
    )
)


# =========================================================
# Real-time fault playback controls
# =========================================================
st.subheader("▶️ Real-Time Fault Playback")

playback_step = st.slider(
    "Fault timeline",
    min_value=0,
    max_value=len(fault_df) - 1,
    value=len(fault_df) - 1,
    step=1,
    help=(
        "Move the slider to replay the fault from the initial "
        "condition to the final simulated state."
    ),
)

playback_row = fault_df.iloc[playback_step]

playback_voltage = float(
    playback_row["true_voltage_V"]
)

playback_temperature = float(
    playback_row["true_temperature_C"]
)

playback_soh = float(
    playback_row["fault_soh_percent"]
)

playback_time = float(
    playback_row["time_s"]
)

playback_risk_result = calculate_fault_risk(
    final_voltage=playback_voltage,
    max_temperature=float(
        fault_df.iloc[: playback_step + 1][
            "true_temperature_C"
        ].max()
    ),
    final_soh=playback_soh,
)

playback_risk_level = (
    playback_risk_result["risk_level"]
)

playback_risk_score = (
    playback_risk_result["risk_score"]
)


left_panel, right_panel = st.columns(
    [1.0, 1.45]
)

with left_panel:
    st.markdown("### Fault-Responsive Digital Twin")

    render_fault_battery(
        playback_soh,
        playback_voltage,
        playback_temperature,
        playback_risk_level,
        fault_type,
    )


with right_panel:
    st.markdown("### Fault Outcome")

    metric_1, metric_2 = st.columns(2)

    with metric_1:
        st.metric(
            "Final Voltage",
            f"{playback_voltage:.3f} V",
            delta=(
                f"{playback_voltage - baseline_voltage:+.3f} V"
            ),
        )

        st.metric(
            "Final SOH",
            f"{playback_soh:.1f}%",
            delta=(
                f"{playback_soh - baseline_soh:+.1f}%"
            ),
        )

    with metric_2:
        st.metric(
            "Maximum Temperature",
            f"{playback_temperature:.1f} °C",
            delta=(
                f"{playback_temperature - baseline_temperature:+.1f} °C"
            ),
        )

        st.metric(
            "Risk Score",
            f"{playback_risk_score}/100",
        )

    st.metric(
        "Playback Time",
        f"{playback_time:.1f} s",
    )

    if playback_risk_level == "Critical":
        st.error(
            "Critical-risk fault condition detected at this step."
        )
    elif playback_risk_level == "High":
        st.warning(
            "High-risk fault condition detected at this step."
        )
    elif playback_risk_level == "Moderate":
        st.warning(
            "Moderate-risk fault condition detected at this step."
        )
    else:
        st.success(
            "Low-risk condition at this playback step."
        )


st.subheader("🎬 Automatic Fault Replay")

st.caption(
    "Use Play, Pause, Reset or the timeline slider. "
    "The gauges, risk indicator and telemetry marker "
    "advance together inside the animation."
)

automatic_replay_figure = (
    create_automatic_fault_replay(
        fault_df=fault_df,
        fault_type=fault_type,
    )
)

st.plotly_chart(
    automatic_replay_figure,
    use_container_width=True,
    key="automatic_fault_replay",
)


st.subheader("Fault Evolution")

voltage_figure = go.Figure()

voltage_figure.add_trace(
    go.Scatter(
        x=fault_df["time_s"],
        y=fault_df["voltage_V"],
        mode="lines",
        name="Baseline predicted voltage",
    )
)

voltage_figure.add_trace(
    go.Scatter(
        x=fault_df["time_s"],
        y=fault_df["true_voltage_V"],
        mode="lines+markers",
        name="Fault-affected voltage",
    )
)

if fault_type == "Sensor drift":
    voltage_figure.add_trace(
        go.Scatter(
            x=fault_df["time_s"],
            y=fault_df["displayed_voltage_V"],
            mode="lines",
            name="Displayed sensor voltage",
            line=dict(dash="dash"),
        )
    )

voltage_figure.add_hline(
    y=3.0,
    line_dash="dash",
    annotation_text="Critical voltage",
)

voltage_figure.add_vline(
    x=playback_time,
    line_dash="dot",
    annotation_text="Playback position",
)

voltage_figure.update_layout(
    title="Voltage Under Injected Fault",
    xaxis_title="Simulation Time (s)",
    yaxis_title="Voltage (V)",
    height=460,
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "white"},
)


temperature_figure = go.Figure()

temperature_figure.add_trace(
    go.Scatter(
        x=fault_df["time_s"],
        y=fault_df["temperature_C"],
        mode="lines",
        name="Baseline predicted temperature",
    )
)

temperature_figure.add_trace(
    go.Scatter(
        x=fault_df["time_s"],
        y=fault_df["true_temperature_C"],
        mode="lines+markers",
        name="Fault-affected temperature",
    )
)

if fault_type == "Sensor drift":
    temperature_figure.add_trace(
        go.Scatter(
            x=fault_df["time_s"],
            y=fault_df["displayed_temperature_C"],
            mode="lines",
            name="Displayed sensor temperature",
            line=dict(dash="dash"),
        )
    )

temperature_figure.add_hline(
    y=45,
    line_dash="dash",
    annotation_text="Thermal warning",
)

temperature_figure.add_vline(
    x=playback_time,
    line_dash="dot",
    annotation_text="Playback position",
)

temperature_figure.update_layout(
    title="Temperature Under Injected Fault",
    xaxis_title="Simulation Time (s)",
    yaxis_title="Temperature (°C)",
    height=460,
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "white"},
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


st.subheader("Fault Propagation Path")

progress_ratio = (
    playback_step
    / max(1, len(fault_df) - 1)
)

propagation_steps = [
    ("Fault introduced", progress_ratio >= 0.00),
    ("Primary variable affected", progress_ratio >= 0.20),
    ("Thermal/electrical stress develops", progress_ratio >= 0.40),
    ("Voltage and SOH response emerges", progress_ratio >= 0.60),
    ("Maintenance intervention required", progress_ratio >= 0.80),
]

for label, active in propagation_steps:
    if active:
        st.success(f"✓ {label}")
    else:
        st.info(f"○ {label}")


st.subheader("🧠 AI Root-Cause Analysis")

root_cause_paths = {
    "Cooling failure": [
        "Cooling effectiveness decreases",
        "Battery temperature accumulates",
        "Electrochemical and resistive stress increases",
        "Voltage performance deteriorates",
        "SOH and RUL are reduced",
    ],
    "Over-current": [
        "Current demand exceeds the baseline",
        "Ohmic voltage loss increases",
        "Heat generation increases",
        "Battery stress accelerates",
        "SOH and RUL are reduced",
    ],
    "Over-temperature": [
        "Thermal condition rises above baseline",
        "Ageing reactions accelerate",
        "Voltage stability decreases",
        "Capacity degradation increases",
        "SOH and RUL are reduced",
    ],
    "Deep discharge": [
        "Voltage approaches the lower operating limit",
        "Deep-discharge stress increases",
        "Voltage recovery becomes weaker",
        "Capacity fade accelerates",
        "SOH and RUL are reduced",
    ],
    "Fast charging": [
        "Charging current increases",
        "Thermal and electrochemical stress rises",
        "Voltage and temperature response become less stable",
        "Ageing rate increases",
        "SOH and RUL are reduced",
    ],
    "Internal resistance increase": [
        "Internal resistance rises",
        "Voltage sag increases under load",
        "Heat generation increases",
        "Usable capacity decreases",
        "SOH and RUL are reduced",
    ],
    "Sensor drift": [
        "Sensor bias develops",
        "Displayed telemetry separates from the physical state",
        "Fault diagnosis becomes unreliable",
        "Control decisions may be incorrect",
        "Sensor calibration is required",
    ],
    "Capacity fade": [
        "Available capacity declines",
        "Usable discharge duration decreases",
        "SOH falls progressively",
        "Maintenance urgency increases",
        "Battery replacement becomes necessary",
    ],
    "Normal": [
        "No injected fault",
        "Voltage remains within the baseline trajectory",
        "Temperature remains within the baseline trajectory",
        "SOH remains stable",
        "Routine monitoring continues",
    ],
}

root_path = root_cause_paths.get(
    fault_type,
    [
        "Fault introduced",
        "Primary battery variable changes",
        "Electrical or thermal stress develops",
        "Battery performance deteriorates",
        "Maintenance action is recommended",
    ],
)

for stage_index, stage_text in enumerate(root_path):
    reached = progress_ratio >= (
        stage_index / max(1, len(root_path) - 1)
    )

    if reached:
        st.success(
            f"✓ {stage_index + 1}. {stage_text}"
        )
    else:
        st.info(
            f"○ {stage_index + 1}. {stage_text}"
        )


st.subheader("Predictive Maintenance Engineer")

maintenance_1, maintenance_2 = st.columns(
    [1.2, 1.0]
)

with maintenance_1:
    st.markdown(
        f"""
        **Detected fault:** {maintenance["fault"]}

        **Risk level:** {maintenance["risk"]}

        **Reason:** {maintenance["reason"]}

        **Recommended action:**  
        {maintenance["recommended_action"]}
        """
    )

with maintenance_2:
    st.metric(
        "Urgency",
        maintenance["urgency"],
    )

    st.metric(
        "Maintenance Window",
        maintenance["maintenance_window"],
    )

    st.metric(
        "Indicative RUL",
        f"{estimated_rul:.0f} cycles",
    )


st.subheader("SOH and Risk Timeline")

timeline_figure = go.Figure()

timeline_figure.add_trace(
    go.Scatter(
        x=fault_df["step"],
        y=fault_df["fault_soh_percent"],
        mode="lines+markers",
        name="Fault-adjusted SOH",
    )
)

timeline_figure.add_hline(
    y=80,
    line_dash="dash",
    annotation_text="80% EOL threshold",
)

timeline_figure.add_vline(
    x=playback_step,
    line_dash="dot",
    annotation_text="Playback position",
)

timeline_figure.update_layout(
    xaxis_title="Simulation Step",
    yaxis_title="SOH (%)",
    yaxis=dict(range=[0, 105]),
    height=430,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "white"},
)

st.plotly_chart(
    timeline_figure,
    use_container_width=True,
)



# =========================================================
# AI incident report generator
# =========================================================
st.subheader("📄 AI Incident Report Generator")

generated_at = datetime.now()
incident_id = create_incident_id(
    selected_battery,
    generated_at,
)

probable_cause_map = {
    "Cooling failure": "Cooling fan degradation or restricted airflow",
    "Over-current": "Excessive electrical load or incorrect current control",
    "Over-temperature": "Elevated ambient temperature or thermal-management weakness",
    "Deep discharge": "Low-voltage cutoff configured below the safe operating limit",
    "Fast charging": "Charging rate exceeds the recommended operating condition",
    "Internal resistance increase": "Cell ageing, connection losses or electrode degradation",
    "Sensor drift": "Voltage or temperature sensor calibration error",
    "Capacity fade": "Progressive electrochemical ageing and loss of usable capacity",
    "Normal": "No injected fault",
}

decision_confidence = float(
    np.clip(
        70.0
        + 0.20 * risk_score
        + 0.10 * severity_percent,
        70.0,
        96.0,
    )
)

event_log = build_event_log(
    fault_df=fault_df,
    baseline_voltage=baseline_voltage,
    baseline_temperature=baseline_temperature,
    baseline_soh=baseline_soh,
)

decision_reason = (
    f"The injected {fault_type.lower()} scenario changed "
    f"voltage from {baseline_voltage:.3f} V to "
    f"{final_voltage:.3f} V, produced a maximum "
    f"temperature of {maximum_temperature:.1f} °C, "
    f"and changed SOH from {baseline_soh:.1f}% to "
    f"{final_soh:.1f}%."
)

digital_twin_summary = (
    f"The digital twin classified this scenario as "
    f"{risk_level.lower()} risk with a score of "
    f"{risk_score}/100. The recommended action is to "
    f"{maintenance['recommended_action'].lower()} "
    f"The indicative maintenance window is "
    f"{maintenance['maintenance_window'].lower()}."
)

incident_record = {
    "incident_id": incident_id,
    "generated_at": generated_at.strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
    "battery_id": selected_battery,
    "cycle": selected_cycle,
    "fault_type": fault_type,
    "severity_percent": severity_percent,
    "simulation_steps": simulation_steps,
    "time_step_s": time_step_s,
    "baseline_voltage": baseline_voltage,
    "baseline_temperature": baseline_temperature,
    "baseline_soh": baseline_soh,
    "baseline_current": baseline_current,
    "final_voltage": final_voltage,
    "maximum_temperature": maximum_temperature,
    "final_soh": final_soh,
    "risk_level": risk_level,
    "risk_score": risk_score,
    "estimated_rul": estimated_rul,
    "probable_cause": probable_cause_map.get(
        fault_type,
        "Battery operating fault",
    ),
    "decision_confidence": decision_confidence,
    "maintenance_urgency": maintenance["urgency"],
    "maintenance_window": maintenance["maintenance_window"],
    "recommended_action": maintenance["recommended_action"],
    "decision_reason": decision_reason,
    "summary": digital_twin_summary,
}

report_col_1, report_col_2, report_col_3 = st.columns(
    [1.0, 1.0, 1.0]
)

with report_col_1:
    st.metric(
        "Incident ID",
        incident_id,
    )

with report_col_2:
    st.metric(
        "AI Decision Confidence",
        f"{decision_confidence:.0f}%",
    )

with report_col_3:
    st.metric(
        "Maintenance Priority",
        maintenance["urgency"],
    )


try:
    incident_pdf = generate_incident_pdf(
        incident=incident_record,
        fault_df=fault_df,
        event_log=event_log,
    )

except Exception as report_error:
    incident_pdf = None
    st.error(
        "The PDF incident report could not be generated."
    )
    st.exception(report_error)


download_col_1, download_col_2, download_col_3 = st.columns(3)

with download_col_1:
    if incident_pdf is not None:
        st.download_button(
            "📄 Download Incident PDF",
            data=incident_pdf,
            file_name=f"{incident_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

with download_col_2:
    st.download_button(
        "📊 Download Fault CSV",
        data=fault_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=f"{incident_id}_simulation.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col_3:
    save_history = st.button(
        "💾 Save to Incident History",
        use_container_width=True,
    )


if save_history:
    try:
        saved_paths = save_incident_bundle(
            project_folder=PROJECT_FOLDER,
            incident_record=incident_record,
            fault_df=fault_df,
            incident_pdf=incident_pdf,
        )

        st.success(
            "Incident saved successfully to the project folder."
        )

        st.write("Main history folder:")

        st.code(
            str(saved_paths["history_folder"]),
            language=None,
        )

        st.write("Saved files:")

        st.code(
            "\n".join(
                [
                    f'History CSV: {saved_paths["history_csv"]}',
                    f'PDF: {saved_paths["pdf"]}',
                    f'Simulation CSV: {saved_paths["simulation_csv"]}',
                    f'Metadata JSON: {saved_paths["metadata_json"]}',
                ]
            ),
            language=None,
        )

    except Exception as save_error:
        st.error(
            "The incident was not saved."
        )

        st.exception(save_error)


with st.expander(
    "View generated event log"
):
    event_log_df = pd.DataFrame(
        event_log
    )

    st.dataframe(
        event_log_df,
        use_container_width=True,
        hide_index=True,
    )


with st.expander(
    "Fault-model assumptions"
):
    st.write(
        "The baseline voltage and temperature trajectories are "
        "generated recursively using the trained next-state models."
    )

    st.write(
        "Fault effects are transparent deterministic stress rules "
        "controlled by the selected severity. They are intended for "
        "research demonstration and decision-support prototyping, "
        "not certified safety analysis."
    )

    st.write(
        "Sensor drift intentionally separates the simulated physical "
        "state from the displayed sensor value."
    )


csv_data = fault_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download Fault Simulation Results",
    data=csv_data,
    file_name=(
        f"{selected_battery}_cycle_{selected_cycle}_"
        f"{fault_type.replace(' ', '_').lower()}_fault.csv"
    ),
    mime="text/csv",
)
