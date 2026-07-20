
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


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

from utils.model_loader import (
    load_models,
    load_rul_model,
)

from utils.prediction import (
    predict_next_state,
    predict_rul,
)


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="What-If Simulation",
    page_icon="🧪",
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

    .simulation-title {
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .simulation-subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 22px;
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


def safe_numeric(value, default=0.0):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return float(default)


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


def update_simulation_row(
    row,
    feature_list,
    time_column,
    voltage_column,
    temperature_column,
    current_column,
    next_time,
    next_voltage,
    next_temperature,
    imposed_current,
    initial_voltage,
    initial_temperature,
):
    updated = row.copy()

    if time_column is not None:
        updated[time_column] = next_time

    if voltage_column is not None:
        updated[voltage_column] = next_voltage

    if temperature_column is not None:
        updated[temperature_column] = next_temperature

    if current_column is not None:
        updated[current_column] = imposed_current

    # Common aliases that may appear in the saved feature list.
    alias_values = {
        "time_s": next_time,
        "Time_s": next_time,
        "time": next_time,
        "voltage_measured_V": next_voltage,
        "Voltage_measured": next_voltage,
        "voltage_V": next_voltage,
        "temperature_measured_C": next_temperature,
        "Temperature_measured": next_temperature,
        "temperature_C": next_temperature,
        "current_measured_A": imposed_current,
        "Current_measured": imposed_current,
        "current_A": imposed_current,
        "normalized_time": next_time,
        "temperature_rise_C": next_temperature - initial_temperature,
        "voltage_drop_from_start_V": initial_voltage - next_voltage,
        "voltage_change_V": next_voltage - safe_numeric(
            row.get(voltage_column, next_voltage),
            next_voltage,
        ),
        "temperature_change_C": next_temperature - safe_numeric(
            row.get(temperature_column, next_temperature),
            next_temperature,
        ),
        "current_change_A": imposed_current - safe_numeric(
            row.get(current_column, imposed_current),
            imposed_current,
        ),
    }

    for feature_name, feature_value in alias_values.items():
        if feature_name in feature_list:
            updated[feature_name] = feature_value

    return updated


def get_battery_visual_state(
    soh,
    voltage,
    temperature,
):
    if (
        soh <= 70
        or voltage < 3.0
        or temperature >= 50
    ):
        return {
            "label": "CRITICAL",
            "color": "#ef4444",
            "glow": "rgba(239,68,68,0.65)",
        }

    if (
        soh <= 80
        or voltage < 3.2
        or temperature >= 45
    ):
        return {
            "label": "DEGRADED",
            "color": "#f97316",
            "glow": "rgba(249,115,22,0.60)",
        }

    if (
        soh < 90
        or voltage < 3.4
        or temperature >= 40
    ):
        return {
            "label": "WARNING",
            "color": "#facc15",
            "glow": "rgba(250,204,21,0.55)",
        }

    return {
        "label": "HEALTHY",
        "color": "#22c55e",
        "glow": "rgba(34,197,94,0.55)",
    }


def render_animated_battery(
    soh,
    voltage,
    temperature,
    current,
):
    visual_state = get_battery_visual_state(
        soh,
        voltage,
        temperature,
    )

    fill_level = float(
        np.clip(
            soh,
            4,
            100,
        )
    )

    if current < -0.05:
        flow_text = "DISCHARGING"
        direction_symbol = "▼"
        particle_direction = "normal"
    elif current > 0.05:
        flow_text = "CHARGING"
        direction_symbol = "▲"
        particle_direction = "reverse"
    else:
        flow_text = "IDLE"
        direction_symbol = "●"
        particle_direction = "normal"

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

        .animation-card {{
            min-height: 470px;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 22px;
            background:
                radial-gradient(
                    circle at 50% 42%,
                    {visual_state["glow"]},
                    rgba(15,23,42,0.97) 52%
                );
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            perspective: 900px;
        }}

        .battery-stage {{
            position: relative;
            transform: rotateY(-10deg) rotateX(4deg);
            transform-style: preserve-3d;
        }}

        .battery-terminal {{
            width: 78px;
            height: 24px;
            background: linear-gradient(180deg, #f8fafc, #94a3b8);
            border-radius: 10px 10px 2px 2px;
            margin: 0 auto -2px auto;
            box-shadow: 0 0 14px rgba(255,255,255,0.25);
        }}

        .battery-shell {{
            position: relative;
            width: 190px;
            height: 310px;
            border: 8px solid #dbeafe;
            border-radius: 28px;
            box-sizing: border-box;
            padding: 11px;
            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.10),
                    rgba(2,6,23,0.95) 38%
                );
            box-shadow:
                18px 18px 0 rgba(2,6,23,0.55),
                0 0 26px {visual_state["glow"]},
                inset 0 0 22px rgba(255,255,255,0.06);
        }}

        .battery-shell::after {{
            content: "";
            position: absolute;
            inset: 16px 18px 18px 16px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            pointer-events: none;
        }}

        .battery-fill {{
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
                    rgba(255,255,255,0.33),
                    {visual_state["color"]}
                );
            box-shadow:
                0 0 32px {visual_state["glow"]};
            animation: batteryPulse 1.8s ease-in-out infinite;
            transition: height 0.8s ease;
            overflow: hidden;
        }}

        .battery-fill::before {{
            content: "";
            position: absolute;
            left: -30%;
            width: 160%;
            height: 24px;
            top: -6px;
            border-radius: 50%;
            background: rgba(255,255,255,0.30);
            animation: liquidWave 2.2s ease-in-out infinite;
        }}

        .percentage {{
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 38px;
            font-weight: 900;
            text-shadow: 0 2px 14px rgba(0,0,0,0.70);
            z-index: 5;
        }}

        .electron {{
            position: absolute;
            left: 50%;
            width: 10px;
            height: 10px;
            margin-left: -5px;
            border-radius: 50%;
            background: white;
            box-shadow: 0 0 12px white;
            animation:
                energyFlow 1.7s linear infinite
                {particle_direction};
            z-index: 4;
        }}

        .e1 {{ animation-delay: 0.00s; }}
        .e2 {{ animation-delay: 0.20s; }}
        .e3 {{ animation-delay: 0.40s; }}
        .e4 {{ animation-delay: 0.60s; }}
        .e5 {{ animation-delay: 0.80s; }}
        .e6 {{ animation-delay: 1.00s; }}
        .e7 {{ animation-delay: 1.20s; }}
        .e8 {{ animation-delay: 1.40s; }}

        @keyframes batteryPulse {{
            0%, 100% {{ filter: brightness(0.92); }}
            50% {{ filter: brightness(1.18); }}
        }}

        @keyframes liquidWave {{
            0%, 100% {{ transform: translateX(-4%) rotate(0deg); }}
            50% {{ transform: translateX(4%) rotate(2deg); }}
        }}

        @keyframes energyFlow {{
            0% {{
                bottom: 84%;
                opacity: 0;
                transform: scale(0.6);
            }}
            16% {{ opacity: 1; }}
            84% {{ opacity: 1; }}
            100% {{
                bottom: 8%;
                opacity: 0;
                transform: scale(1.15);
            }}
        }}

        .status-label {{
            margin-top: 24px;
            font-size: 26px;
            font-weight: 900;
            color: {visual_state["color"]};
            letter-spacing: 1.8px;
        }}

        .flow-label {{
            margin-top: 8px;
            font-size: 17px;
            color: #cbd5e1;
        }}

        .telemetry-line {{
            margin-top: 16px;
            display: grid;
            grid-template-columns: repeat(3, auto);
            gap: 18px;
            font-size: 15px;
            color: #e5e7eb;
        }}

        .telemetry-pill {{
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(15,23,42,0.76);
            border: 1px solid rgba(148,163,184,0.20);
        }}
    </style>
    </head>

    <body>
        <div class="animation-card">
            <div class="battery-stage">
                <div class="battery-terminal"></div>

                <div class="battery-shell">
                    <div class="battery-fill"></div>
                    <div class="percentage">{soh:.1f}%</div>

                    <div class="electron e1"></div>
                    <div class="electron e2"></div>
                    <div class="electron e3"></div>
                    <div class="electron e4"></div>
                    <div class="electron e5"></div>
                    <div class="electron e6"></div>
                    <div class="electron e7"></div>
                    <div class="electron e8"></div>
                </div>
            </div>

            <div class="status-label">
                {visual_state["label"]}
            </div>

            <div class="flow-label">
                {direction_symbol} {flow_text}
            </div>

            <div class="telemetry-line">
                <span class="telemetry-pill">V: {voltage:.2f}</span>
                <span class="telemetry-pill">T: {temperature:.1f} °C</span>
                <span class="telemetry-pill">I: {current:.2f} A</span>
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


# =========================================================
# Header
# =========================================================
st.markdown(
    '<div class="simulation-title">🧪 What-If Digital Twin Simulation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="simulation-subtitle">
    Explore how current, temperature and simulation duration
    influence voltage response, battery health and remaining life.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Load data and models
# =========================================================
battery_files = get_battery_files()

if not battery_files:
    st.error(
        "No processed battery files were found in "
        "`processed_datasets`."
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
st.sidebar.title("Simulation Controls")

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

baseline_voltage = safe_numeric(
    baseline_row[voltage_column]
)

baseline_temperature = safe_numeric(
    baseline_row[temperature_column]
)

if current_column is not None:
    baseline_current = safe_numeric(
        baseline_row[current_column]
    )
else:
    baseline_current = -2.0


time_differences = (
    cycle_df[time_column]
    .diff()
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

time_differences = time_differences[
    time_differences > 0
]

if time_differences.empty:
    default_time_step = 10.0
else:
    default_time_step = float(
        time_differences.median()
    )


st.sidebar.markdown("---")
st.sidebar.subheader("What-If Inputs")

simulated_current = st.sidebar.slider(
    "Applied current (A)",
    min_value=-5.0,
    max_value=2.0,
    value=float(
        np.clip(
            baseline_current,
            -5.0,
            2.0,
        )
    ),
    step=0.05,
)

ambient_temperature = st.sidebar.slider(
    "Starting temperature (°C)",
    min_value=0.0,
    max_value=60.0,
    value=float(
        np.clip(
            baseline_temperature,
            0.0,
            60.0,
        )
    ),
    step=0.5,
)

starting_voltage = st.sidebar.slider(
    "Starting voltage (V)",
    min_value=2.5,
    max_value=4.3,
    value=float(
        np.clip(
            baseline_voltage,
            2.5,
            4.3,
        )
    ),
    step=0.01,
)

simulation_steps = st.sidebar.slider(
    "Prediction steps",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
)

simulation_time_step = st.sidebar.slider(
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


prediction_smoothing = st.sidebar.slider(
    "Prediction stability",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
    help=(
        "Higher values follow the ML prediction more strongly. "
        "Lower values produce a smoother recursive trajectory."
    ),
)

current_capacity_values = pd.to_numeric(
    cycle_df[capacity_column],
    errors="coerce",
).dropna()

current_capacity_values = current_capacity_values[
    current_capacity_values > 0
]

if current_capacity_values.empty:
    st.error(
        "The selected cycle has no valid capacity value."
    )
    st.stop()

current_capacity = float(
    current_capacity_values.median()
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


# =========================================================
# Simulation button
# =========================================================
run_simulation = st.sidebar.button(
    "🚀 Run What-If Simulation",
    use_container_width=True,
    type="primary",
)

if not run_simulation:
    st.info(
        "Select the scenario controls in the sidebar, "
        "then press **Run What-If Simulation**."
    )

    preview_left, preview_right = st.columns(
        [1.05, 1.25]
    )

    with preview_left:
        render_animated_battery(
            baseline_soh,
            starting_voltage,
            ambient_temperature,
            simulated_current,
        )

    with preview_right:
        st.markdown("### Baseline Scenario")

        metric_1, metric_2 = st.columns(2)

        with metric_1:
            st.metric(
                "Starting Voltage",
                f"{starting_voltage:.3f} V",
            )

            st.metric(
                "Starting SOH",
                f"{baseline_soh:.1f}%",
            )

        with metric_2:
            st.metric(
                "Starting Temperature",
                f"{ambient_temperature:.1f} °C",
            )

            st.metric(
                "Applied Current",
                f"{simulated_current:.2f} A",
            )

        st.caption(
            "The animated battery will respond to the "
            "simulated operating condition after the model runs."
        )

    st.stop()


# =========================================================
# Data-informed simulation safety limits
# =========================================================
training_voltage_min = float(
    battery_df[voltage_column].quantile(0.01)
)

training_voltage_max = float(
    battery_df[voltage_column].quantile(0.99)
)

training_temperature_min = float(
    battery_df[temperature_column].quantile(0.01)
)

training_temperature_max = float(
    battery_df[temperature_column].quantile(0.99)
)

# Slightly widen the observed range, while preventing
# recursive drift into clearly implausible values.
safe_voltage_min = max(
    2.0,
    training_voltage_min - 0.15,
)

safe_voltage_max = min(
    4.5,
    training_voltage_max + 0.15,
)

safe_temperature_min = max(
    -5.0,
    training_temperature_min - 3.0,
)

safe_temperature_max = min(
    70.0,
    training_temperature_max + 5.0,
)


# =========================================================
# Recursive what-if simulation
# =========================================================
simulation_row = baseline_row.copy()

simulation_row[voltage_column] = starting_voltage
simulation_row[temperature_column] = ambient_temperature

if current_column is not None:
    simulation_row[current_column] = simulated_current

current_simulation_time = safe_numeric(
    simulation_row[time_column]
)

simulated_records = [
    {
        "step": 0,
        "time_s": current_simulation_time,
        "voltage_V": starting_voltage,
        "temperature_C": ambient_temperature,
    }
]

simulation_error = None

for step_number in range(
    1,
    simulation_steps + 1,
):
    try:
        (
            next_voltage,
            next_temperature,
        ) = predict_next_state(
            simulation_row,
            voltage_model,
            temperature_model,
            feature_list,
        )

    except Exception as error:
        simulation_error = error
        break

    next_voltage = safe_numeric(
        next_voltage,
        simulated_records[-1]["voltage_V"],
    )

    next_temperature = safe_numeric(
        next_temperature,
        simulated_records[-1]["temperature_C"],
    )

    previous_voltage = float(
        simulated_records[-1]["voltage_V"]
    )

    previous_temperature = float(
        simulated_records[-1]["temperature_C"]
    )

    # Stabilize recursive forecasts by blending the new
    # model output with the previous simulated state.
    next_voltage = (
        prediction_smoothing * next_voltage
        + (1.0 - prediction_smoothing) * previous_voltage
    )

    next_temperature = (
        prediction_smoothing * next_temperature
        + (1.0 - prediction_smoothing) * previous_temperature
    )

    # Limit each recursive step to a realistic rate of change.
    maximum_voltage_step = 0.12
    maximum_temperature_step = 2.5

    next_voltage = float(
        np.clip(
            next_voltage,
            previous_voltage - maximum_voltage_step,
            previous_voltage + maximum_voltage_step,
        )
    )

    next_temperature = float(
        np.clip(
            next_temperature,
            previous_temperature - maximum_temperature_step,
            previous_temperature + maximum_temperature_step,
        )
    )

    # Keep the trajectory close to the range represented
    # in the selected battery's measured data.
    next_voltage = float(
        np.clip(
            next_voltage,
            safe_voltage_min,
            safe_voltage_max,
        )
    )

    next_temperature = float(
        np.clip(
            next_temperature,
            safe_temperature_min,
            safe_temperature_max,
        )
    )

    next_simulation_time = (
        current_simulation_time
        + simulation_time_step
    )

    simulated_records.append(
        {
            "step": step_number,
            "time_s": next_simulation_time,
            "voltage_V": next_voltage,
            "temperature_C": next_temperature,
        }
    )

    simulation_row = update_simulation_row(
        row=simulation_row,
        feature_list=feature_list,
        time_column=time_column,
        voltage_column=voltage_column,
        temperature_column=temperature_column,
        current_column=current_column,
        next_time=next_simulation_time,
        next_voltage=next_voltage,
        next_temperature=next_temperature,
        imposed_current=simulated_current,
        initial_voltage=starting_voltage,
        initial_temperature=ambient_temperature,
    )

    current_simulation_time = next_simulation_time


if simulation_error is not None:
    st.error(
        "The recursive simulation stopped because the "
        "next-state model could not process the modified row."
    )
    st.exception(simulation_error)
    st.stop()


simulation_df = pd.DataFrame(
    simulated_records
)

final_voltage = float(
    simulation_df["voltage_V"].iloc[-1]
)

final_temperature = float(
    simulation_df["temperature_C"].iloc[-1]
)


# =========================================================
# Estimated scenario impact
# =========================================================
voltage_stress = max(
    0.0,
    3.2 - simulation_df["voltage_V"].min(),
)

temperature_stress = max(
    0.0,
    simulation_df["temperature_C"].max() - 40.0,
)

current_stress = max(
    0.0,
    abs(simulated_current) - abs(baseline_current),
)

scenario_health_penalty = (
    1.8 * voltage_stress
    + 0.18 * temperature_stress
    + 0.35 * current_stress
)

estimated_scenario_soh = float(
    np.clip(
        baseline_soh - scenario_health_penalty,
        0,
        100,
    )
)


# Create a scenario cycle dataframe for the RUL model.
scenario_cycle_df = cycle_df.copy()

scenario_cycle_df[voltage_column] = (
    final_voltage
)

scenario_cycle_df[temperature_column] = (
    final_temperature
)

if current_column is not None:
    scenario_cycle_df[current_column] = (
        simulated_current
    )

try:
    scenario_rul = predict_rul(
        cycle_df=scenario_cycle_df,
        cycle_column=cycle_column,
        capacity_column=capacity_column,
        temperature_column=temperature_column,
        voltage_column=voltage_column,
        current_column=current_column,
        time_column=time_column,
        rul_model=rul_model,
        rul_feature_columns=rul_feature_columns,
    )

    scenario_rul = safe_numeric(
        scenario_rul,
        np.nan,
    )

except Exception:
    scenario_rul = np.nan


if estimated_scenario_soh <= 80:
    scenario_rul_text = "EOL"

elif np.isnan(scenario_rul):
    scenario_rul_text = "Unknown"

else:
    scenario_rul_text = (
        f"{max(0.0, scenario_rul):.0f} cycles"
    )


# =========================================================
# Results summary
# =========================================================
summary_left, summary_right = st.columns(
    [1.05, 1.45]
)

with summary_left:
    st.markdown("### Animated Digital Twin")

    render_animated_battery(
        estimated_scenario_soh,
        final_voltage,
        final_temperature,
        simulated_current,
    )


with summary_right:
    st.markdown("### Scenario Outcome")

    metric_1, metric_2 = st.columns(2)

    with metric_1:
        st.metric(
            "Final Predicted Voltage",
            f"{final_voltage:.3f} V",
            delta=(
                f"{final_voltage - starting_voltage:+.3f} V"
            ),
        )

        st.metric(
            "Estimated Scenario SOH",
            f"{estimated_scenario_soh:.1f}%",
            delta=(
                f"{estimated_scenario_soh - baseline_soh:+.1f}%"
            ),
        )

    with metric_2:
        st.metric(
            "Final Predicted Temperature",
            f"{final_temperature:.1f} °C",
            delta=(
                f"{final_temperature - ambient_temperature:+.1f} °C"
            ),
        )

        st.metric(
            "Scenario RUL",
            scenario_rul_text,
        )

    visual_state = get_battery_visual_state(
        estimated_scenario_soh,
        final_voltage,
        final_temperature,
    )

    if visual_state["label"] == "CRITICAL":
        st.error(
            "The simulated scenario creates a critical "
            "battery operating condition."
        )

    elif visual_state["label"] == "DEGRADED":
        st.warning(
            "The scenario indicates degraded operation. "
            "Reduce electrical or thermal stress."
        )

    elif visual_state["label"] == "WARNING":
        st.warning(
            "The scenario remains operable but requires "
            "closer monitoring."
        )

    else:
        st.success(
            "The simulated battery remains within a "
            "nominal operating region."
        )


    explanation_parts = []

    if abs(simulated_current) > abs(baseline_current) + 0.10:
        explanation_parts.append(
            "the applied current is more demanding than the baseline"
        )

    if final_temperature > baseline_temperature + 2.0:
        explanation_parts.append(
            "the predicted temperature increased"
        )

    if final_voltage < baseline_voltage - 0.10:
        explanation_parts.append(
            "the predicted voltage declined"
        )

    if estimated_scenario_soh < baseline_soh - 0.2:
        explanation_parts.append(
            "the scenario produced an estimated SOH penalty"
        )

    if explanation_parts:
        explanation_text = (
            "AI interpretation: "
            + "; ".join(explanation_parts)
            + "."
        )
    else:
        explanation_text = (
            "AI interpretation: the selected scenario remains "
            "close to the measured baseline operating condition."
        )

    st.info(explanation_text)


# =========================================================
# Simulation charts
# =========================================================
st.subheader("Predicted What-If Response")

voltage_figure = go.Figure()

voltage_figure.add_trace(
    go.Scatter(
        x=simulation_df["time_s"],
        y=simulation_df["voltage_V"],
        mode="lines+markers",
        name="Predicted voltage",
    )
)

voltage_figure.add_hline(
    y=3.0,
    line_dash="dash",
    annotation_text="Critical voltage threshold",
)

voltage_figure.update_layout(
    title="Simulated Voltage Response",
    xaxis_title="Simulation Time (s)",
    yaxis_title="Voltage (V)",
    height=460,
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
        x=simulation_df["time_s"],
        y=simulation_df["temperature_C"],
        mode="lines+markers",
        name="Predicted temperature",
    )
)

temperature_figure.add_hline(
    y=45,
    line_dash="dash",
    annotation_text="Thermal warning threshold",
)

temperature_figure.update_layout(
    title="Simulated Temperature Response",
    xaxis_title="Simulation Time (s)",
    yaxis_title="Temperature (°C)",
    height=460,
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
# Scenario comparison
# =========================================================
st.subheader("Baseline vs. What-If Scenario")

comparison_df = pd.DataFrame(
    {
        "Metric": [
            "Voltage (V)",
            "Temperature (°C)",
            "SOH (%)",
            "Applied Current (A)",
        ],
        "Baseline": [
            baseline_voltage,
            baseline_temperature,
            baseline_soh,
            baseline_current,
        ],
        "What-If Scenario": [
            final_voltage,
            final_temperature,
            estimated_scenario_soh,
            simulated_current,
        ],
    }
)

comparison_figure = go.Figure()

comparison_figure.add_trace(
    go.Bar(
        x=comparison_df["Metric"],
        y=comparison_df["Baseline"],
        name="Baseline",
    )
)

comparison_figure.add_trace(
    go.Bar(
        x=comparison_df["Metric"],
        y=comparison_df["What-If Scenario"],
        name="What-If Scenario",
    )
)

comparison_figure.update_layout(
    barmode="group",
    height=430,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={
        "color": "white",
        "size": 14,
    },
)

st.plotly_chart(
    comparison_figure,
    use_container_width=True,
)


# =========================================================
# Transparency note and data download
# =========================================================
with st.expander(
    "Important simulation assumptions"
):
    st.write(
        "Voltage and temperature are generated recursively "
        "using the trained next-state machine-learning models. "
        "A smoothing factor, per-step change limit and "
        "data-informed operating bounds are applied to reduce "
        "recursive drift outside the measured domain."
    )

    st.write(
        "The scenario SOH adjustment is a transparent "
        "stress indicator based on low voltage, high "
        "temperature and increased current. It is not yet "
        "a physics-certified degradation model."
    )

    st.write(
        "The RUL value is produced by the existing "
        "cycle-level RUL model and should be treated as "
        "a preliminary research estimate."
    )


csv_data = simulation_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download Simulation Results",
    data=csv_data,
    file_name=(
        f"{selected_battery}_cycle_{selected_cycle}"
        "_what_if_simulation.csv"
    ),
    mime="text/csv",
)
