from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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
    page_title="Battery Comparison",
    page_icon="🔋",
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
                rgba(14, 165, 233, 0.10),
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


def estimate_rul(
    health_df,
    cycle_column,
    capacity_column,
    initial_capacity,
):
    clean_df = (
        health_df[
            [cycle_column, capacity_column]
        ]
        .dropna()
        .copy()
    )

    if len(clean_df) < 5:
        return np.nan

    # Use the latest degradation region rather than the full history.
    recent_points = min(
        40,
        len(clean_df),
    )

    recent_df = clean_df.tail(
        recent_points
    )

    slope, intercept = np.polyfit(
        recent_df[cycle_column],
        recent_df[capacity_column],
        1,
    )

    if slope >= 0:
        return np.nan

    eol_capacity = initial_capacity * 0.80

    predicted_eol_cycle = (
        eol_capacity - intercept
    ) / slope

    latest_cycle = float(
        clean_df[cycle_column].iloc[-1]
    )

    estimated_rul = (
        predicted_eol_cycle
        - latest_cycle
    )

    return max(
        0.0,
        estimated_rul,
    )


def health_label(soh):
    if soh >= 90:
        return "Excellent"

    if soh >= 80:
        return "Healthy"

    if soh >= 70:
        return "Degraded"

    return "End of Life"


# =========================================================
# Header
# =========================================================
st.title("🔋 Battery Fleet Comparison")

st.caption(
    "Compare battery health, degradation and remaining useful life "
    "across the NASA battery fleet"
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


battery_options = list(
    battery_files.keys()
)

default_selection = battery_options[
    :min(4, len(battery_options))
]


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Fleet Controls")

selected_batteries = st.sidebar.multiselect(
    "Select batteries",
    options=battery_options,
    default=default_selection,
)

if not selected_batteries:
    st.warning(
        "Select at least one battery."
    )
    st.stop()


# =========================================================
# Build fleet summary
# =========================================================
fleet_records = []
degradation_curves = []


for battery_id in selected_batteries:

    battery_df = load_battery_data(
        battery_files[battery_id]
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

    capacity_column = find_column(
        battery_df,
        [
            "capacity_Ah",
            "capacity",
            "Capacity",
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

    voltage_column = find_column(
        battery_df,
        [
            "voltage_measured_V",
            "Voltage_measured",
            "voltage_V",
        ],
    )

    if (
        cycle_column is None
        or capacity_column is None
    ):
        continue

    aggregation = {
        capacity_column: "first"
    }

    if temperature_column is not None:
        aggregation[
            temperature_column
        ] = "mean"

    if voltage_column is not None:
        aggregation[
            voltage_column
        ] = "mean"

    health_df = (
        battery_df
        .groupby(
            cycle_column,
            as_index=False,
        )
        .agg(aggregation)
        .sort_values(cycle_column)
        .dropna(
            subset=[
                capacity_column
            ]
        )
        .reset_index(drop=True)
    )

    if health_df.empty:
        continue

    # -----------------------------------------------------
    # Remove impossible or clearly corrupted capacities
    # -----------------------------------------------------
    capacity_series = pd.to_numeric(
        health_df[capacity_column],
        errors="coerce",
    )

    positive_capacity = capacity_series[
        capacity_series > 0
    ]

    if positive_capacity.empty:
        continue

    lower_limit = positive_capacity.quantile(
        0.01
    )

    upper_limit = positive_capacity.quantile(
        0.99
    )

    health_df = health_df[
        (
            health_df[capacity_column]
            >= lower_limit
        )
        & (
            health_df[capacity_column]
            <= upper_limit
        )
    ].copy()

    health_df = health_df.reset_index(
        drop=True
    )

    if health_df.empty:
        continue

    # -----------------------------------------------------
    # Smooth capacity to reduce cycle-to-cycle noise
    # -----------------------------------------------------
    health_df[
        "smoothed_capacity_Ah"
    ] = (
        health_df[capacity_column]
        .rolling(
            window=5,
            min_periods=1,
            center=True,
        )
        .median()
    )

    # -----------------------------------------------------
    # Reference capacity
    # Use a high stable percentile instead of first cycle
    # -----------------------------------------------------
    initial_capacity = float(
        health_df[
            "smoothed_capacity_Ah"
        ].quantile(0.95)
    )

    # Current capacity based on latest five cycles
    latest_capacity = float(
        health_df[
            "smoothed_capacity_Ah"
        ]
        .tail(5)
        .median()
    )

    latest_cycle = int(
        health_df[
            cycle_column
        ].iloc[-1]
    )

    soh = (
        latest_capacity
        / initial_capacity
        * 100
    )

    soh = float(
        np.clip(
            soh,
            0,
            100,
        )
    )

    capacity_loss = max(
        0.0,
        initial_capacity
        - latest_capacity,
    )

    capacity_loss_percent = max(
        0.0,
        100.0 - soh,
    )

    estimated_rul = estimate_rul(
        health_df,
        cycle_column,
        "smoothed_capacity_Ah",
        initial_capacity,
    )

    average_temperature = (
        float(
            health_df[
                temperature_column
            ].mean()
        )
        if temperature_column is not None
        else np.nan
    )

    average_voltage = (
        float(
            health_df[
                voltage_column
            ].mean()
        )
        if voltage_column is not None
        else np.nan
    )

    fleet_records.append(
        {
            "Battery": battery_id,
            "Initial Capacity (Ah)": initial_capacity,
            "Current Capacity (Ah)": latest_capacity,
            "SOH (%)": soh,
            "Capacity Loss (Ah)": capacity_loss,
            "Capacity Loss (%)": capacity_loss_percent,
            "Latest Cycle": latest_cycle,
            "Estimated RUL (cycles)": estimated_rul,
            "Average Temperature (°C)": average_temperature,
            "Average Voltage (V)": average_voltage,
            "Health Status": health_label(soh),
        }
    )

    curve_df = pd.DataFrame(
        {
            "Battery": battery_id,
            "Cycle": health_df[
                cycle_column
            ],
            "Capacity_Ah": health_df[
                "smoothed_capacity_Ah"
            ],
            "SOH_percent": (
                health_df[
                    "smoothed_capacity_Ah"
                ]
                / initial_capacity
                * 100
            ).clip(
                lower=0,
                upper=100,
            ),
        }
    )

    degradation_curves.append(
        curve_df
    )


fleet_df = pd.DataFrame(
    fleet_records
)

if fleet_df.empty:
    st.error(
        "No valid fleet summary could be generated."
    )
    st.stop()


fleet_df = fleet_df.sort_values(
    "SOH (%)",
    ascending=False,
).reset_index(drop=True)

fleet_df[
    "Fleet Rank"
] = np.arange(
    1,
    len(fleet_df) + 1,
)


# =========================================================
# Fleet summary
# =========================================================
best_battery = fleet_df.iloc[0]
worst_battery = fleet_df.iloc[-1]

average_soh = float(
    fleet_df[
        "SOH (%)"
    ].mean()
)

average_capacity = float(
    fleet_df[
        "Current Capacity (Ah)"
    ].mean()
)


st.subheader("Fleet Summary")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Batteries Compared",
        f"{len(fleet_df)}",
    )

with metric_2:
    st.metric(
        "Average Fleet SOH",
        f"{average_soh:.2f}%",
    )

with metric_3:
    st.metric(
        "Best Battery",
        str(
            best_battery[
                "Battery"
            ]
        ),
        delta=(
            f"{best_battery['SOH (%)']:.2f}% SOH"
        ),
    )

with metric_4:
    st.metric(
        "Lowest-SOH Battery",
        str(
            worst_battery[
                "Battery"
            ]
        ),
        delta=(
            f"{worst_battery['SOH (%)']:.2f}% SOH"
        ),
        delta_color="inverse",
    )


# =========================================================
# Battery ranking
# =========================================================
st.subheader("Battery Health Ranking")

ranking_columns = [
    "Fleet Rank",
    "Battery",
    "SOH (%)",
    "Initial Capacity (Ah)",
    "Current Capacity (Ah)",
    "Capacity Loss (%)",
    "Latest Cycle",
    "Estimated RUL (cycles)",
    "Health Status",
]

ranking_display = fleet_df[
    ranking_columns
].copy()

ranking_display[
    "SOH (%)"
] = ranking_display[
    "SOH (%)"
].round(2)

ranking_display[
    "Initial Capacity (Ah)"
] = ranking_display[
    "Initial Capacity (Ah)"
].round(4)

ranking_display[
    "Current Capacity (Ah)"
] = ranking_display[
    "Current Capacity (Ah)"
].round(4)

ranking_display[
    "Capacity Loss (%)"
] = ranking_display[
    "Capacity Loss (%)"
].round(2)

ranking_display[
    "Estimated RUL (cycles)"
] = ranking_display[
    "Estimated RUL (cycles)"
].round(0)

st.dataframe(
    ranking_display,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# SOH comparison
# =========================================================
soh_figure = go.Figure()

soh_figure.add_trace(
    go.Bar(
        x=fleet_df[
            "Battery"
        ],
        y=fleet_df[
            "SOH (%)"
        ],
        text=(
            fleet_df[
                "SOH (%)"
            ]
            .round(1)
            .astype(str)
            + "%"
        ),
        textposition="outside",
        name="SOH",
    )
)

soh_figure.add_hline(
    y=80,
    line_dash="dash",
    annotation_text=(
        "80% EOL threshold"
    ),
)

soh_figure.update_layout(
    title=(
        "Fleet State-of-Health Comparison"
    ),
    xaxis_title="Battery",
    yaxis_title="SOH (%)",
    yaxis_range=[
        0,
        110,
    ],
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={
        "color": "white"
    },
)

st.plotly_chart(
    soh_figure,
    use_container_width=True,
)


# =========================================================
# Initial versus current capacity
# =========================================================
capacity_figure = go.Figure()

capacity_figure.add_trace(
    go.Bar(
        x=fleet_df[
            "Battery"
        ],
        y=fleet_df[
            "Initial Capacity (Ah)"
        ],
        name="Reference capacity",
    )
)

capacity_figure.add_trace(
    go.Bar(
        x=fleet_df[
            "Battery"
        ],
        y=fleet_df[
            "Current Capacity (Ah)"
        ],
        name="Current capacity",
    )
)

capacity_figure.update_layout(
    title=(
        "Reference vs Current Capacity"
    ),
    xaxis_title="Battery",
    yaxis_title="Capacity (Ah)",
    barmode="group",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={
        "color": "white"
    },
)

st.plotly_chart(
    capacity_figure,
    use_container_width=True,
)


# =========================================================
# SOH degradation curves
# =========================================================
if degradation_curves:
    all_curves_df = pd.concat(
        degradation_curves,
        ignore_index=True,
    )

    degradation_figure = go.Figure()

    for battery_id in selected_batteries:

        battery_curve = all_curves_df[
            all_curves_df[
                "Battery"
            ] == battery_id
        ]

        if battery_curve.empty:
            continue

        degradation_figure.add_trace(
            go.Scatter(
                x=battery_curve[
                    "Cycle"
                ],
                y=battery_curve[
                    "SOH_percent"
                ],
                mode="lines",
                name=battery_id,
            )
        )

    degradation_figure.add_hline(
        y=80,
        line_dash="dash",
        annotation_text=(
            "80% EOL threshold"
        ),
    )

    degradation_figure.update_layout(
        title=(
            "SOH Degradation Across Battery Life"
        ),
        xaxis_title="Discharge Cycle",
        yaxis_title="SOH (%)",
        yaxis_range=[
            0,
            110,
        ],
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "white"
        },
    )

    st.plotly_chart(
        degradation_figure,
        use_container_width=True,
    )


# =========================================================
# RUL comparison
# =========================================================
rul_plot_df = fleet_df.dropna(
    subset=[
        "Estimated RUL (cycles)"
    ]
).copy()

if not rul_plot_df.empty:

    rul_figure = go.Figure()

    rul_figure.add_trace(
        go.Bar(
            x=rul_plot_df[
                "Battery"
            ],
            y=rul_plot_df[
                "Estimated RUL (cycles)"
            ],
            name="Estimated RUL",
        )
    )

    rul_figure.update_layout(
        title=(
            "Estimated Remaining Useful Life"
        ),
        xaxis_title="Battery",
        yaxis_title=(
            "Estimated RUL (cycles)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "white"
        },
    )

    st.plotly_chart(
        rul_figure,
        use_container_width=True,
    )


# =========================================================
# Fleet recommendation
# =========================================================
st.subheader(
    "Fleet Maintenance Recommendation"
)

eol_count = int(
    (
        fleet_df[
            "SOH (%)"
        ] <= 80
    ).sum()
)

degraded_count = int(
    (
        (
            fleet_df[
                "SOH (%)"
            ] > 80
        )
        & (
            fleet_df[
                "SOH (%)"
            ] < 90
        )
    ).sum()
)


if eol_count > 0:
    st.error(
        f"{eol_count} selected battery or batteries have "
        "reached the conventional 80% end-of-life threshold. "
        "Replacement planning is recommended."
    )

elif degraded_count > 0:
    st.warning(
        f"{degraded_count} selected battery or batteries show "
        "measurable degradation and require closer monitoring."
    )

else:
    st.success(
        "All selected batteries remain above 90% SOH."
    )


# =========================================================
# Download fleet summary
# =========================================================
fleet_csv = fleet_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download fleet comparison",
    data=fleet_csv,
    file_name=(
        "NASA_battery_fleet_comparison.csv"
    ),
    mime="text/csv",
)