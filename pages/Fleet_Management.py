from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_FOLDER = Path(__file__).resolve().parents[1]

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


from utils.fleet_management import (
    build_fleet_snapshot,
    list_battery_files,
)

from utils.incident_report import (
    load_incident_history,
)


st.set_page_config(
    page_title="Fleet Management",
    page_icon="🛸",
    layout="wide",
)


PROCESSED_FOLDER = (
    PROJECT_FOLDER
    / "processed_datasets"
)

INCIDENT_FOLDER = (
    PROJECT_FOLDER
    / "incident_history"
)


st.markdown(
    """
    <style>
    .fleet-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .fleet-subtitle {
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }

    .fleet-card {
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 16px;
        padding: 1rem;
        background: rgba(15,23,42,0.55);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="fleet-title">🛸 Battery Fleet Management</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="fleet-subtitle">'
    'Monitor the health, risk, incidents and maintenance status '
    'of the complete NASA battery fleet.'
    '</div>',
    unsafe_allow_html=True,
)


battery_files = list_battery_files(
    PROCESSED_FOLDER
)

if not battery_files:
    st.error(
        "No processed battery files were found."
    )
    st.stop()


with st.spinner(
    "Building fleet health snapshot..."
):
    fleet_df, failure_df = build_fleet_snapshot(
        battery_files
    )


if fleet_df.empty:
    st.error(
        "No valid battery snapshots could be created."
    )

    if not failure_df.empty:
        st.dataframe(
            failure_df,
            width="stretch",
            hide_index=True,
        )

    st.stop()


incident_df = load_incident_history(
    INCIDENT_FOLDER
)

if incident_df.empty:
    incident_summary = pd.DataFrame(
        columns=[
            "battery_id",
            "incident_count",
            "max_risk_score",
            "latest_incident",
        ]
    )
else:
    incident_df = incident_df.copy()

    incident_df["risk_score"] = pd.to_numeric(
        incident_df["risk_score"],
        errors="coerce",
    ).fillna(0)

    incident_summary = (
        incident_df.groupby(
            "battery_id",
            as_index=False,
        )
        .agg(
            incident_count=(
                "incident_id",
                "count",
            ),
            max_risk_score=(
                "risk_score",
                "max",
            ),
            latest_incident=(
                "generated_at",
                "max",
            ),
        )
    )


fleet_df = fleet_df.merge(
    incident_summary,
    on="battery_id",
    how="left",
)

fleet_df["incident_count"] = (
    pd.to_numeric(
        fleet_df["incident_count"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

fleet_df["max_risk_score"] = (
    pd.to_numeric(
        fleet_df["max_risk_score"],
        errors="coerce",
    )
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

fleet_df["latest_incident"] = (
    fleet_df["latest_incident"]
    .fillna("None")
)


def operational_priority(row):
    if row["status"] == "Critical":
        return "Immediate action"

    if row["status"] == "High Risk":
        return "Act soon"

    if row["incident_count"] > 0:
        return "Review incident"

    if row["status"] == "Warning":
        return "Schedule inspection"

    return "Routine"


fleet_df["priority"] = fleet_df.apply(
    operational_priority,
    axis=1,
)


filter_1, filter_2, filter_3 = st.columns(3)

with filter_1:
    status_options = [
        "All"
    ] + sorted(
        fleet_df["status"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_status = st.selectbox(
        "Fleet status",
        status_options,
    )

with filter_2:
    minimum_soh = st.slider(
        "Minimum SOH (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )

with filter_3:
    incident_filter = st.selectbox(
        "Incident filter",
        [
            "All",
            "With incidents",
            "Without incidents",
        ],
    )


filtered_df = fleet_df.copy()

if selected_status != "All":
    filtered_df = filtered_df[
        filtered_df["status"]
        == selected_status
    ]

filtered_df = filtered_df[
    filtered_df["soh_percent"]
    >= minimum_soh
]

if incident_filter == "With incidents":
    filtered_df = filtered_df[
        filtered_df["incident_count"] > 0
    ]

elif incident_filter == "Without incidents":
    filtered_df = filtered_df[
        filtered_df["incident_count"] == 0
    ]


if filtered_df.empty:
    st.warning(
        "No batteries match the selected filters."
    )
    st.stop()


filtered_df = filtered_df.copy()

filtered_df["bubble_size"] = (
    pd.to_numeric(
        filtered_df["max_risk_score"],
        errors="coerce",
    )
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .clip(lower=5.0, upper=100.0)
)


healthy_count = int(
    (
        filtered_df["status"]
        == "Healthy"
    ).sum()
)

warning_count = int(
    filtered_df["status"]
    .isin(["Warning", "High Risk"])
    .sum()
)

critical_count = int(
    (
        filtered_df["status"]
        == "Critical"
    ).sum()
)

average_soh = float(
    filtered_df["soh_percent"]
    .mean()
)


kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)

with kpi_1:
    st.metric(
        "Fleet Size",
        len(filtered_df),
    )

with kpi_2:
    st.metric(
        "Healthy",
        healthy_count,
    )

with kpi_3:
    st.metric(
        "Warning",
        warning_count,
    )

with kpi_4:
    st.metric(
        "Critical",
        critical_count,
    )

with kpi_5:
    st.metric(
        "Average SOH",
        f"{average_soh:.1f}%",
    )


st.subheader("Fleet Health Map")

status_symbol = {
    "Healthy": "🟢",
    "Warning": "🟡",
    "High Risk": "🟠",
    "Critical": "🔴",
}

fleet_grid_columns = st.columns(5)

for index, row in filtered_df.reset_index(
    drop=True
).iterrows():
    column = fleet_grid_columns[
        index % 5
    ]

    with column:
        symbol = status_symbol.get(
            row["status"],
            "⚪",
        )

        st.markdown(
            f"""
            <div class="fleet-card">
                <div style="font-size:1.35rem;font-weight:700;">
                    {symbol} {row['battery_id']}
                </div>
                <div>SOH: {row['soh_percent']:.1f}%</div>
                <div>Cycle: {row['latest_cycle']:.0f}</div>
                <div>Incidents: {row['incident_count']}</div>
                <div>Priority: {row['priority']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.subheader("Fleet Overview")

overview_columns = [
    "battery_id",
    "status",
    "soh_percent",
    "latest_cycle",
    "latest_capacity_Ah",
    "median_voltage_V",
    "median_temperature_C",
    "incident_count",
    "max_risk_score",
    "priority",
]

st.dataframe(
    filtered_df[
        overview_columns
    ].sort_values(
        [
            "status",
            "soh_percent",
        ],
        ascending=[
            True,
            True,
        ],
    ),
    width="stretch",
    hide_index=True,
)


chart_1, chart_2 = st.columns(2)

with chart_1:
    soh_figure = px.bar(
        filtered_df.sort_values(
            "soh_percent"
        ),
        x="battery_id",
        y="soh_percent",
        color="status",
        title="Fleet SOH Distribution",
        labels={
            "battery_id": "Battery",
            "soh_percent": "SOH (%)",
        },
    )

    soh_figure.add_hline(
        y=80,
        line_dash="dash",
        annotation_text="80% EOL threshold",
    )

    soh_figure.update_layout(
        height=430,
    )

    st.plotly_chart(
        soh_figure,
        width="stretch",
    )

with chart_2:
    incident_figure = px.scatter(
        filtered_df,
        x="soh_percent",
        y="incident_count",
        size="bubble_size",
        size_max=36,
        color="status",
        hover_name="battery_id",
        hover_data={
            "bubble_size": False,
            "max_risk_score": True,
        },
        title="Incidents vs Battery Health",
        labels={
            "soh_percent": "SOH (%)",
            "incident_count": "Incident Count",
            "max_risk_score": "Maximum Risk Score",
        },
    )

    incident_figure.update_layout(
        height=430,
    )

    st.plotly_chart(
        incident_figure,
        width="stretch",
    )


st.subheader("Fleet Risk Distribution")

risk_counts = (
    filtered_df["status"]
    .value_counts()
    .reindex(
        [
            "Healthy",
            "Warning",
            "High Risk",
            "Critical",
        ],
        fill_value=0,
    )
)

risk_figure = go.Figure(
    data=[
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.55,
        )
    ]
)

risk_figure.update_layout(
    height=400,
)

st.plotly_chart(
    risk_figure,
    width="stretch",
)


st.subheader("Battery Health Passport")

selected_battery = st.selectbox(
    "Select battery",
    filtered_df["battery_id"]
    .astype(str)
    .tolist(),
)

selected_row = filtered_df[
    filtered_df["battery_id"].astype(str)
    == selected_battery
].iloc[0]


passport_1, passport_2, passport_3, passport_4 = st.columns(4)

with passport_1:
    st.metric(
        "SOH",
        f"{selected_row['soh_percent']:.1f}%",
    )

    st.metric(
        "Status",
        selected_row["status"],
    )

with passport_2:
    st.metric(
        "Latest Cycle",
        f"{selected_row['latest_cycle']:.0f}",
    )

    st.metric(
        "Capacity",
        f"{selected_row['latest_capacity_Ah']:.4f} Ah",
    )

with passport_3:
    st.metric(
        "Median Voltage",
        f"{selected_row['median_voltage_V']:.3f} V",
    )

    st.metric(
        "Median Temperature",
        f"{selected_row['median_temperature_C']:.1f} °C",
    )

with passport_4:
    st.metric(
        "Incidents",
        int(
            selected_row["incident_count"]
        ),
    )

    st.metric(
        "Priority",
        selected_row["priority"],
    )


if not incident_df.empty:
    battery_incidents = incident_df[
        incident_df["battery_id"].astype(str)
        == selected_battery
    ].copy()

    if not battery_incidents.empty:
        st.markdown(
            "### Incident Timeline"
        )

        incident_columns = [
            column
            for column in [
                "generated_at",
                "incident_id",
                "fault_type",
                "severity_percent",
                "risk_level",
                "risk_score",
            ]
            if column
            in battery_incidents.columns
        ]

        st.dataframe(
            battery_incidents[
                incident_columns
            ].sort_values(
                "generated_at",
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No incidents are stored for this battery."
        )


st.subheader("Fleet Operations Recommendation")

if critical_count > 0:
    st.error(
        f"{critical_count} critical batteries require immediate action."
    )
elif warning_count > 0:
    st.warning(
        f"{warning_count} batteries require enhanced monitoring or inspection."
    )
else:
    st.success(
        "The filtered fleet is operating within the healthy region."
    )


st.download_button(
    "Download Fleet Snapshot CSV",
    data=filtered_df.drop(
        columns=["bubble_size"],
        errors="ignore",
    ).to_csv(
        index=False
    ).encode("utf-8"),
    file_name="battery_fleet_snapshot.csv",
    mime="text/csv",
)


if not failure_df.empty:
    with st.expander(
        "Battery files that could not be processed"
    ):
        st.dataframe(
            failure_df,
            width="stretch",
            hide_index=True,
        )
