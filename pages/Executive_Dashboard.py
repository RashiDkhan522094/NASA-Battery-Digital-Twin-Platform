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


from utils.executive_dashboard import (
    build_executive_snapshot,
    estimated_rul_from_soh,
    fleet_score,
    list_battery_files,
)

from utils.incident_report import (
    load_incident_history,
)


st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="🚀",
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
    .executive-title {
        font-size: 2.55rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .executive-subtitle {
        color: #94a3b8;
        margin-bottom: 1.3rem;
    }

    .executive-card {
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        background: rgba(15,23,42,0.55);
        min-height: 128px;
    }

    .executive-label {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .executive-value {
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="executive-title">'
    '🚀 NASA Battery Digital Twin Command Center'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="executive-subtitle">'
    'Executive overview of fleet health, risk, incidents, '
    'remaining life and maintenance priorities.'
    '</div>',
    unsafe_allow_html=True,
)


battery_files = list_battery_files(
    PROCESSED_FOLDER
)

if not battery_files:
    st.error(
        "No processed battery datasets were found."
    )
    st.stop()


with st.spinner(
    "Building executive fleet snapshot..."
):
    snapshot_df, failure_df = (
        build_executive_snapshot(
            battery_files
        )
    )


if snapshot_df.empty:
    st.error(
        "No valid battery summaries could be created."
    )

    if not failure_df.empty:
        st.dataframe(
            failure_df,
            use_container_width=True,
            hide_index=True,
        )

    st.stop()


incident_df = load_incident_history(
    INCIDENT_FOLDER
)

snapshot_df = snapshot_df.copy()

snapshot_df["estimated_rul"] = snapshot_df.apply(
    lambda row: estimated_rul_from_soh(
        soh_percent=row["soh_percent"],
        latest_cycle=row["latest_cycle"],
    ),
    axis=1,
)


incident_count_by_battery = pd.Series(
    dtype=int
)

latest_fault_by_battery = pd.Series(
    dtype=object
)

if not incident_df.empty:
    incident_df = incident_df.copy()

    incident_count_by_battery = (
        incident_df.groupby(
            "battery_id"
        )["incident_id"]
        .count()
    )

    latest_fault_by_battery = (
        incident_df.sort_values(
            "generated_at"
        )
        .groupby(
            "battery_id"
        )["fault_type"]
        .last()
    )


snapshot_df["incident_count"] = (
    snapshot_df["battery_id"]
    .map(incident_count_by_battery)
    .fillna(0)
    .astype(int)
)

snapshot_df["latest_fault"] = (
    snapshot_df["battery_id"]
    .map(latest_fault_by_battery)
    .fillna("None")
)


fleet_score_value = fleet_score(
    snapshot_df,
    incident_df,
)

fleet_size = len(snapshot_df)

healthy_count = int(
    (
        snapshot_df["status"]
        == "Healthy"
    ).sum()
)

warning_count = int(
    snapshot_df["status"]
    .isin(
        [
            "Warning",
            "High Risk",
        ]
    )
    .sum()
)

critical_count = int(
    (
        snapshot_df["status"]
        == "Critical"
    ).sum()
)

average_soh = float(
    snapshot_df["soh_percent"].mean()
)

average_rul = float(
    snapshot_df["estimated_rul"].mean()
)

incident_count = len(incident_df)

maintenance_due = int(
    snapshot_df["status"]
    .isin(
        [
            "High Risk",
            "Critical",
        ]
    )
    .sum()
)


kpi_row_1 = st.columns(4)

kpis_1 = [
    ("Fleet Size", fleet_size),
    ("Average SOH", f"{average_soh:.1f}%"),
    ("Average RUL", f"{average_rul:.0f} cycles"),
    ("Fleet Score", f"{fleet_score_value:.0f}/100"),
]

for column, (
    label,
    value,
) in zip(
    kpi_row_1,
    kpis_1,
):
    with column:
        st.metric(
            label,
            value,
        )


kpi_row_2 = st.columns(4)

kpis_2 = [
    ("Healthy", healthy_count),
    ("Warning / High Risk", warning_count),
    ("Critical", critical_count),
    ("Maintenance Due", maintenance_due),
]

for column, (
    label,
    value,
) in zip(
    kpi_row_2,
    kpis_2,
):
    with column:
        st.metric(
            label,
            value,
        )


st.subheader("Live Fleet Status")

if critical_count > 0:
    st.error(
        f"{critical_count} critical batteries require immediate action."
    )
elif warning_count > 0:
    st.warning(
        f"{warning_count} batteries require inspection or enhanced monitoring."
    )
else:
    st.success(
        "The fleet is operating within the healthy region."
    )


st.subheader("Fleet Score")

score_col, summary_col = st.columns(
    [0.9, 1.1]
)

with score_col:
    score_figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fleet_score_value,
            number={"suffix": "/100"},
            title={"text": "Fleet Health Score"},
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "thickness": 0.35
                },
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "rgba(220,38,38,0.25)",
                    },
                    {
                        "range": [50, 75],
                        "color": "rgba(234,179,8,0.25)",
                    },
                    {
                        "range": [75, 100],
                        "color": "rgba(34,197,94,0.25)",
                    },
                ],
            },
        )
    )

    score_figure.update_layout(
        height=360,
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        score_figure,
        use_container_width=True,
    )

with summary_col:
    st.markdown("### Executive Summary")

    st.write(
        f"• Fleet contains **{fleet_size} batteries**."
    )

    st.write(
        f"• Average SOH is **{average_soh:.1f}%**."
    )

    st.write(
        f"• Average estimated RUL is **{average_rul:.0f} cycles**."
    )

    st.write(
        f"• Stored incident count is **{incident_count}**."
    )

    if critical_count > 0:
        st.write(
            f"• **{critical_count} critical batteries** should be reviewed first."
        )
    elif warning_count > 0:
        st.write(
            f"• **{warning_count} batteries** require inspection."
        )
    else:
        st.write(
            "• No immediate fleet-wide maintenance action is required."
        )


st.subheader("Compact Fleet Health Map")

symbol_map = {
    "Healthy": "🟢",
    "Warning": "🟡",
    "High Risk": "🟠",
    "Critical": "🔴",
}

grid_columns = st.columns(8)

for index, row in snapshot_df.sort_values(
    "soh_percent"
).reset_index(drop=True).iterrows():
    with grid_columns[index % 8]:
        symbol = symbol_map.get(
            row["status"],
            "⚪",
        )

        st.markdown(
            f"""
            <div class="executive-card">
                <div class="executive-label">
                    {symbol} {row['battery_id']}
                </div>
                <div class="executive-value">
                    {row['soh_percent']:.1f}%
                </div>
                <div>
                    {row['status']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.subheader("Fleet Health and Remaining Life")

chart_1, chart_2 = st.columns(2)

with chart_1:
    soh_figure = px.histogram(
        snapshot_df,
        x="soh_percent",
        nbins=12,
        title="SOH Distribution",
        labels={
            "soh_percent": "SOH (%)",
        },
    )

    soh_figure.add_vline(
        x=80,
        line_dash="dash",
        annotation_text="80% EOL",
    )

    soh_figure.update_layout(
        height=420,
    )

    st.plotly_chart(
        soh_figure,
        use_container_width=True,
    )

with chart_2:
    rul_figure = px.histogram(
        snapshot_df,
        x="estimated_rul",
        nbins=12,
        title="Estimated RUL Distribution",
        labels={
            "estimated_rul": "Estimated RUL (cycles)",
        },
    )

    rul_figure.update_layout(
        height=420,
    )

    st.plotly_chart(
        rul_figure,
        use_container_width=True,
    )


st.subheader("Critical and High-Risk Batteries")

priority_df = snapshot_df[
    snapshot_df["status"]
    .isin(
        [
            "Critical",
            "High Risk",
        ]
    )
].copy()

if priority_df.empty:
    st.success(
        "No critical or high-risk batteries are present."
    )
else:
    priority_df["priority"] = np.where(
        priority_df["status"]
        == "Critical",
        "Immediate action",
        "Act soon",
    )

    st.dataframe(
        priority_df[
            [
                "battery_id",
                "status",
                "soh_percent",
                "capacity_Ah",
                "estimated_rul",
                "incident_count",
                "latest_fault",
                "priority",
            ]
        ].sort_values(
            "soh_percent"
        ),
        use_container_width=True,
        hide_index=True,
    )


st.subheader("Incident Trend")

if incident_df.empty:
    st.info(
        "No incident history is available yet."
    )
else:
    incident_df["generated_at"] = pd.to_datetime(
        incident_df["generated_at"],
        errors="coerce",
    )

    incident_trend = (
        incident_df.dropna(
            subset=["generated_at"]
        )
        .assign(
            incident_date=lambda frame: (
                frame["generated_at"]
                .dt.date
            )
        )
        .groupby(
            "incident_date",
            as_index=False,
        )["incident_id"]
        .count()
        .rename(
            columns={
                "incident_id": "incident_count"
            }
        )
    )

    trend_figure = px.line(
        incident_trend,
        x="incident_date",
        y="incident_count",
        markers=True,
        title="Incident Count Over Time",
        labels={
            "incident_date": "Date",
            "incident_count": "Incidents",
        },
    )

    trend_figure.update_layout(
        height=380,
    )

    st.plotly_chart(
        trend_figure,
        use_container_width=True,
    )


st.subheader("AI Executive Recommendation")

if critical_count > 0:
    worst_battery = (
        snapshot_df.sort_values(
            "soh_percent"
        )
        .iloc[0]
    )

    st.error(
        f"Prioritize battery {worst_battery['battery_id']} "
        f"for replacement or immediate engineering review. "
        f"Its current SOH is {worst_battery['soh_percent']:.1f}%."
    )
elif warning_count > 0:
    st.warning(
        "Schedule inspection for warning and high-risk batteries "
        "before the next high-load operating period."
    )
else:
    st.success(
        "Continue routine operation and periodic health monitoring."
    )


st.subheader("Quick Exports")

export_col_1, export_col_2 = st.columns(2)

with export_col_1:
    st.download_button(
        "Download Executive Fleet Snapshot",
        data=snapshot_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="executive_fleet_snapshot.csv",
        mime="text/csv",
        use_container_width=True,
    )

with export_col_2:
    if not incident_df.empty:
        st.download_button(
            "Download Incident Summary",
            data=incident_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="executive_incident_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button(
            "No Incident Summary Available",
            disabled=True,
            use_container_width=True,
        )


if not failure_df.empty:
    with st.expander(
        "Battery files that could not be summarized"
    ):
        st.dataframe(
            failure_df,
            use_container_width=True,
            hide_index=True,
        )
