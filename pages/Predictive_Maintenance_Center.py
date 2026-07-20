from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_FOLDER = Path(__file__).resolve().parents[1]

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


from utils.incident_report import load_incident_history


st.set_page_config(
    page_title="Predictive Maintenance Center",
    page_icon="🧠",
    layout="wide",
)


HISTORY_FOLDER = PROJECT_FOLDER / "incident_history"


def risk_rank(level: str) -> int:
    mapping = {
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Critical": 4,
    }
    return mapping.get(str(level), 0)


def maintenance_priority(level: str) -> str:
    mapping = {
        "Low": "Routine",
        "Moderate": "Schedule inspection",
        "High": "Act soon",
        "Critical": "Immediate action",
    }
    return mapping.get(str(level), "Review required")


def estimated_downtime_minutes(
    fault_type: str,
    risk_level: str,
) -> int:
    base_map = {
        "Cooling failure": 45,
        "Over-current": 30,
        "Over-temperature": 60,
        "Deep discharge": 25,
        "Fast charging": 35,
        "Internal resistance increase": 90,
        "Sensor drift": 20,
        "Capacity fade": 120,
        "Normal": 0,
    }

    multiplier_map = {
        "Low": 0.6,
        "Moderate": 1.0,
        "High": 1.4,
        "Critical": 2.0,
    }

    base = base_map.get(str(fault_type), 45)
    multiplier = multiplier_map.get(str(risk_level), 1.0)

    return int(round(base * multiplier))


def estimated_cost_band(
    fault_type: str,
    risk_level: str,
) -> str:
    high_cost_faults = {
        "Capacity fade",
        "Internal resistance increase",
    }

    if risk_level == "Critical":
        return "High"

    if fault_type in high_cost_faults:
        return "High"

    if risk_level == "High":
        return "Medium–High"

    if risk_level == "Moderate":
        return "Medium"

    return "Low"


def probable_actions(
    fault_type: str,
    risk_level: str,
) -> list[str]:
    actions = {
        "Cooling failure": [
            "Inspect the cooling fan or coolant loop.",
            "Check for blocked airflow or restricted cooling paths.",
            "Verify thermal sensor calibration.",
            "Reduce discharge load until cooling is restored.",
        ],
        "Over-current": [
            "Reduce current demand immediately.",
            "Inspect current-control logic and contactors.",
            "Check cable and connector heating.",
            "Verify protection thresholds.",
        ],
        "Over-temperature": [
            "Stop high-load operation.",
            "Inspect cooling performance.",
            "Check ambient thermal conditions.",
            "Repeat thermal diagnostics before reuse.",
        ],
        "Deep discharge": [
            "Stop discharge and recharge safely.",
            "Inspect low-voltage cutoff settings.",
            "Review cell-balancing performance.",
            "Check for irreversible capacity loss.",
        ],
        "Fast charging": [
            "Reduce charging current.",
            "Verify charging protocol limits.",
            "Monitor temperature during the next charge.",
            "Inspect for accelerated ageing indicators.",
        ],
        "Internal resistance increase": [
            "Measure internal resistance at rest and under load.",
            "Inspect electrical connections.",
            "Evaluate capacity and voltage sag.",
            "Plan module or cell replacement if degradation persists.",
        ],
        "Sensor drift": [
            "Calibrate the affected sensor.",
            "Compare sensor output with a reference instrument.",
            "Inspect wiring and connectors.",
            "Replace the sensor if drift persists.",
        ],
        "Capacity fade": [
            "Perform a controlled capacity test.",
            "Review historical SOH and RUL trends.",
            "Reduce load until replacement planning is complete.",
            "Schedule battery replacement.",
        ],
        "Normal": [
            "Continue routine monitoring.",
        ],
    }

    selected = actions.get(
        str(fault_type),
        [
            "Inspect the affected subsystem.",
            "Repeat the diagnostic test.",
            "Review recent battery telemetry.",
        ],
    )

    if risk_level == "Critical":
        selected = [
            "Stop operation immediately.",
        ] + selected

    return selected


def confidence_score(row: pd.Series) -> float:
    risk_score = float(row.get("risk_score", 0))
    severity = float(row.get("severity_percent", 0))
    soh = float(row.get("final_soh", 100))

    confidence = (
        68.0
        + 0.18 * risk_score
        + 0.10 * severity
        + 0.08 * max(0.0, 100.0 - soh)
    )

    return float(np.clip(confidence, 70.0, 97.0))


def safe_operating_cycles(row: pd.Series) -> int:
    rul = float(row.get("estimated_rul", 0))
    level = str(row.get("risk_level", "Low"))

    factor = {
        "Low": 0.75,
        "Moderate": 0.50,
        "High": 0.25,
        "Critical": 0.05,
    }.get(level, 0.40)

    return max(0, int(round(rul * factor)))


def failure_probability_50(row: pd.Series) -> float:
    risk_score = float(row.get("risk_score", 0))
    severity = float(row.get("severity_percent", 0))
    soh_loss = max(
        0.0,
        100.0 - float(row.get("final_soh", 100)),
    )

    probability = (
        5.0
        + 0.65 * risk_score
        + 0.20 * severity
        + 0.55 * soh_loss
    )

    return float(np.clip(probability, 1.0, 99.0))


st.title("🧠 Predictive Maintenance Center")

st.caption(
    "Convert saved battery incidents into maintenance priorities, "
    "recommended actions and operational decisions."
)


history_df = load_incident_history(HISTORY_FOLDER)

if history_df.empty:
    st.info(
        "No saved incidents were found. Create and save an incident "
        "from Fault Laboratory first."
    )
    st.stop()


history_df = history_df.copy()

history_df["risk_rank"] = (
    history_df["risk_level"]
    .astype(str)
    .map(risk_rank)
)

history_df = history_df.sort_values(
    ["risk_rank", "generated_at"],
    ascending=[False, False],
)


filter_1, filter_2, filter_3 = st.columns(3)

with filter_1:
    batteries = [
        "All"
    ] + sorted(
        history_df["battery_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_battery = st.selectbox(
        "Battery",
        batteries,
    )

with filter_2:
    risks = [
        "All"
    ] + sorted(
        history_df["risk_level"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_risk = st.selectbox(
        "Risk",
        risks,
    )

with filter_3:
    faults = [
        "All"
    ] + sorted(
        history_df["fault_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_fault = st.selectbox(
        "Fault",
        faults,
    )


filtered_df = history_df.copy()

if selected_battery != "All":
    filtered_df = filtered_df[
        filtered_df["battery_id"].astype(str)
        == selected_battery
    ]

if selected_risk != "All":
    filtered_df = filtered_df[
        filtered_df["risk_level"].astype(str)
        == selected_risk
    ]

if selected_fault != "All":
    filtered_df = filtered_df[
        filtered_df["fault_type"].astype(str)
        == selected_fault
    ]


kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)

with kpi_1:
    st.metric(
        "Open Incidents",
        len(filtered_df),
    )

with kpi_2:
    st.metric(
        "High/Critical",
        int(
            filtered_df["risk_level"]
            .astype(str)
            .isin(["High", "Critical"])
            .sum()
        ),
    )

with kpi_3:
    st.metric(
        "Average Risk Score",
        f"{filtered_df['risk_score'].mean():.1f}/100",
    )

with kpi_4:
    st.metric(
        "Affected Batteries",
        filtered_df["battery_id"]
        .astype(str)
        .nunique(),
    )


st.subheader("Maintenance Queue")

queue_df = filtered_df.copy()

queue_df["priority"] = (
    queue_df["risk_level"]
    .astype(str)
    .map(maintenance_priority)
)

queue_df["safe_cycles"] = queue_df.apply(
    safe_operating_cycles,
    axis=1,
)

queue_df["failure_probability_50"] = queue_df.apply(
    failure_probability_50,
    axis=1,
)

queue_columns = [
    column
    for column in [
        "incident_id",
        "battery_id",
        "fault_type",
        "risk_level",
        "risk_score",
        "final_soh",
        "estimated_rul",
        "safe_cycles",
        "failure_probability_50",
        "priority",
        "generated_at",
    ]
    if column in queue_df.columns
]

st.dataframe(
    queue_df[queue_columns],
    use_container_width=True,
    hide_index=True,
)


st.subheader("AI Maintenance Engineer")

incident_options = (
    filtered_df["incident_id"]
    .dropna()
    .astype(str)
    .tolist()
)

selected_incident = st.selectbox(
    "Select incident",
    incident_options,
)

selected_row = filtered_df[
    filtered_df["incident_id"].astype(str)
    == selected_incident
].iloc[0]


fault_type = str(
    selected_row.get("fault_type", "Unknown")
)

risk_level = str(
    selected_row.get("risk_level", "Low")
)

risk_score = float(
    selected_row.get("risk_score", 0)
)

final_soh = float(
    selected_row.get("final_soh", 0)
)

estimated_rul = float(
    selected_row.get("estimated_rul", 0)
)

confidence = confidence_score(
    selected_row
)

safe_cycles = safe_operating_cycles(
    selected_row
)

failure_probability = failure_probability_50(
    selected_row
)

downtime = estimated_downtime_minutes(
    fault_type,
    risk_level,
)

cost_band = estimated_cost_band(
    fault_type,
    risk_level,
)

priority = maintenance_priority(
    risk_level
)


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Maintenance Priority",
        priority,
    )

with metric_2:
    st.metric(
        "AI Confidence",
        f"{confidence:.0f}%",
    )

with metric_3:
    st.metric(
        "Safe Operating Window",
        f"{safe_cycles} cycles",
    )

with metric_4:
    st.metric(
        "Failure Probability",
        f"{failure_probability:.0f}%",
        help="Heuristic probability of a meaningful failure event within the next 50 cycles.",
    )


left_panel, right_panel = st.columns(
    [1.15, 0.85]
)

with left_panel:
    st.markdown("### Recommended Maintenance Actions")

    for action in probable_actions(
        fault_type,
        risk_level,
    ):
        st.write(f"✓ {action}")

    st.markdown("### Engineering Decision")

    if risk_level == "Critical":
        st.error(
            "Stop operation and isolate the battery before further use."
        )
    elif risk_level == "High":
        st.warning(
            "Reduce load and schedule maintenance before continued operation."
        )
    elif risk_level == "Moderate":
        st.warning(
            "Continue only with enhanced monitoring and schedule inspection."
        )
    else:
        st.success(
            "Routine operation may continue with normal monitoring."
        )

with right_panel:
    st.markdown("### Maintenance Estimate")

    st.metric(
        "Estimated Downtime",
        f"{downtime} min",
    )

    st.metric(
        "Estimated Cost Band",
        cost_band,
    )

    st.metric(
        "Indicative RUL",
        f"{estimated_rul:.0f} cycles",
    )

    st.metric(
        "Final SOH",
        f"{final_soh:.1f}%",
    )


st.subheader("Risk and Remaining Life")

gauge_col, rul_col = st.columns(2)

with gauge_col:
    risk_figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 20]},
                    {"range": [20, 45]},
                    {"range": [45, 70]},
                    {"range": [70, 100]},
                ],
            },
            title={"text": "Risk Score"},
        )
    )

    risk_figure.update_layout(
        height=360,
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        risk_figure,
        use_container_width=True,
    )

with rul_col:
    rul_figure = go.Figure()

    rul_figure.add_trace(
        go.Bar(
            x=[
                "Safe operating window",
                "Remaining RUL",
            ],
            y=[
                safe_cycles,
                estimated_rul,
            ],
            text=[
                f"{safe_cycles} cycles",
                f"{estimated_rul:.0f} cycles",
            ],
            textposition="auto",
        )
    )

    rul_figure.update_layout(
        height=360,
        yaxis_title="Cycles",
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=40,
        ),
    )

    st.plotly_chart(
        rul_figure,
        use_container_width=True,
    )


st.subheader("Maintenance Work Order")

work_order = pd.DataFrame(
    {
        "Field": [
            "Incident ID",
            "Battery",
            "Fault",
            "Risk",
            "Priority",
            "Recommended completion",
            "Estimated downtime",
            "Cost band",
            "AI confidence",
        ],
        "Value": [
            selected_incident,
            str(selected_row.get("battery_id", "Unknown")),
            fault_type,
            risk_level,
            priority,
            (
                "Immediate"
                if risk_level == "Critical"
                else (
                    "Before next high-load operation"
                    if risk_level == "High"
                    else (
                        "Within the next maintenance window"
                        if risk_level == "Moderate"
                        else "Routine schedule"
                    )
                )
            ),
            f"{downtime} minutes",
            cost_band,
            f"{confidence:.0f}%",
        ],
    }
)

st.dataframe(
    work_order,
    use_container_width=True,
    hide_index=True,
)


work_order_csv = work_order.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "📋 Download Maintenance Work Order",
    data=work_order_csv,
    file_name=(
        f"{selected_incident}_maintenance_work_order.csv"
    ),
    mime="text/csv",
)
