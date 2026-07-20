from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_FOLDER = Path(__file__).resolve().parents[1]

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


from utils.incident_report import (
    load_incident_history,
)


st.set_page_config(
    page_title="Incident History",
    page_icon="📚",
    layout="wide",
)


st.title("📚 Incident History")

st.caption(
    "Search, review and download saved battery incidents."
)


HISTORY_FOLDER = (
    PROJECT_FOLDER
    / "incident_history"
)

history_df = load_incident_history(
    HISTORY_FOLDER
)


if history_df.empty:
    st.info(
        "No incidents have been saved yet. Open Fault "
        "Laboratory, inject a fault, then click "
        "`Save to Incident History`."
    )

    st.write("Storage location:")

    st.code(
        str(HISTORY_FOLDER),
        language=None,
    )

    st.stop()


filter_1, filter_2, filter_3, filter_4 = st.columns(4)

with filter_1:
    battery_options = [
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
        battery_options,
    )

with filter_2:
    fault_options = [
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
        fault_options,
    )

with filter_3:
    risk_options = [
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
        risk_options,
    )

with filter_4:
    search_text = st.text_input(
        "Search incident ID",
        placeholder="INC-...",
    )


filtered_df = history_df.copy()

if selected_battery != "All":
    filtered_df = filtered_df[
        filtered_df["battery_id"].astype(str)
        == selected_battery
    ]

if selected_fault != "All":
    filtered_df = filtered_df[
        filtered_df["fault_type"].astype(str)
        == selected_fault
    ]

if selected_risk != "All":
    filtered_df = filtered_df[
        filtered_df["risk_level"].astype(str)
        == selected_risk
    ]

if search_text.strip():
    filtered_df = filtered_df[
        filtered_df["incident_id"]
        .astype(str)
        .str.contains(
            search_text.strip(),
            case=False,
            na=False,
        )
    ]


summary_1, summary_2, summary_3, summary_4 = st.columns(4)

with summary_1:
    st.metric(
        "Saved Incidents",
        len(filtered_df),
    )

with summary_2:
    st.metric(
        "Critical",
        int(
            (
                filtered_df["risk_level"]
                .astype(str)
                == "Critical"
            ).sum()
        ),
    )

with summary_3:
    st.metric(
        "High Risk",
        int(
            (
                filtered_df["risk_level"]
                .astype(str)
                == "High"
            ).sum()
        ),
    )

with summary_4:
    st.metric(
        "Batteries",
        filtered_df["battery_id"]
        .astype(str)
        .nunique(),
    )


display_columns = [
    column
    for column in [
        "incident_id",
        "generated_at",
        "battery_id",
        "cycle",
        "fault_type",
        "severity_percent",
        "risk_level",
        "risk_score",
        "final_voltage",
        "maximum_temperature",
        "final_soh",
        "estimated_rul",
        "maintenance_urgency",
    ]
    if column in filtered_df.columns
]


st.dataframe(
    filtered_df[
        display_columns
    ].sort_values(
        "generated_at",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)


st.subheader("Open Saved Incident")

incident_options = (
    filtered_df["incident_id"]
    .dropna()
    .astype(str)
    .tolist()
)

if incident_options:
    selected_incident = st.selectbox(
        "Incident",
        incident_options,
    )

    selected_row = filtered_df[
        filtered_df["incident_id"].astype(str)
        == selected_incident
    ].iloc[0]

    detail_1, detail_2, detail_3 = st.columns(3)

    with detail_1:
        st.metric(
            "Battery",
            str(selected_row.get("battery_id", "Unknown")),
        )

        st.metric(
            "Fault",
            str(selected_row.get("fault_type", "Unknown")),
        )

    with detail_2:
        st.metric(
            "Risk",
            str(selected_row.get("risk_level", "Unknown")),
        )

        st.metric(
            "Risk Score",
            f"{selected_row.get('risk_score', 0)}/100",
        )

    with detail_3:
        st.metric(
            "Final SOH",
            f"{float(selected_row.get('final_soh', 0)):.1f}%",
        )

        st.metric(
            "RUL",
            f"{float(selected_row.get('estimated_rul', 0)):.0f} cycles",
        )


    pdf_path = Path(
        str(selected_row.get("pdf_path", ""))
    )

    csv_path = Path(
        str(selected_row.get("csv_path", ""))
    )

    json_path = Path(
        str(selected_row.get("json_path", ""))
    )


    download_1, download_2, download_3 = st.columns(3)

    with download_1:
        if pdf_path.exists():
            st.download_button(
                "📄 Download Saved PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning("Saved PDF file was not found.")

    with download_2:
        if csv_path.exists():
            st.download_button(
                "📊 Download Simulation CSV",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning("Saved simulation CSV was not found.")

    with download_3:
        if json_path.exists():
            st.download_button(
                "🧾 Download Metadata JSON",
                data=json_path.read_bytes(),
                file_name=json_path.name,
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.warning("Saved JSON file was not found.")


st.subheader("Export Filtered History")

st.download_button(
    "Download Incident History CSV",
    data=filtered_df.to_csv(
        index=False
    ).encode("utf-8"),
    file_name="incident_history_filtered.csv",
    mime="text/csv",
)


with st.expander(
    "Storage location"
):
    st.code(
        str(HISTORY_FOLDER),
        language=None,
    )

    st.code(
        """
incident_history/
├── incident_history.csv
├── PDFs/
├── CSV/
└── JSON/
        """.strip(),
        language=None,
    )
