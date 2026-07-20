from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def create_incident_id(
    battery_id: str,
    timestamp: Optional[datetime] = None,
) -> str:
    timestamp = timestamp or datetime.now()

    safe_battery = (
        str(battery_id)
        .replace(" ", "")
        .replace("/", "-")
    )

    return (
        f"INC-{timestamp:%Y%m%d-%H%M%S}-"
        f"{safe_battery}"
    )


def risk_color(risk_level: str):
    mapping = {
        "Low": colors.HexColor("#15803d"),
        "Moderate": colors.HexColor("#ca8a04"),
        "High": colors.HexColor("#ea580c"),
        "Critical": colors.HexColor("#dc2626"),
    }

    return mapping.get(
        str(risk_level),
        colors.HexColor("#475569"),
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return float(default)


def build_event_log(
    fault_df: pd.DataFrame,
    baseline_voltage: float,
    baseline_temperature: float,
    baseline_soh: float,
) -> list[dict]:
    if fault_df.empty:
        return []

    events: list[dict] = []

    start_time = _safe_float(
        fault_df["time_s"].iloc[0]
    )

    events.append(
        {
            "time_s": start_time,
            "event": "Fault simulation initialized",
            "details": "Baseline operating state recorded.",
        }
    )

    voltage_series = pd.to_numeric(
        fault_df["true_voltage_V"],
        errors="coerce",
    )

    temperature_series = pd.to_numeric(
        fault_df["true_temperature_C"],
        errors="coerce",
    )

    soh_series = pd.to_numeric(
        fault_df["fault_soh_percent"],
        errors="coerce",
    )

    event_rules = [
        (
            temperature_series
            >= baseline_temperature + 2.0,
            "Temperature exceeded baseline",
            lambda row: (
                f"Temperature reached "
                f"{row['true_temperature_C']:.1f} °C."
            ),
        ),
        (
            voltage_series
            <= baseline_voltage - 0.20,
            "Voltage decline detected",
            lambda row: (
                f"Voltage decreased to "
                f"{row['true_voltage_V']:.3f} V."
            ),
        ),
        (
            voltage_series <= 3.20,
            "Voltage warning threshold reached",
            lambda row: (
                f"Voltage reached "
                f"{row['true_voltage_V']:.3f} V."
            ),
        ),
        (
            voltage_series <= 3.00,
            "Critical voltage threshold reached",
            lambda row: (
                f"Voltage reached "
                f"{row['true_voltage_V']:.3f} V."
            ),
        ),
        (
            temperature_series >= 40.0,
            "Thermal warning region reached",
            lambda row: (
                f"Temperature reached "
                f"{row['true_temperature_C']:.1f} °C."
            ),
        ),
        (
            temperature_series >= 45.0,
            "Thermal warning threshold exceeded",
            lambda row: (
                f"Temperature reached "
                f"{row['true_temperature_C']:.1f} °C."
            ),
        ),
        (
            soh_series
            <= baseline_soh - 1.0,
            "SOH degradation detected",
            lambda row: (
                f"SOH decreased to "
                f"{row['fault_soh_percent']:.1f}%."
            ),
        ),
        (
            soh_series <= 80.0,
            "End-of-life threshold reached",
            lambda row: (
                f"SOH reached "
                f"{row['fault_soh_percent']:.1f}%."
            ),
        ),
    ]

    recorded_names = set()

    for mask, event_name, details_function in event_rules:
        matching = fault_df.loc[
            mask.fillna(False)
        ]

        if (
            not matching.empty
            and event_name not in recorded_names
        ):
            row = matching.iloc[0]

            events.append(
                {
                    "time_s": _safe_float(
                        row["time_s"]
                    ),
                    "event": event_name,
                    "details": details_function(row),
                }
            )

            recorded_names.add(event_name)

    final_row = fault_df.iloc[-1]

    events.append(
        {
            "time_s": _safe_float(
                final_row["time_s"]
            ),
            "event": "Simulation completed",
            "details": (
                "Final state and maintenance "
                "recommendation generated."
            ),
        }
    )

    return sorted(
        events,
        key=lambda item: item["time_s"],
    )


def _plot_to_png(
    fault_df: pd.DataFrame,
    x_column: str,
    y_columns: Iterable[str],
    labels: Iterable[str],
    title: str,
    y_label: str,
) -> bytes:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(
        figsize=(7.4, 3.4)
    )

    for column, label in zip(
        y_columns,
        labels,
    ):
        axis.plot(
            fault_df[x_column],
            fault_df[column],
            label=label,
            linewidth=2,
        )

    axis.set_title(title)
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()

    buffer = BytesIO()

    figure.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)
    buffer.seek(0)

    return buffer.getvalue()


def generate_incident_pdf(
    incident: Dict[str, Any],
    fault_df: pd.DataFrame,
    event_log: list[dict],
) -> bytes:
    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="AI Battery Incident Report",
        author="NASA Battery Digital Twin",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "IncidentTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "IncidentSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )

    heading_style = ParagraphStyle(
        "IncidentHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=7,
    )

    body_style = ParagraphStyle(
        "IncidentBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )

    story = [
        Paragraph(
            "AI BATTERY INCIDENT REPORT",
            title_style,
        ),
        Paragraph(
            "NASA Battery Digital Twin - "
            "Fault Diagnosis and Predictive Maintenance",
            subtitle_style,
        ),
    ]

    report_risk_color = risk_color(
        incident["risk_level"]
    )

    risk_table = Table(
        [
            [
                Paragraph(
                    "<b>INCIDENT ID</b>",
                    body_style,
                ),
                incident["incident_id"],
                Paragraph(
                    "<b>RISK LEVEL</b>",
                    body_style,
                ),
                incident["risk_level"].upper(),
            ]
        ],
        colWidths=[
            30 * mm,
            58 * mm,
            27 * mm,
            45 * mm,
        ],
    )

    risk_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (3, 0), (3, 0), report_risk_color),
                ("FONTNAME", (3, 0), (3, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.extend(
        [
            risk_table,
            Spacer(1, 10),
            Paragraph(
                "1. Incident Information",
                heading_style,
            ),
        ]
    )

    incident_rows = [
        ["Battery", incident["battery_id"]],
        ["Date generated", incident["generated_at"]],
        ["Fault type", incident["fault_type"]],
        ["Fault severity", f"{incident['severity_percent']}%"],
        ["Starting discharge cycle", str(incident["cycle"])],
        ["Simulation steps", str(incident["simulation_steps"])],
        ["Time step", f"{incident['time_step_s']:.2f} s"],
    ]

    incident_table = Table(
        incident_rows,
        colWidths=[55 * mm, 105 * mm],
    )

    incident_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e2e8f0")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.extend(
        [
            incident_table,
            Paragraph(
                "2. Initial and Final Operating State",
                heading_style,
            ),
        ]
    )

    state_data = [
        [
            "Metric",
            "Initial",
            "Final / Maximum",
            "Change",
        ],
        [
            "Voltage (V)",
            f"{incident['baseline_voltage']:.3f}",
            f"{incident['final_voltage']:.3f}",
            f"{incident['final_voltage'] - incident['baseline_voltage']:+.3f}",
        ],
        [
            "Temperature (°C)",
            f"{incident['baseline_temperature']:.1f}",
            f"{incident['maximum_temperature']:.1f}",
            f"{incident['maximum_temperature'] - incident['baseline_temperature']:+.1f}",
        ],
        [
            "SOH (%)",
            f"{incident['baseline_soh']:.1f}",
            f"{incident['final_soh']:.1f}",
            f"{incident['final_soh'] - incident['baseline_soh']:+.1f}",
        ],
        [
            "Risk score",
            "0",
            f"{incident['risk_score']}",
            f"+{incident['risk_score']}",
        ],
    ]

    state_table = Table(
        state_data,
        colWidths=[
            45 * mm,
            35 * mm,
            45 * mm,
            35 * mm,
        ],
    )

    state_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.extend(
        [
            state_table,
            Paragraph(
                "3. Fault Progression Timeline",
                heading_style,
            ),
        ]
    )

    event_rows = [
        ["Simulation time", "Event", "Details"]
    ]

    for event in event_log:
        event_rows.append(
            [
                f"{event['time_s']:.1f} s",
                event["event"],
                event["details"],
            ]
        )

    event_table = Table(
        event_rows,
        repeatRows=1,
        colWidths=[
            28 * mm,
            55 * mm,
            77 * mm,
        ],
    )

    event_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.extend(
        [
            event_table,
            PageBreak(),
            Paragraph(
                "4. Fault Response Curves",
                heading_style,
            ),
        ]
    )

    voltage_png = _plot_to_png(
        fault_df=fault_df,
        x_column="time_s",
        y_columns=[
            "voltage_V",
            "true_voltage_V",
        ],
        labels=[
            "Baseline predicted voltage",
            "Fault-affected voltage",
        ],
        title="Voltage Response",
        y_label="Voltage (V)",
    )

    temperature_png = _plot_to_png(
        fault_df=fault_df,
        x_column="time_s",
        y_columns=[
            "temperature_C",
            "true_temperature_C",
        ],
        labels=[
            "Baseline predicted temperature",
            "Fault-affected temperature",
        ],
        title="Temperature Response",
        y_label="Temperature (°C)",
    )

    soh_png = _plot_to_png(
        fault_df=fault_df,
        x_column="time_s",
        y_columns=[
            "fault_soh_percent",
        ],
        labels=[
            "Fault-adjusted SOH",
        ],
        title="SOH Response",
        y_label="SOH (%)",
    )

    for image_bytes, caption in [
        (voltage_png, "Figure 1. Voltage response under the injected fault."),
        (temperature_png, "Figure 2. Temperature response under the injected fault."),
        (soh_png, "Figure 3. SOH response under the injected fault."),
    ]:
        image_buffer = BytesIO(image_bytes)

        story.extend(
            [
                Image(
                    image_buffer,
                    width=175 * mm,
                    height=80 * mm,
                ),
                Paragraph(
                    caption,
                    body_style,
                ),
                Spacer(1, 8),
            ]
        )

    story.extend(
        [
            Paragraph(
                "5. AI Diagnosis and Maintenance Recommendation",
                heading_style,
            )
        ]
    )

    diagnosis_data = [
        ["Most probable cause", incident["probable_cause"]],
        ["AI confidence", f"{incident['decision_confidence']:.0f}%"],
        ["Risk level", incident["risk_level"]],
        ["Maintenance urgency", incident["maintenance_urgency"]],
        ["Maintenance window", incident["maintenance_window"]],
        ["Indicative RUL", f"{incident['estimated_rul']:.0f} cycles"],
        ["Recommended action", incident["recommended_action"]],
        ["Decision reason", incident["decision_reason"]],
    ]

    diagnosis_table = Table(
        diagnosis_data,
        colWidths=[50 * mm, 110 * mm],
    )

    diagnosis_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e2e8f0")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.extend(
        [
            diagnosis_table,
            Spacer(1, 10),
            Paragraph(
                "Digital Twin Summary",
                heading_style,
            ),
            Paragraph(
                incident["summary"],
                body_style,
            ),
            Spacer(1, 12),
            Paragraph(
                (
                    "<i>This report was generated automatically "
                    "by the NASA Battery Digital Twin research "
                    "prototype. Results are intended for research "
                    "and decision-support demonstration and are "
                    "not certified safety instructions.</i>"
                ),
                body_style,
            ),
        ]
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(
            colors.HexColor("#64748b")
        )
        canvas.drawString(
            16 * mm,
            9 * mm,
            incident["incident_id"],
        )
        canvas.drawRightString(
            A4[0] - 16 * mm,
            9 * mm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    document.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
    )

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()



def ensure_incident_directories(
    history_folder: Path,
) -> dict[str, Path]:
    history_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    directories = {
        "root": history_folder,
        "pdf": history_folder / "PDFs",
        "csv": history_folder / "CSV",
        "json": history_folder / "JSON",
    }

    for directory in directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


def save_complete_incident(
    incident: Dict[str, Any],
    fault_df: pd.DataFrame,
    pdf_bytes: bytes,
    history_folder: Path,
) -> dict[str, Path]:
    directories = ensure_incident_directories(
        history_folder
    )

    incident_id = str(
        incident["incident_id"]
    )

    pdf_path = (
        directories["pdf"]
        / f"{incident_id}.pdf"
    )

    csv_path = (
        directories["csv"]
        / f"{incident_id}_simulation.csv"
    )

    json_path = (
        directories["json"]
        / f"{incident_id}.json"
    )

    history_csv_path = (
        directories["root"]
        / "incident_history.csv"
    )

    pdf_path.write_bytes(
        pdf_bytes
    )

    fault_df.to_csv(
        csv_path,
        index=False,
    )

    metadata = dict(incident)
    metadata["pdf_path"] = str(pdf_path)
    metadata["csv_path"] = str(csv_path)
    metadata["json_path"] = str(json_path)

    json_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    new_row = pd.DataFrame(
        [metadata]
    )

    if history_csv_path.exists():
        try:
            history_df = pd.read_csv(
                history_csv_path
            )
        except pd.errors.EmptyDataError:
            history_df = pd.DataFrame()

        if (
            not history_df.empty
            and "incident_id" in history_df.columns
        ):
            history_df = history_df[
                history_df["incident_id"].astype(str)
                != incident_id
            ]

        history_df = pd.concat(
            [history_df, new_row],
            ignore_index=True,
        )
    else:
        history_df = new_row

    history_df.to_csv(
        history_csv_path,
        index=False,
    )

    return {
        "history_csv": history_csv_path,
        "pdf": pdf_path,
        "csv": csv_path,
        "json": json_path,
    }


def load_incident_history(
    history_folder: Path,
) -> pd.DataFrame:
    history_csv_path = (
        history_folder
        / "incident_history.csv"
    )

    if not history_csv_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            history_csv_path
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
