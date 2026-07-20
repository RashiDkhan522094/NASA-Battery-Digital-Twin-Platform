from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_incident_bundle(
    project_folder: Path,
    incident_record: dict[str, Any],
    fault_df: pd.DataFrame,
    incident_pdf: bytes,
) -> dict[str, Path]:
    """
    Save one incident permanently inside:

        <project_folder>/incident_history/
            incident_history.csv
            PDFs/<incident_id>.pdf
            CSV/<incident_id>_simulation.csv
            JSON/<incident_id>.json
    """
    project_folder = Path(project_folder).resolve()

    if not project_folder.exists():
        raise FileNotFoundError(
            f"Project folder does not exist: {project_folder}"
        )

    incident_id = str(
        incident_record.get("incident_id", "")
    ).strip()

    if not incident_id:
        raise ValueError(
            "incident_record must contain a non-empty incident_id."
        )

    if incident_pdf is None or len(incident_pdf) == 0:
        raise ValueError(
            "incident_pdf is empty. Generate the PDF before saving."
        )

    if fault_df is None or fault_df.empty:
        raise ValueError(
            "fault_df is empty. Run the fault simulation before saving."
        )

    history_folder = (
        project_folder
        / "incident_history"
    )

    pdf_folder = history_folder / "PDFs"
    csv_folder = history_folder / "CSV"
    json_folder = history_folder / "JSON"

    for folder in (
        history_folder,
        pdf_folder,
        csv_folder,
        json_folder,
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    pdf_path = (
        pdf_folder
        / f"{incident_id}.pdf"
    )

    simulation_csv_path = (
        csv_folder
        / f"{incident_id}_simulation.csv"
    )

    metadata_json_path = (
        json_folder
        / f"{incident_id}.json"
    )

    history_csv_path = (
        history_folder
        / "incident_history.csv"
    )

    # Save files.
    pdf_path.write_bytes(
        incident_pdf
    )

    fault_df.to_csv(
        simulation_csv_path,
        index=False,
    )

    stored_record = dict(
        incident_record
    )

    stored_record.update(
        {
            "pdf_path": str(pdf_path),
            "csv_path": str(
                simulation_csv_path
            ),
            "json_path": str(
                metadata_json_path
            ),
        }
    )

    metadata_json_path.write_text(
        json.dumps(
            stored_record,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    new_row = pd.DataFrame(
        [stored_record]
    )

    if history_csv_path.exists():
        try:
            history_df = pd.read_csv(
                history_csv_path
            )
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            history_df = pd.DataFrame()

        if (
            not history_df.empty
            and "incident_id"
            in history_df.columns
        ):
            history_df = history_df[
                history_df[
                    "incident_id"
                ].astype(str)
                != incident_id
            ]

        history_df = pd.concat(
            [
                history_df,
                new_row,
            ],
            ignore_index=True,
        )
    else:
        history_df = new_row

    history_df.to_csv(
        history_csv_path,
        index=False,
    )

    # Hard verification: fail immediately if any file was not created.
    required_paths = {
        "history_folder": history_folder,
        "history_csv": history_csv_path,
        "pdf": pdf_path,
        "simulation_csv": simulation_csv_path,
        "metadata_json": metadata_json_path,
    }

    missing_paths = [
        str(path)
        for path in required_paths.values()
        if not path.exists()
    ]

    if missing_paths:
        raise OSError(
            "Incident saving did not complete. Missing: "
            + ", ".join(missing_paths)
        )

    return required_paths
