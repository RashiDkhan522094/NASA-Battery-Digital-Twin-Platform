from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MIN_VALID_CAPACITY_AH = 0.05
MAX_VALID_CAPACITY_AH = 5.0
REFERENCE_WINDOW_CYCLES = 20
LATEST_SMOOTHING_CYCLES = 3


def find_column(
    dataframe: pd.DataFrame,
    possible_names: Iterable[str],
) -> str | None:
    for name in possible_names:
        if name in dataframe.columns:
            return name
    return None


def list_battery_files(
    processed_folder: Path,
) -> dict[str, Path]:
    files: dict[str, Path] = {}

    for path in sorted(
        processed_folder.glob("*_realtime_supervised.csv")
    ):
        battery_id = path.stem.replace(
            "_realtime_supervised",
            "",
        )
        files[battery_id] = path

    return files


def detect_columns(
    dataframe: pd.DataFrame,
) -> dict[str, str | None]:
    return {
        "cycle": find_column(
            dataframe,
            [
                "discharge_cycle",
                "cycle",
                "cycle_number",
                "operation_index",
            ],
        ),
        "voltage": find_column(
            dataframe,
            [
                "voltage_measured_V",
                "voltage_V",
                "Voltage_measured",
                "Voltage",
            ],
        ),
        "temperature": find_column(
            dataframe,
            [
                "temperature_measured_C",
                "temperature_C",
                "Temperature_measured",
                "Temperature",
            ],
        ),
        "capacity": find_column(
            dataframe,
            [
                "capacity_Ah",
                "capacity",
                "Capacity",
            ],
        ),
    }


def clean_cycle_capacity(
    dataframe: pd.DataFrame,
    cycle_column: str,
    capacity_column: str,
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "cycle": pd.to_numeric(
                dataframe[cycle_column],
                errors="coerce",
            ),
            "capacity_Ah": pd.to_numeric(
                dataframe[capacity_column],
                errors="coerce",
            ),
        }
    ).dropna()

    table = table[
        np.isfinite(table["capacity_Ah"])
        & (table["capacity_Ah"] >= MIN_VALID_CAPACITY_AH)
        & (table["capacity_Ah"] <= MAX_VALID_CAPACITY_AH)
    ]

    return (
        table.groupby(
            "cycle",
            as_index=False,
        )["capacity_Ah"]
        .median()
        .sort_values("cycle")
        .reset_index(drop=True)
    )


def reference_capacity(
    cycle_capacity_df: pd.DataFrame,
    capacity_column: str = "capacity_Ah",
) -> float:
    if cycle_capacity_df.empty:
        raise ValueError(
            "No valid cycle-capacity data are available."
        )

    reference_window = cycle_capacity_df.head(
        min(
            REFERENCE_WINDOW_CYCLES,
            len(cycle_capacity_df),
        )
    )

    values = pd.to_numeric(
        reference_window[capacity_column],
        errors="coerce",
    ).dropna()

    values = values[
        (values >= MIN_VALID_CAPACITY_AH)
        & (values <= MAX_VALID_CAPACITY_AH)
    ]

    if values.empty:
        raise ValueError(
            "No valid early-life capacity values were found."
        )

    return float(
        values.quantile(0.95)
    )


def robust_latest_capacity(
    cycle_capacity_df: pd.DataFrame,
) -> tuple[float, float]:
    latest_cycle = float(
        cycle_capacity_df["cycle"].iloc[-1]
    )

    latest_capacity = float(
        cycle_capacity_df.tail(
            min(
                LATEST_SMOOTHING_CYCLES,
                len(cycle_capacity_df),
            )
        )["capacity_Ah"].median()
    )

    return latest_cycle, latest_capacity


def battery_summary(
    battery_id: str,
    file_path: Path,
) -> dict:
    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(
            f"Empty battery dataset: {battery_id}"
        )

    columns = detect_columns(dataframe)

    required = [
        "cycle",
        "voltage",
        "temperature",
        "capacity",
    ]

    missing = [
        key
        for key in required
        if columns.get(key) is None
    ]

    if missing:
        raise KeyError(
            f"{battery_id}: missing columns "
            + ", ".join(missing)
        )

    cycle_column = columns["cycle"]
    voltage_column = columns["voltage"]
    temperature_column = columns["temperature"]
    capacity_column = columns["capacity"]

    cycle_capacity = clean_cycle_capacity(
        dataframe=dataframe,
        cycle_column=cycle_column,
        capacity_column=capacity_column,
    )

    if cycle_capacity.empty:
        raise ValueError(
            f"{battery_id}: no physically valid capacity history."
        )

    reference = reference_capacity(
        cycle_capacity,
        "capacity_Ah",
    )

    latest_cycle, latest_capacity = (
        robust_latest_capacity(
            cycle_capacity
        )
    )

    soh_raw = (
        100.0
        * latest_capacity
        / reference
    )

    soh = float(
        np.clip(
            soh_raw,
            0.0,
            105.0,
        )
    )

    latest_cycle_df = dataframe[
        pd.to_numeric(
            dataframe[cycle_column],
            errors="coerce",
        )
        == latest_cycle
    ]

    voltage = float(
        pd.to_numeric(
            latest_cycle_df[voltage_column],
            errors="coerce",
        ).median()
    )

    temperature = float(
        pd.to_numeric(
            latest_cycle_df[temperature_column],
            errors="coerce",
        ).median()
    )

    if soh < 70:
        status = "Critical"
    elif soh < 80:
        status = "High Risk"
    elif soh < 90:
        status = "Warning"
    else:
        status = "Healthy"

    return {
        "battery_id": battery_id,
        "latest_cycle": latest_cycle,
        "capacity_Ah": latest_capacity,
        "reference_capacity_Ah": reference,
        "soh_percent": soh,
        "median_voltage_V": voltage,
        "median_temperature_C": temperature,
        "status": status,
        "valid_capacity_cycles": int(
            len(cycle_capacity)
        ),
        "soh_quality_flag": (
            "Check reference"
            if soh_raw > 105.0
            else "Valid"
        ),
    }


def build_executive_snapshot(
    battery_files: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    failures = []

    for battery_id, file_path in battery_files.items():
        try:
            rows.append(
                battery_summary(
                    battery_id=battery_id,
                    file_path=file_path,
                )
            )
        except Exception as error:
            failures.append(
                {
                    "battery_id": battery_id,
                    "error": str(error),
                }
            )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(failures),
    )


def fleet_score(
    snapshot_df: pd.DataFrame,
    incident_df: pd.DataFrame,
) -> float:
    if snapshot_df.empty:
        return 0.0

    average_soh = float(
        snapshot_df["soh_percent"].mean()
    )

    critical_fraction = float(
        (
            snapshot_df["status"]
            == "Critical"
        ).mean()
    )

    high_risk_fraction = float(
        (
            snapshot_df["status"]
            == "High Risk"
        ).mean()
    )

    warning_fraction = float(
        (
            snapshot_df["status"]
            == "Warning"
        ).mean()
    )

    incident_penalty = 0.0

    if not incident_df.empty:
        incident_penalty = min(
            10.0,
            0.5 * len(incident_df),
        )

    score = (
        0.75 * average_soh
        + 25.0
        - 30.0 * critical_fraction
        - 15.0 * high_risk_fraction
        - 7.5 * warning_fraction
        - incident_penalty
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


def estimated_rul_from_soh(
    soh_percent: float,
    latest_cycle: float,
) -> float:
    if soh_percent <= 80:
        return 0.0

    degradation_per_cycle = max(
        (100.0 - soh_percent)
        / max(latest_cycle, 1.0),
        0.02,
    )

    return float(
        np.clip(
            (soh_percent - 80.0)
            / degradation_per_cycle,
            0.0,
            500.0,
        )
    )
