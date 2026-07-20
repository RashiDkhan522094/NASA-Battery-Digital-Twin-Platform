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
        "time": find_column(
            dataframe,
            ["time_s", "Time_s", "time", "Time"],
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
        "current": find_column(
            dataframe,
            [
                "current_measured_A",
                "current_A",
                "Current_measured",
                "Current",
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
    cycle_numeric = pd.to_numeric(
        dataframe[cycle_column],
        errors="coerce",
    )

    capacity_numeric = pd.to_numeric(
        dataframe[capacity_column],
        errors="coerce",
    )

    table = pd.DataFrame(
        {
            "cycle": cycle_numeric,
            "capacity_Ah": capacity_numeric,
        }
    ).dropna()

    table = table[
        np.isfinite(table["capacity_Ah"])
        & (table["capacity_Ah"] >= MIN_VALID_CAPACITY_AH)
        & (table["capacity_Ah"] <= MAX_VALID_CAPACITY_AH)
    ]

    table = (
        table.groupby(
            "cycle",
            as_index=False,
        )["capacity_Ah"]
        .median()
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    return table


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

    valid_reference_values = pd.to_numeric(
        reference_window[capacity_column],
        errors="coerce",
    ).dropna()

    valid_reference_values = valid_reference_values[
        (valid_reference_values >= MIN_VALID_CAPACITY_AH)
        & (valid_reference_values <= MAX_VALID_CAPACITY_AH)
    ]

    if valid_reference_values.empty:
        raise ValueError(
            "No physically valid reference capacities were found."
        )

    # Robust high-end reference: avoids one abnormal spike while still
    # representing the healthy early-life capacity.
    reference = float(
        valid_reference_values.quantile(0.95)
    )

    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(
            "Reference capacity is invalid."
        )

    return reference


def robust_latest_capacity(
    cycle_capacity_df: pd.DataFrame,
) -> tuple[float, float]:
    if cycle_capacity_df.empty:
        raise ValueError(
            "No valid cycle-capacity records are available."
        )

    latest_window = cycle_capacity_df.tail(
        min(
            LATEST_SMOOTHING_CYCLES,
            len(cycle_capacity_df),
        )
    )

    latest_capacity = float(
        latest_window["capacity_Ah"].median()
    )

    latest_cycle = float(
        cycle_capacity_df["cycle"].iloc[-1]
    )

    return latest_cycle, latest_capacity


def battery_health_snapshot(
    battery_id: str,
    file_path: Path,
) -> dict:
    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(
            f"Empty dataset for {battery_id}."
        )

    columns = detect_columns(dataframe)

    required = [
        "cycle",
        "voltage",
        "temperature",
        "capacity",
    ]

    missing = [
        name
        for name in required
        if columns.get(name) is None
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
    ].copy()

    latest_voltage = float(
        pd.to_numeric(
            latest_cycle_df[voltage_column],
            errors="coerce",
        ).median()
    )

    latest_temperature = float(
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

    invalid_capacity_rows = int(
        (
            pd.to_numeric(
                dataframe[capacity_column],
                errors="coerce",
            ).isna()
            | (
                pd.to_numeric(
                    dataframe[capacity_column],
                    errors="coerce",
                )
                < MIN_VALID_CAPACITY_AH
            )
            | (
                pd.to_numeric(
                    dataframe[capacity_column],
                    errors="coerce",
                )
                > MAX_VALID_CAPACITY_AH
            )
        ).sum()
    )

    return {
        "battery_id": battery_id,
        "latest_cycle": latest_cycle,
        "latest_capacity_Ah": latest_capacity,
        "reference_capacity_Ah": reference,
        "soh_percent": soh,
        "median_voltage_V": latest_voltage,
        "median_temperature_C": latest_temperature,
        "status": status,
        "valid_capacity_cycles": int(
            len(cycle_capacity)
        ),
        "invalid_capacity_rows": invalid_capacity_rows,
        "soh_quality_flag": (
            "Check reference"
            if soh_raw > 105.0
            else "Valid"
        ),
    }


def build_fleet_snapshot(
    battery_files: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    failures = []

    for battery_id, file_path in battery_files.items():
        try:
            rows.append(
                battery_health_snapshot(
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
