from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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
    files = {}

    for path in sorted(
        processed_folder.glob(
            "*_realtime_supervised.csv"
        )
    ):
        battery_id = path.stem.replace(
            "_realtime_supervised",
            "",
        )

        files[battery_id] = path

    return files


def load_battery_dataset(
    file_path: Path,
) -> pd.DataFrame:
    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(
            f"Battery dataset is empty: {file_path}"
        )

    return dataframe


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
            [
                "time_s",
                "Time_s",
                "time",
                "Time",
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
        "next_voltage": find_column(
            dataframe,
            [
                "next_voltage_V",
                "next_voltage",
            ],
        ),
        "next_temperature": find_column(
            dataframe,
            [
                "next_temperature_C",
                "next_temperature",
            ],
        ),
    }


def validate_required_columns(
    columns: dict[str, str | None],
) -> None:
    required = [
        "cycle",
        "time",
        "voltage",
        "temperature",
        "current",
        "capacity",
    ]

    missing = [
        name
        for name in required
        if columns.get(name) is None
    ]

    if missing:
        raise KeyError(
            "Required telemetry columns were not found: "
            + ", ".join(missing)
        )


def cycle_capacity_table(
    dataframe: pd.DataFrame,
    cycle_column: str,
    capacity_column: str,
) -> pd.DataFrame:
    table = (
        dataframe.groupby(
            cycle_column,
            as_index=False,
        )[capacity_column]
        .median()
        .sort_values(cycle_column)
        .reset_index(drop=True)
    )

    return table


def reference_capacity(
    cycle_capacity_df: pd.DataFrame,
    capacity_column: str,
) -> float:
    reference_window = cycle_capacity_df.head(
        min(20, len(cycle_capacity_df))
    )

    reference = float(
        reference_window[
            capacity_column
        ].max()
    )

    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(
            "Reference capacity is invalid."
        )

    return reference


def cycle_soh_percent(
    cycle_capacity: float,
    reference_capacity_value: float,
) -> float:
    soh = (
        100.0
        * float(cycle_capacity)
        / float(reference_capacity_value)
    )

    return float(
        np.clip(
            soh,
            0.0,
            105.0,
        )
    )


def classify_operating_state(
    voltage: float,
    temperature: float,
    soh: float,
) -> tuple[str, str]:
    if (
        voltage < 3.0
        or temperature >= 50.0
        or soh < 70.0
    ):
        return "CRITICAL", "Immediate intervention required"

    if (
        voltage < 3.2
        or temperature >= 45.0
        or soh < 80.0
    ):
        return "HIGH RISK", "Reduce load and inspect battery"

    if (
        voltage < 3.4
        or temperature >= 40.0
        or soh < 90.0
    ):
        return "WARNING", "Enhanced monitoring recommended"

    return "NOMINAL", "Battery operating normally"


def mission_confidence(
    voltage_error: float,
    temperature_error: float,
) -> float:
    score = (
        97.0
        - 18.0 * abs(float(voltage_error))
        - 1.7 * abs(float(temperature_error))
    )

    return float(
        np.clip(
            score,
            55.0,
            99.0,
        )
    )
