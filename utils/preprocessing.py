from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


COLUMN_CANDIDATES = {
    "cycle": [
        "discharge_cycle",
        "cycle",
        "cycle_number",
        "operation_index",
    ],
    "time": [
        "time_s",
        "Time_s",
        "time",
    ],
    "voltage": [
        "voltage_measured_V",
        "Voltage_measured",
        "voltage_V",
    ],
    "temperature": [
        "temperature_measured_C",
        "Temperature_measured",
        "temperature_C",
    ],
    "current": [
        "current_measured_A",
        "Current_measured",
        "current_A",
    ],
    "capacity": [
        "capacity_Ah",
        "capacity",
        "Capacity",
    ],
}


def _find_column(
    dataframe: pd.DataFrame,
    possible_names: list[str],
) -> str | None:
    for name in possible_names:
        if name in dataframe.columns:
            return name
    return None


@st.cache_data(show_spinner=False)
def preprocess_battery_data(
    file_path: str | Path,
) -> dict[str, Any]:
    """
    Load, validate, clean and summarize one battery dataset.

    Results are cached by file path, so repeated Streamlit reruns do not
    repeat numeric conversion, missing-value removal, cycle extraction,
    or cycle-level capacity aggregation.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Battery dataset was not found:\n{file_path}"
        )

    battery_df = pd.read_csv(
        file_path,
        engine="c",
        low_memory=False,
    )

    columns = {
        role: _find_column(battery_df, candidates)
        for role, candidates in COLUMN_CANDIDATES.items()
    }

    required_roles = [
        "cycle",
        "time",
        "voltage",
        "temperature",
        "capacity",
    ]

    missing_roles = [
        role
        for role in required_roles
        if columns[role] is None
    ]

    if missing_roles:
        raise ValueError(
            "Required columns are missing for: "
            + ", ".join(missing_roles)
            + "\nAvailable columns: "
            + ", ".join(map(str, battery_df.columns))
        )

    numeric_columns = [
        columns["cycle"],
        columns["time"],
        columns["voltage"],
        columns["temperature"],
        columns["capacity"],
    ]

    if columns["current"] is not None:
        numeric_columns.append(columns["current"])

    for column in numeric_columns:
        battery_df[column] = pd.to_numeric(
            battery_df[column],
            errors="coerce",
        )

    battery_df = battery_df.dropna(
        subset=[
            columns["cycle"],
            columns["time"],
            columns["voltage"],
            columns["temperature"],
            columns["capacity"],
        ]
    ).copy()

    available_cycles = sorted(
        battery_df[columns["cycle"]]
        .dropna()
        .unique()
        .tolist()
    )

    cycle_capacity_df = (
        battery_df.groupby(
            columns["cycle"],
            as_index=False,
        )[columns["capacity"]]
        .median()
        .sort_values(columns["cycle"])
        .reset_index(drop=True)
    )

    cycle_capacity_df[columns["capacity"]] = pd.to_numeric(
        cycle_capacity_df[columns["capacity"]],
        errors="coerce",
    )

    cycle_capacity_df = cycle_capacity_df.dropna(
        subset=[columns["capacity"]]
    )

    cycle_capacity_df = cycle_capacity_df[
        cycle_capacity_df[columns["capacity"]] > 0
    ].copy()

    return {
        "battery_df": battery_df,
        "columns": columns,
        "available_cycles": available_cycles,
        "cycle_capacity_df": cycle_capacity_df,
    }
