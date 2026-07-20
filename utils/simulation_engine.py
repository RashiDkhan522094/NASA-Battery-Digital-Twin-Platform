from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass

    return float(default)


def build_recursive_simulation(
    baseline_row: pd.Series,
    voltage_model,
    temperature_model,
    feature_list: Iterable[str],
    predict_next_state,
    time_column: str,
    voltage_column: str,
    temperature_column: str,
    current_column: str | None,
    applied_current: float,
    simulation_steps: int,
    time_step_s: float,
    smoothing: float = 0.35,
    voltage_bounds: tuple[float, float] = (2.0, 4.5),
    temperature_bounds: tuple[float, float] = (-5.0, 70.0),
) -> pd.DataFrame:
    """
    Generate a stabilized recursive next-state simulation.

    The supplied predict_next_state function must accept:
        row, voltage_model, temperature_model, feature_list
    """
    simulation_steps = int(max(1, simulation_steps))
    time_step_s = float(max(0.1, time_step_s))
    smoothing = float(np.clip(smoothing, 0.05, 0.95))

    row = baseline_row.copy()

    current_time = safe_float(row.get(time_column, 0.0))
    current_voltage = safe_float(row.get(voltage_column, 4.0))
    current_temperature = safe_float(
        row.get(temperature_column, 25.0)
    )

    if current_column is not None:
        row[current_column] = applied_current

    records = [
        {
            "step": 0,
            "time_s": current_time,
            "voltage_V": current_voltage,
            "temperature_C": current_temperature,
            "current_A": applied_current,
        }
    ]

    for step_number in range(1, simulation_steps + 1):
        predicted_voltage, predicted_temperature = predict_next_state(
            row,
            voltage_model,
            temperature_model,
            feature_list,
        )

        predicted_voltage = safe_float(
            predicted_voltage,
            current_voltage,
        )

        predicted_temperature = safe_float(
            predicted_temperature,
            current_temperature,
        )

        next_voltage = (
            smoothing * predicted_voltage
            + (1.0 - smoothing) * current_voltage
        )

        next_temperature = (
            smoothing * predicted_temperature
            + (1.0 - smoothing) * current_temperature
        )

        next_voltage = float(
            np.clip(
                next_voltage,
                current_voltage - 0.12,
                current_voltage + 0.12,
            )
        )

        next_temperature = float(
            np.clip(
                next_temperature,
                current_temperature - 2.5,
                current_temperature + 2.5,
            )
        )

        next_voltage = float(
            np.clip(
                next_voltage,
                voltage_bounds[0],
                voltage_bounds[1],
            )
        )

        next_temperature = float(
            np.clip(
                next_temperature,
                temperature_bounds[0],
                temperature_bounds[1],
            )
        )

        next_time = current_time + time_step_s

        records.append(
            {
                "step": step_number,
                "time_s": next_time,
                "voltage_V": next_voltage,
                "temperature_C": next_temperature,
                "current_A": applied_current,
            }
        )

        row[time_column] = next_time
        row[voltage_column] = next_voltage
        row[temperature_column] = next_temperature

        if current_column is not None:
            row[current_column] = applied_current

        aliases = {
            "time_s": next_time,
            "Time_s": next_time,
            "time": next_time,
            "voltage_measured_V": next_voltage,
            "Voltage_measured": next_voltage,
            "voltage_V": next_voltage,
            "temperature_measured_C": next_temperature,
            "Temperature_measured": next_temperature,
            "temperature_C": next_temperature,
            "current_measured_A": applied_current,
            "Current_measured": applied_current,
            "current_A": applied_current,
            "normalized_time": next_time,
            "temperature_rise_C": (
                next_temperature
                - records[0]["temperature_C"]
            ),
            "voltage_drop_from_start_V": (
                records[0]["voltage_V"]
                - next_voltage
            ),
            "voltage_change_V": (
                next_voltage
                - current_voltage
            ),
            "temperature_change_C": (
                next_temperature
                - current_temperature
            ),
        }

        for feature_name, feature_value in aliases.items():
            if feature_name in feature_list:
                row[feature_name] = feature_value

        current_time = next_time
        current_voltage = next_voltage
        current_temperature = next_temperature

    return pd.DataFrame(records)


def data_informed_bounds(
    battery_df: pd.DataFrame,
    voltage_column: str,
    temperature_column: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    voltage_series = pd.to_numeric(
        battery_df[voltage_column],
        errors="coerce",
    ).dropna()

    temperature_series = pd.to_numeric(
        battery_df[temperature_column],
        errors="coerce",
    ).dropna()

    voltage_min = max(
        2.0,
        safe_float(voltage_series.quantile(0.01), 2.5) - 0.15,
    )

    voltage_max = min(
        4.5,
        safe_float(voltage_series.quantile(0.99), 4.2) + 0.15,
    )

    temperature_min = max(
        -5.0,
        safe_float(temperature_series.quantile(0.01), 0.0) - 3.0,
    )

    temperature_max = min(
        70.0,
        safe_float(temperature_series.quantile(0.99), 50.0) + 5.0,
    )

    return (
        (voltage_min, voltage_max),
        (temperature_min, temperature_max),
    )
