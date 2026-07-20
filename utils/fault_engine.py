from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FaultConfig:
    name: str
    description: str
    default_severity: float


FAULT_LIBRARY: Dict[str, FaultConfig] = {
    "Normal": FaultConfig(
        name="Normal",
        description="No injected fault. Baseline operating condition.",
        default_severity=0.0,
    ),
    "Over-current": FaultConfig(
        name="Over-current",
        description="Higher current demand causes stronger voltage sag and heating.",
        default_severity=0.40,
    ),
    "Over-temperature": FaultConfig(
        name="Over-temperature",
        description="Elevated thermal condition accelerates degradation.",
        default_severity=0.40,
    ),
    "Cooling failure": FaultConfig(
        name="Cooling failure",
        description="Reduced cooling causes progressive temperature accumulation.",
        default_severity=0.50,
    ),
    "Deep discharge": FaultConfig(
        name="Deep discharge",
        description="Lower voltage operation increases degradation risk.",
        default_severity=0.50,
    ),
    "Fast charging": FaultConfig(
        name="Fast charging",
        description="High charging current increases thermal and ageing stress.",
        default_severity=0.40,
    ),
    "Internal resistance increase": FaultConfig(
        name="Internal resistance increase",
        description="Additional internal resistance increases voltage loss and heat generation.",
        default_severity=0.40,
    ),
    "Sensor drift": FaultConfig(
        name="Sensor drift",
        description="Displayed telemetry gradually deviates from the simulated physical state.",
        default_severity=0.30,
    ),
    "Capacity fade": FaultConfig(
        name="Capacity fade",
        description="Available capacity and SOH decline more rapidly.",
        default_severity=0.40,
    ),
}


def available_faults() -> list[str]:
    return list(FAULT_LIBRARY.keys())


def validate_severity(severity: float) -> float:
    return float(np.clip(float(severity), 0.0, 1.0))


def inject_fault(
    simulation_df: pd.DataFrame,
    fault_type: str,
    severity: float,
    baseline_current: float,
    baseline_soh: float,
) -> pd.DataFrame:
    """
    Apply transparent, deterministic fault effects to an already generated
    simulation trajectory.

    Required columns:
        step, time_s, voltage_V, temperature_C

    Added columns:
        true_voltage_V, true_temperature_C, displayed_voltage_V,
        displayed_temperature_C, fault_soh_percent, fault_current_A
    """
    if simulation_df.empty:
        raise ValueError("simulation_df is empty.")

    required = {"step", "time_s", "voltage_V", "temperature_C"}
    missing = required.difference(simulation_df.columns)

    if missing:
        raise ValueError(
            "Missing simulation columns: " + ", ".join(sorted(missing))
        )

    severity = validate_severity(severity)

    if fault_type not in FAULT_LIBRARY:
        raise ValueError(f"Unknown fault type: {fault_type}")

    df = simulation_df.copy()

    progress = np.linspace(0.0, 1.0, len(df))
    current_profile = np.full(len(df), float(baseline_current))

    true_voltage = (
        pd.to_numeric(
            df["voltage_V"],
            errors="coerce",
        )
        .to_numpy(dtype=float)
        .copy()
    )

    true_temperature = (
        pd.to_numeric(
            df["temperature_C"],
            errors="coerce",
        )
        .to_numpy(dtype=float)
        .copy()
    )

    displayed_voltage = np.array(
        true_voltage,
        dtype=float,
        copy=True,
    )

    displayed_temperature = np.array(
        true_temperature,
        dtype=float,
        copy=True,
    )

    soh_penalty = np.zeros(len(df), dtype=float)

    if fault_type == "Over-current":
        current_profile = baseline_current * (1.0 + 1.50 * severity)
        true_voltage -= (0.10 + 0.35 * progress) * severity
        true_temperature += (2.0 + 11.0 * progress) * severity
        soh_penalty += 4.0 * severity * progress

    elif fault_type == "Over-temperature":
        true_temperature += (6.0 + 18.0 * progress) * severity
        true_voltage -= 0.10 * severity * progress
        soh_penalty += 5.0 * severity * progress

    elif fault_type == "Cooling failure":
        true_temperature += 22.0 * severity * np.power(progress, 1.35)
        true_voltage -= 0.12 * severity * progress
        soh_penalty += 6.0 * severity * progress

    elif fault_type == "Deep discharge":
        true_voltage -= (0.18 + 0.55 * progress) * severity
        true_temperature += 4.0 * severity * progress
        soh_penalty += 7.0 * severity * progress

    elif fault_type == "Fast charging":
        current_profile = abs(baseline_current) * (1.0 + 1.80 * severity)
        true_voltage += 0.08 * severity * (1.0 - progress)
        true_temperature += (3.0 + 15.0 * progress) * severity
        soh_penalty += 6.0 * severity * progress

    elif fault_type == "Internal resistance increase":
        resistance_factor = 0.12 + 0.55 * progress
        true_voltage -= resistance_factor * severity
        true_temperature += 13.0 * severity * progress
        soh_penalty += 6.5 * severity * progress

    elif fault_type == "Sensor drift":
        displayed_voltage += 0.22 * severity * progress
        displayed_temperature += 9.0 * severity * progress
        soh_penalty += 0.8 * severity * progress

    elif fault_type == "Capacity fade":
        true_voltage -= 0.18 * severity * progress
        soh_penalty += 14.0 * severity * progress

    true_voltage = np.clip(true_voltage, 2.0, 4.5)
    true_temperature = np.clip(true_temperature, -5.0, 90.0)
    displayed_voltage = np.clip(displayed_voltage, 2.0, 4.5)
    displayed_temperature = np.clip(displayed_temperature, -5.0, 90.0)

    fault_soh = np.clip(
        float(baseline_soh) - soh_penalty,
        0.0,
        100.0,
    )

    df["true_voltage_V"] = true_voltage
    df["true_temperature_C"] = true_temperature
    df["displayed_voltage_V"] = displayed_voltage
    df["displayed_temperature_C"] = displayed_temperature
    df["fault_soh_percent"] = fault_soh
    df["fault_current_A"] = current_profile
    df["fault_type"] = fault_type
    df["fault_severity"] = severity

    return df


def calculate_fault_risk(
    final_voltage: float,
    max_temperature: float,
    final_soh: float,
) -> dict:
    score = 0

    if final_voltage < 3.0:
        score += 45
    elif final_voltage < 3.2:
        score += 25
    elif final_voltage < 3.4:
        score += 10

    if max_temperature >= 55:
        score += 45
    elif max_temperature >= 45:
        score += 30
    elif max_temperature >= 40:
        score += 15

    if final_soh <= 70:
        score += 40
    elif final_soh <= 80:
        score += 25
    elif final_soh < 90:
        score += 10

    score = int(np.clip(score, 0, 100))

    if score >= 70:
        level = "Critical"
    elif score >= 45:
        level = "High"
    elif score >= 20:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "risk_score": score,
        "risk_level": level,
    }
