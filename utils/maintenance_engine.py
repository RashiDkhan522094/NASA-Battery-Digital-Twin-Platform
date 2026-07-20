from __future__ import annotations

from typing import Dict

import numpy as np


def generate_maintenance_recommendation(
    fault_type: str,
    risk_level: str,
    final_voltage: float,
    max_temperature: float,
    final_soh: float,
    estimated_rul: float | None = None,
) -> Dict[str, str]:
    actions = {
        "Normal": "Continue operation and routine monitoring.",
        "Over-current": "Reduce current demand and inspect load-control settings.",
        "Over-temperature": "Reduce load and inspect the thermal-management system.",
        "Cooling failure": "Restore cooling before continued high-load operation.",
        "Deep discharge": "Increase the low-voltage cutoff and avoid further deep discharge.",
        "Fast charging": "Reduce charging rate and monitor battery temperature.",
        "Internal resistance increase": "Perform impedance testing and inspect cell ageing or connection losses.",
        "Sensor drift": "Calibrate or replace the affected voltage or temperature sensor.",
        "Capacity fade": "Plan battery replacement and reduce high-stress operating conditions.",
    }

    urgency_by_risk = {
        "Low": "Routine",
        "Moderate": "Schedule inspection",
        "High": "Act soon",
        "Critical": "Immediate action",
    }

    reasons = []

    if final_voltage < 3.0:
        reasons.append("voltage entered the critical low-voltage region")
    elif final_voltage < 3.2:
        reasons.append("voltage entered the warning region")

    if max_temperature >= 55:
        reasons.append("temperature reached a critical thermal level")
    elif max_temperature >= 45:
        reasons.append("temperature exceeded the warning threshold")

    if final_soh <= 70:
        reasons.append("battery health is critically degraded")
    elif final_soh <= 80:
        reasons.append("battery health reached the conventional end-of-life region")
    elif final_soh < 90:
        reasons.append("measurable degradation is present")

    if not reasons:
        reasons.append("operating variables remained within the configured limits")

    if estimated_rul is None or not np.isfinite(estimated_rul):
        maintenance_window = "Not available"
    elif estimated_rul <= 10:
        maintenance_window = "Within 5 cycles"
    elif estimated_rul <= 30:
        maintenance_window = "Within 10 cycles"
    elif estimated_rul <= 75:
        maintenance_window = "Within 25 cycles"
    else:
        maintenance_window = "Routine monitoring interval"

    return {
        "fault": fault_type,
        "risk": risk_level,
        "urgency": urgency_by_risk.get(risk_level, "Review"),
        "reason": "; ".join(reasons).capitalize() + ".",
        "recommended_action": actions.get(
            fault_type,
            "Inspect the battery system and continue close monitoring.",
        ),
        "maintenance_window": maintenance_window,
    }
