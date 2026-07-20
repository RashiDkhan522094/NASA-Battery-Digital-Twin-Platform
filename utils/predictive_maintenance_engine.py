from __future__ import annotations

from typing import Any

import numpy as np


def classify_priority(
    risk_level: str,
) -> str:
    mapping = {
        "Low": "Routine",
        "Moderate": "Schedule inspection",
        "High": "Act soon",
        "Critical": "Immediate action",
    }

    return mapping.get(
        str(risk_level),
        "Review required",
    )


def decision_confidence(
    risk_score: float,
    severity_percent: float,
    final_soh: float,
) -> float:
    confidence = (
        68.0
        + 0.18 * float(risk_score)
        + 0.10 * float(severity_percent)
        + 0.08 * max(
            0.0,
            100.0 - float(final_soh),
        )
    )

    return float(
        np.clip(
            confidence,
            70.0,
            97.0,
        )
    )


def safe_operating_window(
    estimated_rul: float,
    risk_level: str,
) -> int:
    factor = {
        "Low": 0.75,
        "Moderate": 0.50,
        "High": 0.25,
        "Critical": 0.05,
    }.get(
        str(risk_level),
        0.40,
    )

    return max(
        0,
        int(
            round(
                float(estimated_rul)
                * factor
            )
        ),
    )
