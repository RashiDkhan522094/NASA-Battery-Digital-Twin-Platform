from pathlib import Path

import joblib
import streamlit as st


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MODEL_FOLDER = PROJECT_FOLDER / "trained_models"


def _load_joblib_file(filename: str):
    file_path = MODEL_FOLDER / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required model file was not found:\n{file_path}"
        )

    return joblib.load(file_path)


@st.cache_resource(show_spinner=False)
def load_models():
    voltage_model = _load_joblib_file(
        "best_realtime_voltage_model.joblib"
    )

    temperature_model = _load_joblib_file(
        "best_realtime_temperature_model.joblib"
    )

    feature_list = _load_joblib_file(
        "realtime_feature_list.joblib"
    )

    return voltage_model, temperature_model, feature_list


@st.cache_resource(show_spinner=False)
def load_rul_model():
    rul_model = _load_joblib_file(
        "best_rul_model.joblib"
    )

    rul_features = _load_joblib_file(
        "rul_feature_columns.joblib"
    )

    return rul_model, rul_features