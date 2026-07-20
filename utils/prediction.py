from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def _feature_order(feature_list):
    return tuple(feature_list)


def prepare_model_input(row, feature_list):
    ordered = _feature_order(tuple(feature_list))
    values = {feature: row.get(feature, np.nan) for feature in ordered}
    return pd.DataFrame([values], columns=ordered)


def predict_next_state(
    row,
    voltage_model,
    temperature_model,
    feature_list,
):
    model_input = prepare_model_input(row, feature_list)

    predicted_voltage = float(
        voltage_model.predict(model_input)[0]
    )

    predicted_temperature = float(
        temperature_model.predict(model_input)[0]
    )

    return predicted_voltage, predicted_temperature


def build_rul_features(
    cycle_df,
    cycle_column,
    capacity_column,
    temperature_column,
    voltage_column,
    current_column,
    time_column,
    rul_feature_columns,
):
    feature_values = {
        "capacity_Ah": float(pd.to_numeric(
            cycle_df[capacity_column], errors="coerce"
        ).median()),
        "sample_count": int(len(cycle_df)),
    }

    if temperature_column:
        t = pd.to_numeric(cycle_df[temperature_column], errors="coerce")
        feature_values.update({
            "mean_temperature_C": float(t.mean()),
            "max_temperature_C": float(t.max()),
            "temperature_rise_C": float(t.iloc[-1] - t.iloc[0]),
        })

    if voltage_column:
        v = pd.to_numeric(cycle_df[voltage_column], errors="coerce")
        feature_values.update({
            "mean_voltage_V": float(v.mean()),
            "min_voltage_V": float(v.min()),
            "voltage_drop_V": float(v.iloc[0] - v.iloc[-1]),
        })

    if current_column:
        c = pd.to_numeric(cycle_df[current_column], errors="coerce")
        feature_values.update({
            "mean_current_A": float(c.mean()),
            "current_std_A": float(c.std()),
        })

    if time_column:
        tm = pd.to_numeric(cycle_df[time_column], errors="coerce")
        feature_values["discharge_duration_s"] = float(
            tm.max() - tm.min()
        )

    ordered = _feature_order(tuple(rul_feature_columns))

    return pd.DataFrame(
        [{f: feature_values.get(f, np.nan) for f in ordered}],
        columns=ordered,
    )


def predict_rul(
    cycle_df,
    cycle_column,
    capacity_column,
    temperature_column,
    voltage_column,
    current_column,
    time_column,
    rul_model,
    rul_feature_columns,
):
    model_input = build_rul_features(
        cycle_df,
        cycle_column,
        capacity_column,
        temperature_column,
        voltage_column,
        current_column,
        time_column,
        rul_feature_columns,
    )

    prediction = float(rul_model.predict(model_input)[0])
    return max(0.0, prediction)
