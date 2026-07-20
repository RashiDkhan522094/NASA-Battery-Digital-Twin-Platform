from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st


# =========================================================
# Project path
# =========================================================
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))


from utils.data_loader import (
    get_battery_files,
    load_battery_data,
)

from utils.model_loader import load_models


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Explainable AI",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# Styling
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(124, 58, 237, 0.10),
                transparent 30%
            ),
            #08111f;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 16px;
        padding: 15px;
    }

    div[data-testid="stPlotlyChart"] {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Helper functions
# =========================================================
def find_column(dataframe, possible_names):
    for name in possible_names:
        if name in dataframe.columns:
            return name

    return None


def create_shap_explanation(
    model,
    background_data,
    sample_data,
    feature_names,
):
    """
    Create a model-agnostic SHAP explanation.

    This works with scikit-learn pipelines containing
    imputers, scalers and regression models.
    """

    explainer = shap.Explainer(
        model.predict,
        background_data,
        feature_names=feature_names,
        algorithm="permutation",
    )

    maximum_evaluations = max(
        2 * len(feature_names) + 1,
        50,
    )

    explanation = explainer(
        sample_data,
        max_evals=maximum_evaluations,
    )

    return explanation


def prepare_explanation_table(
    explanation,
    sample_data,
    feature_names,
):
    shap_values = np.asarray(
        explanation.values
    ).reshape(-1)

    feature_values = (
        sample_data.iloc[0]
        .reindex(feature_names)
        .to_numpy()
    )

    result = pd.DataFrame(
        {
            "Feature": feature_names,
            "Feature_Value": feature_values,
            "SHAP_Value": shap_values,
        }
    )

    result["Absolute_Impact"] = (
        result["SHAP_Value"].abs()
    )

    result["Direction"] = np.where(
        result["SHAP_Value"] >= 0,
        "Increases prediction",
        "Decreases prediction",
    )

    total_impact = result[
        "Absolute_Impact"
    ].sum()

    if total_impact > 0:
        result["Impact_Percent"] = (
            result["Absolute_Impact"]
            / total_impact
            * 100
        )
    else:
        result["Impact_Percent"] = 0.0

    return result.sort_values(
        "Absolute_Impact",
        ascending=False,
    ).reset_index(drop=True)


def create_impact_chart(
    explanation_table,
    chart_title,
    unit,
    number_of_features=10,
):
    plot_df = (
        explanation_table
        .head(number_of_features)
        .sort_values(
            "SHAP_Value",
            ascending=True,
        )
    )

    bar_colors = np.where(
        plot_df["SHAP_Value"] >= 0,
        "#22c55e",
        "#ef4444",
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=plot_df["SHAP_Value"],
            y=plot_df["Feature"],
            orientation="h",
            marker_color=bar_colors,
            customdata=np.column_stack(
                [
                    plot_df["Feature_Value"],
                    plot_df["Impact_Percent"],
                    plot_df["Direction"],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "SHAP impact: %{x:.6f} "
                + unit
                + "<br>"
                "Feature value: %{customdata[0]:.6f}<br>"
                "Relative importance: "
                "%{customdata[1]:.2f}%<br>"
                "%{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_vline(
        x=0,
        line_dash="dash",
    )

    figure.update_layout(
        title=chart_title,
        xaxis_title=(
            "Contribution to prediction "
            f"({unit})"
        ),
        yaxis_title="Feature",
        height=480,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=40,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "white"
        },
    )

    return figure


def format_feature_value(value):
    try:
        if pd.isna(value):
            return "Missing"

        return f"{float(value):.5f}"

    except Exception:
        return str(value)


# =========================================================
# Header
# =========================================================
st.title("🧠 SHAP Explainable AI")

st.caption(
    "Understand which battery measurements increase or "
    "decrease each AI prediction"
)


# =========================================================
# Load files and models
# =========================================================
battery_files = get_battery_files()

if not battery_files:
    st.error(
        "No processed battery datasets were found."
    )
    st.stop()


try:
    (
        voltage_model,
        temperature_model,
        feature_list,
    ) = load_models()

except Exception as error:
    st.error(
        "The trained models could not be loaded."
    )
    st.exception(error)
    st.stop()


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Explainability Controls")

selected_battery = st.sidebar.selectbox(
    "Select battery",
    options=list(battery_files.keys()),
)

battery_df = load_battery_data(
    battery_files[selected_battery]
).copy()


# =========================================================
# Detect columns
# =========================================================
cycle_column = find_column(
    battery_df,
    [
        "discharge_cycle",
        "cycle",
        "cycle_number",
        "operation_index",
    ],
)

time_column = find_column(
    battery_df,
    [
        "time_s",
        "Time_s",
        "time",
    ],
)

voltage_column = find_column(
    battery_df,
    [
        "voltage_measured_V",
        "Voltage_measured",
        "voltage_V",
    ],
)

temperature_column = find_column(
    battery_df,
    [
        "temperature_measured_C",
        "Temperature_measured",
        "temperature_C",
    ],
)


required_columns = {
    "cycle": cycle_column,
    "time": time_column,
    "voltage": voltage_column,
    "temperature": temperature_column,
}

missing_columns = [
    label
    for label, column in required_columns.items()
    if column is None
]

if missing_columns:
    st.error(
        "Required columns are missing: "
        + ", ".join(missing_columns)
    )

    st.write("Available columns:")
    st.write(list(battery_df.columns))
    st.stop()


missing_features = [
    feature
    for feature in feature_list
    if feature not in battery_df.columns
]

if missing_features:
    st.error(
        "The following model features are missing "
        "from this battery dataset:"
    )

    st.write(missing_features)
    st.stop()


# =========================================================
# Select cycle
# =========================================================
cycle_counts = (
    battery_df
    .groupby(cycle_column)
    .size()
)

valid_cycles = sorted(
    cycle_counts[
        cycle_counts >= 2
    ].index.tolist()
)

if not valid_cycles:
    st.error(
        "No cycles contain enough rows for explanation."
    )
    st.stop()


selected_cycle = st.sidebar.selectbox(
    "Select discharge cycle",
    options=valid_cycles,
)

cycle_df = (
    battery_df[
        battery_df[cycle_column] == selected_cycle
    ]
    .copy()
    .sort_values(time_column)
    .reset_index(drop=True)
)


# =========================================================
# Select sample
# =========================================================
selected_index = st.sidebar.slider(
    "Select mission time step",
    min_value=0,
    max_value=len(cycle_df) - 1,
    value=min(20, len(cycle_df) - 1),
    step=1,
)

top_feature_count = st.sidebar.slider(
    "Features to display",
    min_value=5,
    max_value=min(15, len(feature_list)),
    value=min(10, len(feature_list)),
    step=1,
)


current_row = cycle_df.iloc[selected_index]

current_time = float(
    current_row[time_column]
)

current_voltage = float(
    current_row[voltage_column]
)

current_temperature = float(
    current_row[temperature_column]
)


# =========================================================
# Prepare model input
# =========================================================
sample_input = pd.DataFrame(
    [
        {
            feature: current_row.get(
                feature,
                np.nan,
            )
            for feature in feature_list
        }
    ]
)

sample_input = sample_input.replace(
    [np.inf, -np.inf],
    np.nan,
)


# =========================================================
# Create representative background dataset
# =========================================================
background_size = min(
    50,
    len(battery_df),
)

background_data = (
    battery_df[feature_list]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .sample(
        n=background_size,
        random_state=42,
    )
    .reset_index(drop=True)
)


# =========================================================
# Predictions
# =========================================================
try:
    predicted_voltage = float(
        voltage_model.predict(
            sample_input
        )[0]
    )

    predicted_temperature = float(
        temperature_model.predict(
            sample_input
        )[0]
    )

except Exception as error:
    st.error(
        "The models could not generate predictions."
    )
    st.exception(error)
    st.stop()


# =========================================================
# Current-state summary
# =========================================================
st.subheader(
    f"Battery {selected_battery} — "
    f"Cycle {selected_cycle}, "
    f"Time {current_time:.1f} s"
)

summary_1, summary_2, summary_3, summary_4 = st.columns(4)

with summary_1:
    st.metric(
        "Measured Voltage",
        f"{current_voltage:.4f} V",
    )

with summary_2:
    st.metric(
        "Predicted Next Voltage",
        f"{predicted_voltage:.4f} V",
    )

with summary_3:
    st.metric(
        "Measured Temperature",
        f"{current_temperature:.2f} °C",
    )

with summary_4:
    st.metric(
        "Predicted Next Temperature",
        f"{predicted_temperature:.2f} °C",
    )


# =========================================================
# Generate SHAP explanations
# =========================================================
with st.spinner(
    "Calculating SHAP explanations..."
):
    try:
        voltage_explanation = (
            create_shap_explanation(
                voltage_model,
                background_data,
                sample_input,
                feature_list,
            )
        )

        temperature_explanation = (
            create_shap_explanation(
                temperature_model,
                background_data,
                sample_input,
                feature_list,
            )
        )

    except Exception as error:
        st.error(
            "SHAP could not explain the predictions."
        )
        st.exception(error)
        st.stop()


# =========================================================
# Prepare explanation tables
# =========================================================
voltage_table = prepare_explanation_table(
    voltage_explanation,
    sample_input,
    feature_list,
)

temperature_table = prepare_explanation_table(
    temperature_explanation,
    sample_input,
    feature_list,
)


# =========================================================
# Voltage explanation
# =========================================================
st.subheader("Voltage Prediction Explanation")

st.info(
    "Green bars increase the predicted voltage. "
    "Red bars decrease the predicted voltage."
)

voltage_chart = create_impact_chart(
    voltage_table,
    "Top Influences on Next-Voltage Prediction",
    "V",
    top_feature_count,
)

st.plotly_chart(
    voltage_chart,
    use_container_width=True,
)


# =========================================================
# Temperature explanation
# =========================================================
st.subheader("Temperature Prediction Explanation")

st.info(
    "Green bars increase the predicted temperature. "
    "Red bars decrease the predicted temperature."
)

temperature_chart = create_impact_chart(
    temperature_table,
    "Top Influences on Next-Temperature Prediction",
    "°C",
    top_feature_count,
)

st.plotly_chart(
    temperature_chart,
    use_container_width=True,
)


# =========================================================
# Top-feature interpretation
# =========================================================
st.subheader("AI Interpretation")

interpretation_1, interpretation_2 = st.columns(2)


with interpretation_1:
    st.markdown("#### Voltage Model")

    top_voltage_feature = (
        voltage_table.iloc[0]
    )

    voltage_direction = (
        "upward"
        if top_voltage_feature["SHAP_Value"] >= 0
        else "downward"
    )

    st.write(
        f"The most influential voltage feature is "
        f"**{top_voltage_feature['Feature']}**."
    )

    st.write(
        f"Its current value is "
        f"**{format_feature_value(top_voltage_feature['Feature_Value'])}** "
        f"and it pushes the prediction "
        f"**{voltage_direction}** by approximately "
        f"**{abs(top_voltage_feature['SHAP_Value']):.6f} V**."
    )

    st.write(
        f"It accounts for approximately "
        f"**{top_voltage_feature['Impact_Percent']:.2f}%** "
        f"of the total local explanation."
    )


with interpretation_2:
    st.markdown("#### Temperature Model")

    top_temperature_feature = (
        temperature_table.iloc[0]
    )

    temperature_direction = (
        "upward"
        if top_temperature_feature["SHAP_Value"] >= 0
        else "downward"
    )

    st.write(
        f"The most influential temperature feature is "
        f"**{top_temperature_feature['Feature']}**."
    )

    st.write(
        f"Its current value is "
        f"**{format_feature_value(top_temperature_feature['Feature_Value'])}** "
        f"and it pushes the prediction "
        f"**{temperature_direction}** by approximately "
        f"**{abs(top_temperature_feature['SHAP_Value']):.6f} °C**."
    )

    st.write(
        f"It accounts for approximately "
        f"**{top_temperature_feature['Impact_Percent']:.2f}%** "
        f"of the total local explanation."
    )


# =========================================================
# Detailed tables
# =========================================================
with st.expander(
    "View complete voltage explanation table"
):
    voltage_display = voltage_table.copy()

    voltage_display[
        "Feature_Value"
    ] = voltage_display[
        "Feature_Value"
    ].apply(format_feature_value)

    voltage_display[
        "SHAP_Value"
    ] = voltage_display[
        "SHAP_Value"
    ].round(7)

    voltage_display[
        "Impact_Percent"
    ] = voltage_display[
        "Impact_Percent"
    ].round(2)

    st.dataframe(
        voltage_display[
            [
                "Feature",
                "Feature_Value",
                "SHAP_Value",
                "Impact_Percent",
                "Direction",
            ]
        ],
        use_container_width=True,
    )


with st.expander(
    "View complete temperature explanation table"
):
    temperature_display = temperature_table.copy()

    temperature_display[
        "Feature_Value"
    ] = temperature_display[
        "Feature_Value"
    ].apply(format_feature_value)

    temperature_display[
        "SHAP_Value"
    ] = temperature_display[
        "SHAP_Value"
    ].round(7)

    temperature_display[
        "Impact_Percent"
    ] = temperature_display[
        "Impact_Percent"
    ].round(2)

    st.dataframe(
        temperature_display[
            [
                "Feature",
                "Feature_Value",
                "SHAP_Value",
                "Impact_Percent",
                "Direction",
            ]
        ],
        use_container_width=True,
    )


# =========================================================
# Download explanation results
# =========================================================
voltage_download = voltage_table.copy()
voltage_download["Prediction_Target"] = (
    "Next Voltage"
)

temperature_download = temperature_table.copy()
temperature_download["Prediction_Target"] = (
    "Next Temperature"
)

combined_download = pd.concat(
    [
        voltage_download,
        temperature_download,
    ],
    ignore_index=True,
)

combined_download["Battery"] = selected_battery
combined_download["Cycle"] = selected_cycle
combined_download["Time_s"] = current_time

shap_csv = combined_download.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download SHAP explanation",
    data=shap_csv,
    file_name=(
        f"{selected_battery}_cycle_"
        f"{selected_cycle}_time_"
        f"{selected_index}_shap.csv"
    ),
    mime="text/csv",
)