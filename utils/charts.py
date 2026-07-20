from __future__ import annotations

import plotly.graph_objects as go


def _base_layout(fig, title, x_title, y_title):
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "size": 14},
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def voltage_chart(cycle_df, current_time, predicted_voltage, actual_next_voltage):
    next_row = cycle_df.loc[cycle_df["time_s"] > current_time].head(1)

    if next_row.empty:
        next_time = current_time
    else:
        next_time = float(next_row["time_s"].iloc[0])

    current_voltage = float(
        cycle_df.loc[
            cycle_df["time_s"] == current_time,
            "voltage_measured_V",
        ].iloc[0]
    )

    fig = go.Figure()

    fig.add_scatter(
        x=cycle_df["time_s"],
        y=cycle_df["voltage_measured_V"],
        mode="lines",
        name="Measured voltage",
    )

    fig.add_scatter(
        x=[current_time, next_time],
        y=[current_voltage, predicted_voltage],
        mode="lines+markers",
        name="AI prediction",
        line=dict(dash="dash"),
    )

    fig.add_scatter(
        x=[next_time],
        y=[actual_next_voltage],
        mode="markers",
        name="Actual next voltage",
    )

    fig.add_vline(x=current_time, line_dash="dot")

    return _base_layout(
        fig,
        "Voltage Monitoring",
        "Time (s)",
        "Voltage (V)",
    )


def temperature_chart(
    cycle_df,
    current_time,
    predicted_temperature,
    actual_next_temperature,
):
    next_row = cycle_df.loc[cycle_df["time_s"] > current_time].head(1)

    if next_row.empty:
        next_time = current_time
    else:
        next_time = float(next_row["time_s"].iloc[0])

    current_temperature = float(
        cycle_df.loc[
            cycle_df["time_s"] == current_time,
            "temperature_measured_C",
        ].iloc[0]
    )

    fig = go.Figure()

    fig.add_scatter(
        x=cycle_df["time_s"],
        y=cycle_df["temperature_measured_C"],
        mode="lines",
        name="Measured temperature",
    )

    fig.add_scatter(
        x=[current_time, next_time],
        y=[current_temperature, predicted_temperature],
        mode="lines+markers",
        name="AI prediction",
        line=dict(dash="dash"),
    )

    fig.add_scatter(
        x=[next_time],
        y=[actual_next_temperature],
        mode="markers",
        name="Actual next temperature",
    )

    fig.add_vline(x=current_time, line_dash="dot")

    return _base_layout(
        fig,
        "Temperature Monitoring",
        "Time (s)",
        "Temperature (°C)",
    )
