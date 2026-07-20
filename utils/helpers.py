def battery_status(
    voltage,
    temperature,
    capacity
):
    if voltage >= 3.5:
        voltage_status = "Normal"
    elif voltage >= 3.0:
        voltage_status = "Warning"
    else:
        voltage_status = "Critical"

    if temperature < 40:
        temperature_status = "Normal"
    elif temperature < 50:
        temperature_status = "Warning"
    else:
        temperature_status = "Critical"

    if capacity >= 1.4:
        health_status = "Healthy"
    elif capacity >= 1.2:
        health_status = "Degraded"
    else:
        health_status = "Aged"

    return (
        voltage_status,
        temperature_status,
        health_status
    )