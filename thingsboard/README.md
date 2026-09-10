# ThingsBoard integration

The manuscript used MQTT telemetry sent to a ThingsBoard Community Edition instance.

## MQTT settings reproduced in this repository

- Topic: `v1/devices/me/telemetry`
- Default port in the manuscript: `1883`
- Client: `paho-mqtt`
- QoS used by the manuscript communication test: `1`
- Device authentication: ThingsBoard device access token

## Suggested telemetry keys

Classifier outputs:

- `disease_prediction`
- `confidence`
- `alert_level`
- `inference_latency_ms`

Optional synthetic communication-test keys:

- `temperature_c`
- `humidity_pct`
- `soil_moisture_pct`
- `illuminance_lux`
- `battery_v`
- `environmental_telemetry_status`

The last group must **not** be interpreted as measured field-sensor data unless the authors replace the simulation with physical sensors and document the acquisition process.

## Dashboard

`dashboard_config.example.json` describes the widgets and keys needed for a manuscript-style dashboard. It is not represented as an exported ThingsBoard dashboard because the manuscript does not contain the original export JSON.

Before resubmission:

1. Open the actual ThingsBoard CE instance used in the study.
2. Export the dashboard JSON.
3. Export any rule chain used for alerts.
4. Replace the example file(s) here with those original exports.
5. Commit them to the public repository.
6. Update this README with the exact ThingsBoard CE version and import steps.

This avoids presenting a fabricated platform export as an experimental artifact.
