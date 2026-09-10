from __future__ import annotations

import argparse
import json
import ssl
import time
from typing import Optional

import paho.mqtt.client as mqtt


def alert_level(prediction: str, confidence: float, threshold: float = 0.80) -> str:
    if prediction.lower() == "healthy":
        return "NONE"
    return "HIGH" if confidence >= threshold else "MEDIUM"


def build_payload(prediction, confidence, latency_ms=None, simulated_telemetry=None):
    payload = {
        "timestamp_unix": time.time(),
        "disease_prediction": prediction,
        "confidence": float(confidence),
        "alert_level": alert_level(prediction, confidence),
    }
    if latency_ms is not None:
        payload["inference_latency_ms"] = float(latency_ms)

    if simulated_telemetry:
        payload.update(simulated_telemetry)
        payload["environmental_telemetry_status"] = "SIMULATED_COMMUNICATION_TEST_ONLY"

    return payload


def publish(
    host: str,
    access_token: str,
    payload: dict,
    port: int = 1883,
    topic: str = "v1/devices/me/telemetry",
    qos: int = 1,
    use_tls: bool = False,
):
    client = mqtt.Client()
    client.username_pw_set(access_token)

    if use_tls:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

    client.connect(host, port, keepalive=60)
    client.loop_start()
    info = client.publish(topic, json.dumps(payload), qos=qos)
    info.wait_for_publish()
    rc = info.rc
    client.loop_stop()
    client.disconnect()

    if rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(f"MQTT publish failed with return code {rc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--access-token", required=True)
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--latency-ms", type=float, default=None)
    parser.add_argument("--tls", action="store_true")
    args = parser.parse_args()

    payload = build_payload(args.prediction, args.confidence, args.latency_ms)
    publish(
        host=args.host,
        access_token=args.access_token,
        payload=payload,
        port=args.port,
        use_tls=args.tls,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
