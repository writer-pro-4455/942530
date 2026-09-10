from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .data import load_config, load_manifest
from .mqtt_publish import build_payload, publish


def preprocess(path, size):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, size)
    image = tf.cast(image, tf.float32) / 255.0
    return np.expand_dims(image.numpy(), 0)


def set_input(interpreter, detail, x):
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = detail["quantization"]
        x = np.round(x / scale + zero_point)
        info = np.iinfo(dtype)
        x = np.clip(x, info.min, info.max).astype(dtype)
    else:
        x = x.astype(dtype)
    interpreter.set_tensor(detail["index"], x)


def get_output(interpreter, detail):
    y = interpreter.get_tensor(detail["index"])[0]
    if np.issubdtype(detail["dtype"], np.integer):
        scale, zero_point = detail["quantization"]
        y = (y.astype(np.float32) - zero_point) * scale
    return y.astype(np.float32)


def simulated_environment():
    # These values are deliberately labeled synthetic and are only for
    # exercising MQTT/ThingsBoard telemetry transport.
    return {
        "temperature_c": round(random.uniform(22.0, 38.0), 2),
        "humidity_pct": round(random.uniform(55.0, 90.0), 2),
        "soil_moisture_pct": round(random.uniform(20.0, 70.0), 2),
        "illuminance_lux": round(random.uniform(2000.0, 20000.0), 1),
        "battery_v": round(random.uniform(3.1, 4.1), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--tflite-model", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--access-token", required=True)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--simulate-telemetry", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--output", default="results/iot_cycles.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    df = load_manifest(cfg, "test")
    if args.cycles > len(df):
        raise ValueError("Requested more cycles than available test images.")

    sample = df.sample(n=args.cycles, random_state=int(cfg["seed"]), replace=False).reset_index(drop=True)

    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    rows = []
    classes = cfg["class_names"]
    port = args.port or int(cfg["iot"]["mqtt_port"])

    for i, row in sample.iterrows():
        x = preprocess(str(row["path"]), tuple(cfg["image_size"]))
        set_input(interpreter, input_detail, x)

        t0 = time.perf_counter()
        interpreter.invoke()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = get_output(interpreter, output_detail)
        pred_idx = int(np.argmax(probs))
        prediction = classes[pred_idx]
        confidence = float(probs[pred_idx])

        sim = simulated_environment() if args.simulate_telemetry else None
        payload = build_payload(
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            simulated_telemetry=sim,
        )

        publish(
            host=args.host,
            access_token=args.access_token,
            payload=payload,
            port=port,
            topic=cfg["iot"]["telemetry_topic"],
            qos=1,
        )

        rows.append({
            "cycle": i + 1,
            "image_path": row["path"],
            "true_class": row["class_name"],
            "predicted_class": prediction,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "correct": prediction == row["class_name"],
            "alert_level": payload["alert_level"],
            "simulated_environmental_telemetry": bool(args.simulate_telemetry),
        })
        print(json.dumps(payload))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved cycle log to {out}")


if __name__ == "__main__":
    main()
