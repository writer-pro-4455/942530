from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import load_config, load_manifest


def load_first_image(config):
    df = load_manifest(config, "test")
    path = str(df.iloc[0]["path"])
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, tuple(config["image_size"]))
    image = tf.cast(image, tf.float32) / 255.0
    return np.expand_dims(image.numpy(), axis=0)


def set_input(interpreter, input_detail, x):
    dtype = input_detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = input_detail["quantization"]
        if scale == 0:
            raise ValueError("Quantized input scale is zero.")
        x = np.round(x / scale + zero_point)
        info = np.iinfo(dtype)
        x = np.clip(x, info.min, info.max).astype(dtype)
    else:
        x = x.astype(dtype)
    interpreter.set_tensor(input_detail["index"], x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--tflite-model", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    x = load_first_image(cfg)

    for _ in range(args.warmup):
        set_input(interpreter, input_detail, x)
        interpreter.invoke()

    times = []
    tracemalloc.start()
    for _ in range(args.runs):
        set_input(interpreter, input_detail, x)
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    arr = np.asarray(times, dtype=float)
    mean_ms = float(arr.mean())
    metrics = {
        "runs": args.runs,
        "warmup_runs": args.warmup,
        "mean_latency_ms": mean_ms,
        "latency_sd_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "fps": float(1000.0 / mean_ms) if mean_ms > 0 else None,
        "model_size_mb": os.path.getsize(args.tflite_model) / (1024.0 ** 2),
        "python_tracemalloc_peak_mb": peak_bytes / (1024.0 ** 2),
        "energy_per_inference": None,
        "energy_note": (
            "Not estimated. Processor TDP is not used as a surrogate for "
            "hardware-specific inference energy."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
