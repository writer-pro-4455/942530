from __future__ import annotations

"""
Small orchestration helper.

This intentionally stops short of uploading anything to ThingsBoard because
broker credentials are author-specific. Run MQTT/IoT commands separately.
"""

import subprocess
import sys


def run(*args):
    cmd = [sys.executable, *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    config = "configs/experiment.json"

    run("-m", "src.data", "--config", config, "--make-splits")

    for model in ["custom_cnn", "mobilenet_v2", "mobilenet_v3_small", "efficientnet_b0"]:
        run("-m", "src.train", "--config", config, "--model", model)
        run(
            "-m", "src.evaluate",
            "--config", config,
            "--model-path", f"models/{model}_best.keras",
            "--split", "test",
            "--output-dir", f"results/{model}_test",
        )

    run(
        "-m", "src.quantize",
        "--config", config,
        "--model-path", "models/mobilenet_v3_small_best.keras",
        "--mode", "fp16",
        "--output", "models/mobilenet_v3_small_fp16.tflite",
    )

    run(
        "-m", "src.quantize",
        "--config", config,
        "--model-path", "models/mobilenet_v3_small_best.keras",
        "--mode", "int8",
        "--output", "models/mobilenet_v3_small_int8.tflite",
    )

    run(
        "-m", "src.benchmark",
        "--config", config,
        "--tflite-model", "models/mobilenet_v3_small_int8.tflite",
        "--warmup", "5",
        "--runs", "100",
        "--output", "results/int8_benchmark.json",
    )


if __name__ == "__main__":
    main()
