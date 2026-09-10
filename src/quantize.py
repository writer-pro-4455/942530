from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import load_config, load_manifest


def preprocess_image(path: str, image_size):
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()


def representative_dataset(config, n_samples):
    df = load_manifest(config, "train")
    sample = df.sample(
        n=min(n_samples, len(df)),
        random_state=int(config["seed"]),
        replace=False,
    )
    image_size = tuple(config["image_size"])
    for path in sample["path"]:
        image = preprocess_image(str(path), image_size)
        yield [np.expand_dims(image, axis=0).astype(np.float32)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", required=True, choices=["fp16", "int8"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = tf.keras.models.load_model(args.model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.mode == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        n = int(cfg["quantization"]["representative_samples"])
        converter.representative_dataset = lambda: representative_dataset(cfg, n)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Keep a float input/output interface for simple reproducible inference
        # while quantizing internal weights/activations.
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(tflite_model)

    print(f"Saved {args.mode} model to {output} ({output.stat().st_size / (1024**2):.3f} MB)")


if __name__ == "__main__":
    main()
