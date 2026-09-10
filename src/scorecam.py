from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from .data import load_config


def load_image(path, size):
    image = Image.open(path).convert("RGB").resize((size[1], size[0]))
    arr = np.asarray(image).astype(np.float32) / 255.0
    return image, arr


def find_target_conv_layer(model):
    # Search backwards for the last rank-4 layer output that can be connected.
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
        except Exception:
            continue
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer.name
    raise RuntimeError("No suitable rank-4 layer found for ScoreCAM.")


def scorecam(model, image_batch, class_index=None, max_channels=50, target_layer=None):
    if target_layer is None:
        target_layer = find_target_conv_layer(model)

    layer = model.get_layer(target_layer)
    activation_model = tf.keras.Model(model.inputs, [layer.output, model.output])

    activations, predictions = activation_model(image_batch, training=False)
    activations = activations[0].numpy()
    predictions = predictions[0].numpy()
    if class_index is None:
        class_index = int(np.argmax(predictions))

    n_channels = activations.shape[-1]
    channel_indices = np.linspace(
        0, n_channels - 1, num=min(max_channels, n_channels), dtype=int
    )

    heatmap = np.zeros(activations.shape[:2], dtype=np.float32)
    h, w = image_batch.shape[1:3]

    for ch in channel_indices:
        fmap = activations[..., ch]
        fmin, fmax = float(fmap.min()), float(fmap.max())
        if fmax - fmin < 1e-12:
            continue

        norm = (fmap - fmin) / (fmax - fmin)
        mask = tf.image.resize(norm[..., None], (h, w)).numpy()[..., 0]
        masked = image_batch[0] * mask[..., None]
        score = float(model(masked[None, ...], training=False)[0, class_index].numpy())
        heatmap += score * fmap

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = tf.image.resize(heatmap[..., None], (h, w)).numpy()[..., 0]
    return heatmap, class_index, predictions, target_layer


def save_overlay(original, heatmap, output, alpha=0.45):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("ScoreCAM")
    axes[1].axis("off")

    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--channels", type=int, default=50)
    parser.add_argument("--target-layer", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = tf.keras.models.load_model(args.model_path)
    original, arr = load_image(args.image, tuple(cfg["image_size"]))
    batch = arr[None, ...]

    heatmap, class_index, probs, target = scorecam(
        model,
        batch,
        max_channels=args.channels,
        target_layer=args.target_layer,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_overlay(original, heatmap, output, alpha=float(cfg["scorecam"]["overlay_alpha"]))

    print(f"Predicted class: {cfg['class_names'][class_index]}")
    print(f"Confidence: {float(probs[class_index]):.4f}")
    print(f"Target layer: {target}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
