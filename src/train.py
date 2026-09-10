from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tensorflow as tf

from .data import class_weights_from_training, load_config, make_dataset, set_seed
from .models import build_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def callbacks_for(path: Path, reduce_lr=True):
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(path), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=str(path.parent / "tensorboard" / path.stem)),
    ]
    if reduce_lr:
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1))
    return cbs


def unfreeze_final_fraction(backbone: tf.keras.Model, fraction: float):
    n = len(backbone.layers)
    start = max(0, int(round(n * (1.0 - fraction))))
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= start


def merge_histories(h1, h2=None):
    out = {k: list(map(float, vals)) for k, vals in h1.history.items()}
    if h2:
        for k, vals in h2.history.items():
            out.setdefault(k, []).extend(map(float, vals))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--model", required=True, choices=["custom_cnn", "mobilenet_v2", "mobilenet_v3_small", "efficientnet_b0"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    models_dir = Path(cfg["paths"]["models_dir"])
    results_dir = Path(cfg["paths"]["results_dir"]) / args.model
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(cfg, "train", training=True)
    val_ds = make_dataset(cfg, "val", training=False)
    class_weight = class_weights_from_training(cfg)
    model, backbone = build_model(args.model, input_shape=tuple(cfg["image_size"]) + (3,), num_classes=len(cfg["class_names"]))

    best_path = models_dir / f"{args.model}_best.keras"
    t0 = time.perf_counter()
    if args.model == "custom_cnn":
        compile_model(model, cfg["training"]["phase1_learning_rate"])
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=int(cfg["training"]["custom_cnn_epochs"]), class_weight=class_weight, callbacks=callbacks_for(best_path, reduce_lr=True))
        h2 = None
    else:
        compile_model(model, cfg["training"]["phase1_learning_rate"])
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=int(cfg["training"]["phase1_epochs"]), class_weight=class_weight, callbacks=callbacks_for(best_path, reduce_lr=True))
        unfreeze_final_fraction(backbone, float(cfg["training"]["fine_tune_fraction"]))
        steps_per_epoch = max(1, int(tf.data.experimental.cardinality(train_ds).numpy()))
        decay_steps = steps_per_epoch * int(cfg["training"]["phase2_epochs"])
        schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=float(cfg["training"]["phase2_learning_rate"]), decay_steps=decay_steps)
        compile_model(model, schedule)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=int(cfg["training"]["phase2_epochs"]), class_weight=class_weight, callbacks=callbacks_for(best_path, reduce_lr=False))

    train_seconds = time.perf_counter() - t0
    with open(results_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(merge_histories(h1, h2), f, indent=2)
    metadata = {"model": args.model, "training_seconds": train_seconds, "seed": int(cfg["seed"]), "batch_size": int(cfg["batch_size"]), "best_checkpoint": str(best_path)}
    with open(results_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
