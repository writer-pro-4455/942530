from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

AUTOTUNE = tf.data.AUTOTUNE


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_class_dir(root: Path, canonical: str, aliases: list[str]) -> Path:
    candidates = [canonical] + aliases
    for name in candidates:
        p = root / name
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        f"Could not find folder for class {canonical!r} under {root}. "
        f"Tried: {candidates}"
    )


def collect_images(config: dict) -> pd.DataFrame:
    root = Path(config["paths"]["primary_root"])
    rows = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_index, class_name in enumerate(config["class_names"]):
        aliases = config.get("class_aliases", {}).get(class_name, [])
        class_dir = resolve_class_dir(root, class_name, aliases)
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                rows.append({
                    "path": str(path.resolve()),
                    "class_name": class_name,
                    "class_index": class_index
                })

    if not rows:
        raise RuntimeError(f"No images found under {root}")

    return pd.DataFrame(rows)


def make_splits(config: dict) -> dict[str, pd.DataFrame]:
    seed = int(config["seed"])
    df = collect_images(config)
    ratios = config["split"]
    test_size = float(ratios["test"])
    val_size = float(ratios["val"])

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["class_index"],
        shuffle=True,
    )

    # Convert requested validation fraction into a fraction of the train+val pool.
    val_relative = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        random_state=seed,
        stratify=train_val["class_index"],
        shuffle=True,
    )

    out_dir = Path(config["paths"]["split_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        part = part.sort_values(["class_index", "path"]).reset_index(drop=True)
        part.to_csv(out_dir / f"{name}.csv", index=False)
        splits[name] = part

    return splits


def load_manifest(config: dict, split: str) -> pd.DataFrame:
    path = Path(config["paths"]["split_dir"]) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data --config ... --make-splits` first."
        )
    return pd.read_csv(path)


def _decode(path, label, image_size):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def _augment(image, label, cfg):
    aug = cfg["augmentation"]
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=float(aug["brightness_delta"]))
    image = tf.image.random_contrast(
        image, lower=float(aug["contrast_lower"]), upper=float(aug["contrast_upper"])
    )
    image = tf.image.random_saturation(
        image, lower=float(aug["saturation_lower"]), upper=float(aug["saturation_upper"])
    )
    pad_h, pad_w = aug["pad_resize"]
    crop_h, crop_w = aug["random_crop"]
    image = tf.image.resize_with_pad(image, pad_h, pad_w)
    image = tf.image.random_crop(image, [crop_h, crop_w, 3])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def make_dataset(
    config: dict,
    split: str,
    training: bool = False,
    batch_size: int | None = None,
    shuffle: bool | None = None,
) -> tf.data.Dataset:
    df = load_manifest(config, split)
    batch_size = batch_size or int(config["batch_size"])
    if shuffle is None:
        shuffle = training

    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].astype(str).values, df["class_index"].astype(int).values)
    )

    if shuffle:
        ds = ds.shuffle(len(df), seed=int(config["seed"]), reshuffle_each_iteration=True)

    image_size = tuple(config["image_size"])
    ds = ds.map(lambda p, y: _decode(p, y, image_size), num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(lambda x, y: _augment(x, y, config), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def class_weights_from_training(config: dict) -> dict[int, float]:
    df = load_manifest(config, "train")
    classes = np.array(sorted(df["class_index"].unique()))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=df["class_index"].values
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def summarize(splits: dict[str, pd.DataFrame], config: dict) -> None:
    for name, df in splits.items():
        print(f"\n{name.upper()}: {len(df)}")
        print(df["class_name"].value_counts().reindex(config["class_names"]).fillna(0).astype(int))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--make-splits", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    if args.make_splits:
        splits = make_splits(cfg)
        summarize(splits, cfg)
    else:
        df = collect_images(cfg)
        print(df["class_name"].value_counts())
        print(f"Total: {len(df)}")


if __name__ == "__main__":
    main()
