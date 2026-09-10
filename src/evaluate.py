from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .data import load_config, load_manifest, make_dataset


def save_confusion(cm, labels, out_path, normalize=False):
    arr = cm.astype(float)
    if normalize:
        denom = arr.sum(axis=1, keepdims=True)
        arr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    ax.set_title(title)

    threshold = arr.max() / 2.0 if arr.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            txt = f"{arr[i, j]:.2f}" if normalize else str(int(arr[i, j]))
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if arr[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_manifest(cfg, args.split)
    ds = make_dataset(cfg, args.split, training=False, shuffle=False)
    model = tf.keras.models.load_model(args.model_path)

    probs = model.predict(ds, verbose=1)
    y_pred = probs.argmax(axis=1)
    y_true = df["class_index"].astype(int).values
    labels = cfg["class_names"]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(labels))
    try:
        auc = roc_auc_score(y_true_onehot, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = None

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_roc_auc_ovr": None if auc is None else float(auc),
    }

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")

    pred_df = df.copy()
    pred_df["predicted_index"] = y_pred
    pred_df["predicted_class"] = [labels[i] for i in y_pred]
    pred_df["confidence"] = probs.max(axis=1)
    for i, name in enumerate(labels):
        pred_df[f"prob_{name}"] = probs[:, i]
    pred_df.to_csv(out / "predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    np.savetxt(out / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    save_confusion(cm, labels, out / "confusion_matrix.png", normalize=False)
    save_confusion(cm, labels, out / "confusion_matrix_normalized.png", normalize=True)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
