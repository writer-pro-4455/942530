from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def _classification_head(x, num_classes: int):
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    return layers.Dense(num_classes, activation="softmax", name="predictions")(x)


def custom_cnn(input_shape=(224, 224, 3), num_classes=4):
    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    for i, filters in enumerate([32, 64, 128], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.ReLU(name=f"relu_{i}")(x)
        x = layers.MaxPooling2D(pool_size=2, strides=2, name=f"pool_{i}")(x)

    outputs = _classification_head(x, num_classes)
    return models.Model(inputs, outputs, name="CustomCNN"), None


def transfer_model(name: str, input_shape=(224, 224, 3), num_classes=4):
    # Input images are normalized to [0,1] by the data pipeline.
    # The application backbones include/expect their own preprocessing conventions.
    if name == "mobilenet_v2":
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif name == "mobilenet_v3_small":
        backbone = tf.keras.applications.MobileNetV3Small(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif name == "efficientnet_b0":
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported model: {name}")

    backbone.trainable = False
    inputs = layers.Input(shape=input_shape, name="image")
    x = backbone(inputs, training=False)
    outputs = _classification_head(x, num_classes)
    model = models.Model(inputs, outputs, name=name)
    return model, backbone


def build_model(name: str, input_shape=(224, 224, 3), num_classes=4):
    if name == "custom_cnn":
        return custom_cnn(input_shape, num_classes)
    return transfer_model(name, input_shape, num_classes)
