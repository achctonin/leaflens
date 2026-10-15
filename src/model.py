"""LeafLens models: baseline CNN and MobileNetV3 (Small / Large) classifiers.

All models take raw 0-255 images of shape (224, 224, 3) from src/data.py.
- The baseline rescales to 0-1 itself (first layer).
- Keras MobileNetV3 has its preprocessing built in (include_preprocessing=True),
  so it must receive raw 0-255 pixels. Do NOT normalize before it.
"""
import tensorflow as tf
from keras import layers

IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = 8


def build_baseline(num_classes=NUM_CLASSES, dropout=0.3):
    """Small CNN from scratch (the model MobileNetV3 has to beat)."""
    inputs = layers.Input(shape=IMG_SHAPE)
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=8)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_mobilenet(variant="small", num_classes=NUM_CLASSES, dropout=0.3, weights="imagenet"):
    """MobileNetV3 backbone + new classification head.

    Returns (model, base). The base starts FROZEN (stage 1: train the head only).
    Call unfreeze(base, n_layers) for stage 2 (fine-tuning).
    """
    app = {
        "small": tf.keras.applications.MobileNetV3Small,
        "large": tf.keras.applications.MobileNetV3Large,
    }[variant]
    base = app(input_shape=IMG_SHAPE, include_top=False, weights=weights, include_preprocessing=True)
    base.trainable = False

    inputs = layers.Input(shape=IMG_SHAPE)
    x = base(inputs, training=False)  # training=False keeps BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"mobilenetv3_{variant}")
    return model, base


def unfreeze(base, n_layers=30):
    """Unfreeze the top n_layers of the backbone for fine-tuning.
    BatchNorm layers stay frozen (standard practice for small datasets)."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


if __name__ == "__main__":
    # Quick check: build each model without downloading weights.
    for name, model in [
        ("baseline", build_baseline()),
        ("mobilenet small", build_mobilenet("small", weights=None)[0]),
        ("mobilenet large", build_mobilenet("large", weights=None)[0]),
    ]:
        print(f"{name}: {model.count_params():,} parameters, output {model.output_shape}")