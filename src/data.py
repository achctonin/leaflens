"""LeafLens data pipeline: manifest -> batched tf.data datasets.

Preprocessing: read the image, stretch-resize to 224x224, keep raw 0-255 pixels
(Keras MobileNetV3 has its own preprocessing built in, so do NOT normalize here).
Augmentation: training split only.

Usage from a notebook or script (any working directory):

    from src.data import make_dataset, get_class_names, get_class_weights
    train_ds = make_dataset("train", batch_size=32, augment=True)
    val_ds = make_dataset("val", batch_size=32)
    test_ds = make_dataset("test", batch_size=32)   # use once, at the very end

Sanity check (run from the repo root):  python src/data.py
"""
from pathlib import Path

import pandas as pd
import tensorflow as tf

SEED = 42
IMG_SIZE = (224, 224)
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest_grouped_10s.csv"
AUTOTUNE = tf.data.AUTOTUNE


def get_class_names(manifest=MANIFEST):
    """Sorted class names. The index in this list is the integer label."""
    df = pd.read_csv(manifest)
    return sorted(df["label"].unique())


def get_class_weights(manifest=MANIFEST):
    """Class weights from the TRAIN split (for model.fit(class_weight=...)).
    Cutting Weevil has about half the images of the other classes."""
    df = pd.read_csv(manifest)
    class_names = sorted(df["label"].unique())
    counts = df[df["split"] == "train"]["label"].value_counts()
    total, k = counts.sum(), len(class_names)
    return {i: float(total / (k * counts[name])) for i, name in enumerate(class_names)}


# ---------------------------------------------------------------- preprocessing
def _load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)  # stretch to 224x224 (float32, 0-255)
    return img, label


# ---------------------------------------------------------------- augmentation
def build_augmentation():
    """Random flip, small rotation, zoom, brightness and contrast (training only)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            tf.keras.layers.RandomRotation(0.05, seed=SEED),  # about +/-18 degrees
            tf.keras.layers.RandomZoom(0.1, seed=SEED),
            tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255), seed=SEED),
            tf.keras.layers.RandomContrast(0.2, seed=SEED),
        ],
        name="augmentation",
    )


# ---------------------------------------------------------------- dataset
def make_dataset(split, batch_size=32, augment=False, shuffle=None, manifest=MANIFEST):
    """Build a batched dataset of (images, integer labels) for one split.

    split:   "train", "val" or "test" (read from the manifest, never from folders)
    augment: only allowed for the train split
    shuffle: defaults to True for train, False otherwise
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train, val or test, got {split!r}")
    if augment and split != "train":
        raise ValueError("Augmentation is for the train split only.")
    if shuffle is None:
        shuffle = split == "train"

    df = pd.read_csv(manifest)
    class_names = sorted(df["label"].unique())
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    df = df[df["split"] == split]
    paths = [str(ROOT / p) for p in df["filepath"]]
    labels = df["label"].map(label_to_idx).astype("int32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment:
        aug = build_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ---------------------------------------------------------------- sanity check
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    names = get_class_names()
    print("classes:", names)
    print("class weights:", {names[i]: round(w, 2) for i, w in get_class_weights().items()})

    for split in ["train", "val", "test"]:
        n = sum(1 for _ in make_dataset(split, batch_size=1, shuffle=False))
        print(f"{split}: {n} images")

    train_ds = make_dataset("train", batch_size=8, augment=True)
    x, y = next(iter(train_ds))
    print("batch:", x.shape, x.dtype, "min/max:", float(tf.reduce_min(x)), float(tf.reduce_max(x)))

    out = ROOT / "outputs" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, img, lab in zip(axes.ravel(), x, y):
        ax.imshow(tf.cast(img, tf.uint8).numpy())
        ax.set_title(names[int(lab)], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / "augmentation_preview.png", dpi=150)
    print("saved", out / "augmentation_preview.png")