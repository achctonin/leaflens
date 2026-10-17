"""Grad-CAM for the LeafLens MobileNetV3 models.

Reuses the project's own pipeline in src/data.py, so images are loaded and preprocessed
exactly as in training and evaluate.py (tf.io decode, stretch-resize to 224x224, raw 0-255
pixels), and the manifest and class names come from data.py (manifest_grouped_10s.csv).

Single image (any photo, e.g. a local photo):
    python src/gradcam.py outputs/models/<model>.keras --image "data/local/Healthy/leaf1.jpg"

One or more images per class from a split:
    python src/gradcam.py outputs/models/<model>.keras --split val --per-class 1

Misclassified images of a split:
    python src/gradcam.py outputs/models/<model>.keras --split val --max 24

Use --split val until the final model is chosen. The test split is for the single final run
and needs --allow-test. MobileNetV3 models only (the baseline CNN has no nested backbone).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, MANIFEST, get_class_names, make_dataset, _load_image  # noqa: E402

OUT_DIR = ROOT / "outputs" / "gradcam"


def resolve(path):
    """Use the path as given if it exists, otherwise treat it as relative to the repo root."""
    p = Path(path)
    return p if p.exists() else ROOT / p


def load_image(path):
    """Same loading and resize as training. Returns float32 (224, 224, 3), values 0-255."""
    img, _ = _load_image(str(resolve(path)), 0)
    return img.numpy()


def split_model(model):
    """Return (backbone, head layers). The backbone's output is the last conv feature map."""
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            return layer, model.layers[i + 1:]
    raise SystemExit("Grad-CAM here supports the MobileNetV3 models (nested backbone) only.")


def gradcam(model, x, class_idx=None):
    """Return (heatmap in [0,1], class index used, probabilities)."""
    base, head = split_model(model)
    with tf.GradientTape() as tape:
        fmap = base(x[None], training=False)
        tape.watch(fmap)
        y = fmap
        for layer in head:
            y = layer(y, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(y[0]))
        score = y[0, class_idx]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), class_idx, y[0].numpy()


def overlay(img, cam, alpha=0.4):
    h, w = img.shape[:2]
    cam_r = tf.image.resize(cam[..., None], (h, w), method="bilinear").numpy()[..., 0]
    heat = plt.get_cmap("jet")(cam_r)[..., :3]
    base = img.astype("float32") / 255.0
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def explain(model, classes, path):
    x = load_image(path)
    cam, idx, probs = gradcam(model, x)
    return x.astype("uint8"), overlay(x, cam), classes[idx], float(probs[idx]), probs


def save_grid(model, classes, items, out, cols=4):
    """items: list of (path, true_label). Saves one grid of Grad-CAM overlays."""
    n_rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(4 * cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for a, (path, true) in zip(axes, items):
        _, ov, pred, p, _ = explain(model, classes, path)
        a.imshow(ov)
        a.set_title(f"true: {true}\npred: {pred} ({p:.2f})", fontsize=9)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--image")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--max", type=int, default=24)
    ap.add_argument("--per-class", type=int, default=0,
                    help="N images per class from --split, saved as one grid")
    ap.add_argument("--allow-test", action="store_true",
                    help="needed for --split test (use it only for the final run)")
    args = ap.parse_args()

    if args.split == "test" and not args.image and not args.allow_test:
        raise SystemExit("The test split is for the single final run. Add --allow-test if that is intended.")

    classes = get_class_names()
    model = tf.keras.models.load_model(args.model, compile=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(args.model).stem

    if args.image:
        img, ov, pred, p, probs = explain(model, classes, args.image)
        print(f"Predicted: {pred} ({p * 100:.1f}%)")
        for i in np.argsort(probs)[::-1]:
            print(f"  {classes[i]}: {probs[i] * 100:.1f}%")
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img); ax[0].set_title("input (as the model sees it)", fontsize=9)
        ax[1].imshow(ov); ax[1].set_title(f"{pred} ({p:.2f})")
        for a in ax:
            a.axis("off")
        tag = f"{Path(args.image).parent.name}_{Path(args.image).stem}".replace(" ", "_")
        out = OUT_DIR / f"{name}_{tag}_single.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print("saved", out)
        return

    df = pd.read_csv(MANIFEST)
    rows = df[df["split"] == args.split].reset_index(drop=True)

    if args.per_class:
        picks = pd.concat([rows[rows["label"] == c].head(args.per_class) for c in classes])
        items = list(zip(picks["filepath"], picks["label"]))
        save_grid(model, classes, items, OUT_DIR / f"{name}_{args.split}_per_class.png")
        return

    # Misclassified review: batch prediction with the same dataset code as evaluate.py
    ds = make_dataset(args.split, batch_size=32, augment=False, shuffle=False)
    probs = model.predict(ds, verbose=0)
    y_true = rows["label"].map({c: i for i, c in enumerate(classes)}).values
    y_pred = probs.argmax(axis=1)
    wrong = np.where(y_pred != y_true)[0]
    print(f"{len(wrong)} misclassified of {len(rows)} in split '{args.split}'")

    table = pd.DataFrame({
        "filepath": rows.loc[wrong, "filepath"].values,
        "true": rows.loc[wrong, "label"].values,
        "pred": [classes[i] for i in y_pred[wrong]],
        "conf": probs[wrong].max(axis=1),
    })
    table.to_csv(OUT_DIR / f"{name}_{args.split}_misclassified.csv", index=False)

    if len(table):
        items = list(zip(table["filepath"], table["true"]))[: args.max]
        save_grid(model, classes, items, OUT_DIR / f"{name}_{args.split}_misclassified.png")


if __name__ == "__main__":
    main()