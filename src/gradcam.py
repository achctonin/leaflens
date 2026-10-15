"""Grad-CAM for the LeafLens MobileNetV3 models.

Single image:
    python src/gradcam.py outputs/models/mnv3_small_v1_stage2.keras --image "data/raw/Healthy/xxx.jpg"

Misclassified review (use val until the final model is chosen):
    python src/gradcam.py outputs/models/mnv3_small_v1_stage2.keras --split val --max 24

Assumptions to verify against src/data.py and src/model.py:
  - images are resized to 224x224 with a plain resize and fed as raw 0-255 floats
  - class index order is the alphabetical order of the labels in data/manifest.csv
  - the final Dense layer outputs probabilities (softmax)
"""
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMG_SIZE = (224, 224)
OUT_DIR = "outputs/gradcam"


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img, np.array(img.resize(IMG_SIZE), dtype="float32")  # raw 0-255


def gradcam(model, x, class_idx=None):
    """Return (heatmap 7x7 in [0,1], predicted/used class index, probabilities)."""
    base = model.layers[1]          # nested MobileNetV3 base -> (None, 7, 7, C)
    head = model.layers[2:]         # pooling, dropout, dense
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
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR)
    heat = plt.get_cmap("jet")(np.array(cam_img) / 255.0)[..., :3]
    base = np.array(img, dtype="float32") / 255.0
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def explain(model, classes, path, true_label=None):
    img, x = load_image(path)
    cam, idx, probs = gradcam(model, x)
    return img, overlay(img, cam), classes[idx], float(probs[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--image")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max", type=int, default=24)
    ap.add_argument("--per-class", type=int, default=0,
                    help="N images per class from --split, saved as one grid")
    args = ap.parse_args()

    df = pd.read_csv("data/manifest.csv")
    classes = sorted(df["label"].unique())
    model = tf.keras.models.load_model(args.model)
    os.makedirs(OUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.model))[0]

    if args.image:
        img, ov, pred, p = explain(model, classes, args.image)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img); ax[0].set_title("input")
        ax[1].imshow(ov); ax[1].set_title(f"{pred} ({p:.2f})")
        for a in ax: a.axis("off")
        out = os.path.join(OUT_DIR, f"{name}_single.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print("saved", out)
        return

    if args.per_class:
        sub = df[df["split"] == args.split]
        picks = pd.concat(
            [sub[sub["label"] == c].head(args.per_class) for c in classes]
        )
        n_rows = len(picks) // 4 + (len(picks) % 4 > 0)
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for a in axes: a.axis("off")
        for a, (_, r) in zip(axes, picks.iterrows()):
            img, ov, pred, p = explain(model, classes, r["filepath"])
            a.imshow(ov)
            a.set_title(f"true: {r['label']}\npred: {pred} ({p:.2f})", fontsize=9)
        out = os.path.join(OUT_DIR, f"{name}_{args.split}_per_class.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print("saved", out)
        return

    # Misclassified review on one split
    rows = df[df["split"] == args.split]
    wrong = []
    for _, r in rows.iterrows():
        img, x = load_image(r["filepath"])
        probs = model(x[None], training=False).numpy()[0]
        pred = classes[int(np.argmax(probs))]
        if pred != r["label"]:
            wrong.append((r["filepath"], r["label"], pred, float(probs.max())))
    print(f"{len(wrong)} misclassified of {len(rows)} in split '{args.split}'")
    pd.DataFrame(wrong, columns=["filepath", "true", "pred", "conf"]).to_csv(
        os.path.join(OUT_DIR, f"{name}_{args.split}_misclassified.csv"), index=False)

    wrong = wrong[: args.max]
    if not wrong:
        return
    cols = 4
    n_rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(4 * cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for a in axes: a.axis("off")
    for a, (path, true, pred, conf) in zip(axes, wrong):
        img, ov, _, _ = explain(model, classes, path)
        a.imshow(ov)
        a.set_title(f"true: {true}\npred: {pred} ({conf:.2f})", fontsize=9)
    out = os.path.join(OUT_DIR, f"{name}_{args.split}_misclassified.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()