"""Diagnose why the baseline CNN scores near chance on validation while training accuracy is high.

Usage (from the repo root, after a baseline run has finished):

    python src/diagnose_bn.py outputs/models/baseline_noaug_stage1.keras

It measures accuracy two ways on the same images:
  - inference mode   : BatchNorm uses its saved running statistics (what validation does)
  - batch-stats mode : BatchNorm uses each batch's own statistics (what training does)
Images are shuffled so every batch mixes classes. Dropout is active in batch-stats mode,
so that number is slightly pessimistic. It only uses train and val images, never test.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import get_class_names, make_dataset  # noqa: E402

N_BATCHES = 20  # 20 batches x 32 images = 640 images per measurement


def accuracy(model, ds, training):
    correct = total = 0
    preds = []
    for x, y in ds.take(N_BATCHES):
        p = model(x, training=training).numpy().argmax(axis=1)
        correct += int((p == y.numpy()).sum())
        total += len(p)
        preds.extend(p.tolist())
    return correct / total, np.bincount(preds, minlength=8)


def main():
    model = tf.keras.models.load_model(sys.argv[1])
    names = get_class_names()
    for split in ["train", "val"]:
        ds = make_dataset(split, batch_size=32, augment=False, shuffle=True)
        acc_inf, hist_inf = accuracy(model, ds, training=False)
        acc_bs, _ = accuracy(model, ds, training=True)
        print(f"{split:5s}  inference-mode acc: {acc_inf:.3f}   batch-stats acc: {acc_bs:.3f}")
        print("       predicted-class counts (inference mode):", dict(zip(names, hist_inf.tolist())))


if __name__ == "__main__":
    main()