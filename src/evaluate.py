import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from data import ROOT, MANIFEST, get_class_names, make_dataset


def evaluate(model_path, batch_size=32):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir = ROOT / "outputs" / "evaluation" / model_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {model_path.name}")
    model = tf.keras.models.load_model(model_path)

    test_ds = make_dataset(
        "test",
        batch_size=batch_size,
        augment=False
    )

    class_names = get_class_names()

    # Test labels must remain in prediction order.
    y_true = np.concatenate([
        labels.numpy() for _, labels in test_ds
    ])

    expected_count = int(
        (pd.read_csv(MANIFEST)["split"] == "test").sum()
    )

    if len(y_true) != expected_count:
        raise ValueError(
            f"Expected {expected_count} test images, found {len(y_true)}"
        )

    # Warm up the model before timing.
    sample_images, _ = next(iter(test_ds))
    model(sample_images, training=False).numpy()

    start = time.perf_counter()
    probabilities = model.predict(test_ds, verbose=1)
    elapsed = time.perf_counter() - start

    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    ))

    # Save per-class precision, recall, and F1.
    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv"
    )

    # Save confusion matrix as CSV and PNG.
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(class_names))
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    ).to_csv(output_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    ).plot(ax=ax, cmap="Blues", xticks_rotation=45)

    ax.set_title(f"{model_path.stem} - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # Save prediction details.
    pd.DataFrame({
        "true_label": [class_names[i] for i in y_true],
        "predicted_label": [class_names[i] for i in y_pred],
        "confidence": probabilities.max(axis=1)
    }).to_csv(output_dir / "predictions.csv", index=False)

    # Model efficiency.
    size_mb = model_path.stat().st_size / (1024 ** 2)
    ms_per_image = (elapsed / len(y_true)) * 1000

    summary = {
        "model": model_path.stem,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "model_size_mb": size_mb,
        "prediction_seconds": elapsed,
        "ms_per_image": ms_per_image,
        "test_images": len(y_true)
    }

    pd.DataFrame([summary]).to_csv(
        output_dir / "summary.csv",
        index=False
    )

    print("\nEvaluation Summary")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    evaluate(args.model_path, args.batch_size)