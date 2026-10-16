"""LeafLens training script.

Examples (run from the repo root, with the .venv active):

    python src/train.py --model baseline --epochs 3 --run-name smoke_test   # quick timing test
    python src/train.py --model baseline --epochs 30
    python src/train.py --model mobilenet_small --epochs 15 --finetune-epochs 15
    python src/train.py --model mobilenet_large --epochs 15 --finetune-epochs 15

What it does:
- Loads train/val data through data/manifest_grouped_10s.csv (configured in src/data.py), seed 42.
- Trains with class weights (Cutting Weevil is the small class), early stopping on
  val_loss, and keeps the best weights.
- MobileNetV3 runs in two stages: (1) frozen backbone, train the head; (2) unfreeze the
  top layers and fine-tune with a low learning rate. Set --finetune-epochs 0 to skip stage 2.
- Saves models to outputs/models/.
- Saves epoch-by-epoch training history to outputs/history/.
- Appends one row per run to outputs/runs.csv.

The TEST split is never touched here. It is used once, at the very end, by evaluate.py.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from data import (
    SEED, ROOT, MANIFEST,
    get_class_names, get_class_weights, make_dataset
)

from model import build_baseline, build_mobilenet, unfreeze  # noqa: E402

MODELS_DIR = ROOT / "outputs" / "models"
RUNS_CSV = ROOT / "outputs" / "runs.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Train a LeafLens model")
    p.add_argument("--model", choices=["baseline", "mobilenet_small", "mobilenet_large"], required=True)
    p.add_argument("--epochs", type=int, default=20, help="epochs for stage 1 (or the whole baseline run)")
    p.add_argument("--finetune-epochs", type=int, default=0, help="stage 2 epochs (MobileNet only)")
    p.add_argument("--unfreeze-layers", type=int, default=30, help="top backbone layers to unfreeze")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate for stage 1 / baseline")
    p.add_argument("--finetune-lr", type=float, default=1e-5, help="learning rate for stage 2")
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    p.add_argument("--no-augment", action="store_true", help="turn off training augmentation")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

def fit(model, train_ds, val_ds, epochs, class_weight, patience, checkpoint_path):
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight, callbacks=cbs, verbose=2)

def evaluate_val(model, val_ds):
    """Validation loss, accuracy and macro-F1 (val split only)."""
    y_true = np.concatenate([y.numpy() for _, y in val_ds])
    probs = model.predict(val_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    loss, acc = model.evaluate(val_ds, verbose=0)
    return {
        "val_loss": round(float(loss), 4),
        "val_acc": round(float(acc), 4),
        "val_macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def log_run(row):
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if RUNS_CSV.exists():
        df = pd.concat([pd.read_csv(RUNS_CSV), df], ignore_index=True)
    df.to_csv(RUNS_CSV, index=False)

def save_history(history, run_name, stage):
    history_dir = ROOT / "outputs" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(history.history)
    df.index = df.index + 1
    df.index.name = "epoch"

    path = history_dir / f"{run_name}_{stage}.csv"
    df.to_csv(path)

    print(f"Training history saved: {path}")


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)
    run_name = args.run_name or f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset("train", args.batch_size, augment=not args.no_augment)
    val_ds = make_dataset("val", args.batch_size)
    class_weight = None if args.no_class_weights else get_class_weights()
    num_classes = len(get_class_names())

    start = time.time()
    base = None
    if args.model == "baseline":
        model = build_baseline(num_classes)
    else:
        variant = args.model.split("_")[1]
        model, base = build_mobilenet(variant, num_classes)
    model.summary()

    # ---- stage 1 (whole run for the baseline; frozen backbone for MobileNet)
    compile_model(model, args.lr)
    h1 = fit(model, train_ds, val_ds, args.epochs, class_weight, args.patience, MODELS_DIR / f"{run_name}_stage1_best.keras")
    save_history(h1, run_name, "stage1")
    results = evaluate_val(model, val_ds)
    print("after stage 1:", results)
    model.save(MODELS_DIR / f"{run_name}_stage1.keras")
    epochs_run = len(h1.history["loss"])
    final_stage = "stage1"

    # ---- stage 2: fine-tuning (MobileNet only)
    if base is not None and args.finetune_epochs > 0:
        unfreeze(base, args.unfreeze_layers)
        compile_model(model, args.finetune_lr)  # must recompile after changing trainable
        h2 = fit(model, train_ds, val_ds, args.finetune_epochs, class_weight, args.patience, MODELS_DIR / f"{run_name}_stage2_best.keras")
        save_history(h2, run_name, "stage2")
        stage2 = evaluate_val(model, val_ds)
        print("after stage 2:", stage2)
        model.save(MODELS_DIR / f"{run_name}_stage2.keras")
        epochs_run += len(h2.history["loss"])
        # keep whichever stage has the lower validation loss as the final model
        if stage2["val_loss"] <= results["val_loss"]:
            results, final_stage = stage2, "stage2"
    final_path = MODELS_DIR / f"{run_name}_{final_stage}.keras"

    minutes = round((time.time() - start) / 60, 1)
    log_run(
        {
            "run_name": run_name,
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "seed": SEED,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "finetune_lr": args.finetune_lr if base is not None and args.finetune_epochs > 0 else "",
            "unfreeze_layers": args.unfreeze_layers if base is not None and args.finetune_epochs > 0 else "",
            "max_epochs": args.epochs,
            "max_finetune_epochs": args.finetune_epochs,
            "epochs_run": epochs_run,
            "augment": not args.no_augment,
            "class_weights": not args.no_class_weights,
            "final_stage": final_stage,
            **results,
            "minutes": minutes,
            "model_path": str(final_path.relative_to(ROOT)),
            "manifest": MANIFEST.name,
        }
    )
    print(f"\nDone in {minutes} min. Best model: {final_path}")
    print(f"Run logged to {RUNS_CSV}")


if __name__ == "__main__":
    main()
