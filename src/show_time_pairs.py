"""Show test (or val) images next to their closest same-class TRAIN photo taken seconds apart.

Run `python src/check_time_leakage.py` first (it writes outputs/time_leakage_pairs.csv), then:

    python src/show_time_pairs.py                 # 10 random test pairs taken <= 5 seconds apart
    python src/show_time_pairs.py --n 16 --seed 7 # a different sample

Output: outputs/plots/time_leakage_pairs.png (also usable as a figure in the report).
Pairs are picked at random (not the closest ones) so the picture is representative.
"""
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "time_leakage_pairs.csv"))
    ap.add_argument("--root", default=str(ROOT), help="folder that the file paths in the CSV are relative to")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max-gap", type=int, default=5, help="only pairs taken at most this many seconds apart")
    ap.add_argument("--n", type=int, default=10, help="number of pairs to show")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "plots" / "time_leakage_pairs.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[(df["split"] == args.split) & (df["gap_seconds"] <= args.max_gap)]
    if df.empty:
        raise SystemExit("No pairs match. Try a larger --max-gap.")
    sample = df.sample(min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    pairs_per_row = 2
    rows = math.ceil(len(sample) / pairs_per_row)
    fig, axes = plt.subplots(rows, pairs_per_row * 2, figsize=(pairs_per_row * 2 * 2.6, rows * 2.9))
    axes = axes.reshape(rows, pairs_per_row * 2)
    for ax in axes.ravel():
        ax.axis("off")

    root = Path(args.root)
    for k, r in sample.iterrows():
        row, col = divmod(k, pairs_per_row)
        a, b = axes[row, col * 2], axes[row, col * 2 + 1]
        a.imshow(Image.open(root / r["file"]).convert("RGB"))
        a.set_title(f"{r['label']}\n{args.split}", fontsize=8)
        b.imshow(Image.open(root / r["closest_train_file"]).convert("RGB"))
        b.set_title(f"train, {int(r['gap_seconds'])} s apart", fontsize=8)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {len(sample)} pairs to {out}")
    print("Left of each pair = the", args.split, "image. Right = its closest train photo of the same class.")


if __name__ == "__main__":
    main()