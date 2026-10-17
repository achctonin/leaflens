"""Check whether val/test images were photographed seconds apart from train images of the same class.

The image file names contain the capture time (for example 20211008_124249 or
IMG_20211106_140012). Photos of the same leaf are often taken a few seconds apart, so a random
image-level split can put near-identical shots of one leaf into both train and test, which makes
the test score optimistic. Perceptual hashes can miss these (different angle or zoom), time stamps do not.

Usage (from the repo root):

    python src/check_time_leakage.py

Output: a summary table in the terminal and outputs/time_leakage_pairs.csv (val/test images with the
closest same-class train photo and the gap in seconds).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAPS = [2, 5, 10, 30, 60, 300]
STAMP = re.compile(r"(\d{8})_(\d{6})")


def parse_time(path):
    m = STAMP.search(Path(str(path)).name)
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S", errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "time_leakage_pairs.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype=str)
    df["time"] = df["filepath"].map(parse_time)
    n_missing = int(df["time"].isna().sum())
    print(f"Images: {len(df)}   without a time stamp in the file name: {n_missing}")
    df = df.dropna(subset=["time"]).copy()
    df["t"] = (df["time"] - pd.Timestamp("1970-01-01")).dt.total_seconds().round().astype("int64")  # seconds

    rows = []
    for label, g in df.groupby("label"):
        train = g[g["split"] == "train"].sort_values("t")
        if train.empty:
            continue
        tt = train["t"].values
        for _, r in g[g["split"] != "train"].iterrows():
            k = np.searchsorted(tt, r["t"])
            cand = [i for i in (k - 1, k) if 0 <= i < len(tt)]
            best = min(cand, key=lambda i: abs(tt[i] - r["t"]))
            rows.append(
                {
                    "split": r["split"],
                    "label": label,
                    "file": r["filepath"],
                    "closest_train_file": train.iloc[best]["filepath"],
                    "gap_seconds": int(abs(tt[best] - r["t"])),
                }
            )
    res = pd.DataFrame(rows)

    print("\nShare of images whose closest SAME-CLASS train photo was taken within N seconds:")
    print(f"{'N (s)':>6} | {'test':>14} | {'val':>14}")
    for n in GAPS:
        cells = []
        for split in ("test", "val"):
            sub = res[res["split"] == split]
            k = int((sub["gap_seconds"] <= n).sum())
            cells.append(f"{k:>4}/{len(sub)} ({100 * k / max(len(sub), 1):4.1f}%)")
        print(f"{n:>6} | {cells[0]:>14} | {cells[1]:>14}")

    print("\nMedian gap to the closest same-class train photo (seconds):")
    for split in ("test", "val"):
        print(f"  {split}: {res[res['split'] == split]['gap_seconds'].median():.0f}")

    print("\nPer class, test images with a train photo within 10 seconds:")
    sub = res[res["split"] == "test"]
    print(sub.groupby("label")["gap_seconds"].agg(total="count", within_10s=lambda s: int((s <= 10).sum())).to_string())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.sort_values("gap_seconds").to_csv(out, index=False)
    print(f"\nSaved {len(res)} rows to {out}")


if __name__ == "__main__":
    main()