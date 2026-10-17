"""Check for near-duplicate images that cross splits (train / val / test).

Why: if a near-copy of a training image sits in the test set, the test score is inflated.
The earlier dedup removed near-duplicates BEFORE the split; this is a second, independent
check on the final split, using the perceptual hashes already stored in
data/manifest_with_hash.csv (no images are read, no models are used).

Usage (run from the repo root):

    python src/check_leakage.py
    python src/check_leakage.py --max-dist 12 --hash-col phash     # if auto-detection picks the wrong column

Output:
  - a summary printed to the terminal
  - outputs/leakage_pairs.csv with every cross-split pair within --max-dist (open the images and look!)

How to read the distance (Hamming distance between 64-bit perceptual hashes):
  0 to 4   almost certainly the same image or a trivial edit (crop, flip, brightness)
  5 to 10  often the same leaf photographed again; check these by eye
  above 10 usually different images
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PAIRS = [("train", "test"), ("train", "val"), ("val", "test")]


def norm(path):
    return str(path).replace("\\", "/").strip()


def hex_to_bits(h):
    """Hash string (hex, as printed by imagehash) -> array of 0/1 values."""
    s = str(h).strip()
    n = int(s, 16)
    nbits = len(s) * 4
    return np.array([(n >> i) & 1 for i in range(nbits - 1, -1, -1)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Check for near-duplicates across splits")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.csv"))
    ap.add_argument("--hashes", default=str(ROOT / "data" / "manifest_with_hash.csv"))
    ap.add_argument("--hash-col", default=None, help="name of the hash column (auto-detected if omitted)")
    ap.add_argument("--max-dist", type=int, default=10, help="report pairs up to this Hamming distance")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "leakage_pairs.csv"))
    args = ap.parse_args()

    m = pd.read_csv(args.manifest, dtype=str)
    h = pd.read_csv(args.hashes, dtype=str)

    path_col = "filepath" if "filepath" in h.columns else h.columns[0]
    hash_col = args.hash_col or next((c for c in h.columns if "hash" in c.lower()), None)
    if hash_col is None:
        raise SystemExit(f"Could not find a hash column. Columns are {list(h.columns)}. Use --hash-col NAME.")
    print(f"Using path column '{path_col}' and hash column '{hash_col}' from {args.hashes}")

    m["key"] = m["filepath"].map(norm)
    h["key"] = h[path_col].map(norm)
    df = m.merge(h[["key", hash_col]].drop_duplicates("key"), on="key", how="left")
    missing = int(df[hash_col].isna().sum())
    print(f"Manifest rows: {len(m)}   without a hash: {missing}")
    if missing:
        print("  (these rows are skipped; check that the file paths in both CSVs match)")
    df = df.dropna(subset=[hash_col]).reset_index(drop=True)

    try:
        bits = np.stack([hex_to_bits(x) for x in df[hash_col]])
    except ValueError:
        raise SystemExit(f"Hash values in '{hash_col}' are not hex strings (example: {df[hash_col].iloc[0]!r}). "
                         "This script expects the hex form printed by imagehash.")
    ones = bits.sum(axis=1)
    print(f"Hash length: {bits.shape[1]} bits\n")

    rows = []
    for a, b in SPLIT_PAIRS:
        ia = np.where(df["split"] == a)[0]
        ib = np.where(df["split"] == b)[0]
        # Hamming distance between every image in split a and every image in split b
        dist = np.rint(ones[ia][:, None] + ones[ib][None, :] - 2 * bits[ia] @ bits[ib].T).astype(int)
        ii, jj = np.where(dist <= args.max_dist)
        for i, j in zip(ii, jj):
            rows.append(
                {
                    "split_a": a,
                    "split_b": b,
                    "file_a": df.at[ia[i], "filepath"],
                    "file_b": df.at[ib[j], "filepath"],
                    "label_a": df.at[ia[i], "label"],
                    "label_b": df.at[ib[j], "label"],
                    "distance": int(dist[i, j]),
                }
            )
    pairs = pd.DataFrame(rows).sort_values("distance") if rows else pd.DataFrame(
        columns=["split_a", "split_b", "file_a", "file_b", "label_a", "label_b", "distance"]
    )

    # ---- summary: how many images in the later split have a twin in the earlier one
    print("Images that have a near-twin in another split (Hamming distance <= t):")
    print(f"{'t':>3} | {'test imgs w/ train twin':>24} | {'val imgs w/ train twin':>23} | {'test imgs w/ val twin':>22}")
    for t in sorted({0, 2, 4, 6, 8, args.max_dist}):
        if t > args.max_dist:
            continue
        sel = pairs[pairs["distance"] <= t]
        n_tt = sel[(sel.split_a == "train") & (sel.split_b == "test")]["file_b"].nunique()
        n_tv = sel[(sel.split_a == "train") & (sel.split_b == "val")]["file_b"].nunique()
        n_vt = sel[(sel.split_a == "val") & (sel.split_b == "test")]["file_b"].nunique()
        print(f"{t:>3} | {n_tt:>24} | {n_tv:>23} | {n_vt:>22}")

    print(f"\nTotal test images: {(df.split == 'test').sum()}   val images: {(df.split == 'val').sum()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out, index=False)
    print(f"\nSaved {len(pairs)} cross-split pairs to {out}")
    if len(pairs):
        print("Closest pairs:")
        print(pairs.head(10)[["split_a", "split_b", "file_a", "file_b", "distance"]].to_string(index=False))
        print("\nOpen a few of the closest pairs and compare the images by eye.")
    else:
        print(f"No cross-split pairs within distance {args.max_dist}.")


if __name__ == "__main__":
    main()