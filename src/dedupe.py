import sys
import numpy as np
import pandas as pd

TH = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # max hash distance to count as duplicate

df = pd.read_csv("data/manifest_with_hash.csv")
labels = df["label"].values

# 64-bit phash -> +1/-1 vector; hamming = (64 - dot) / 2
bits = np.array([[int(b) for b in format(int(h, 16), "064b")] for h in df["phash"]],
                dtype=np.float32)
s = bits * 2 - 1
dist = ((64 - s @ s.T) / 2).astype(int)

def group_ids(th):
    parent = list(range(len(df)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    ii, jj = np.where(np.triu(dist <= th, k=1))
    for i, j in zip(ii, jj):
        if labels[i] == labels[j]:          # only merge within the same class
            parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(df))])

# Overview at several thresholds
print("threshold | images that would be removed | cross-label pairs")
for th in (0, 2, 4, 6, 8):
    g = group_ids(th)
    removed = len(df) - len(set(g))
    ii, jj = np.where(np.triu(dist <= th, k=1))
    cross = int((labels[ii] != labels[jj]).sum())
    print(f"{th:>9} | {removed:>28} | {cross}")

# Apply the chosen threshold
g = group_ids(TH)
df["group"] = g
keep = ~df.duplicated("group", keep="first")
kept, dropped = df[keep], df[~keep]

kept[["filepath", "label"]].to_csv("data/manifest_dedup.csv", index=False)
dropped[["filepath", "label"]].to_csv("data/duplicates_removed.csv", index=False)

print(f"\nUsing threshold {TH}: kept {len(kept)}, removed {len(dropped)}")
print(kept["label"].value_counts())
