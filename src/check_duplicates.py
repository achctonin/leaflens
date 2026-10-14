import pandas as pd
import imagehash
from PIL import Image

df = pd.read_csv("data/manifest.csv")

hashes = []
for p in df["filepath"]:
    with Image.open(p) as im:
        hashes.append(imagehash.phash(im))
df["phash"] = [str(h) for h in hashes]

# Exact phash matches first (fast)
dups = df[df.duplicated("phash", keep=False)].sort_values("phash")
print("Images sharing an identical hash:", len(dups))
print("Groups:", dups["phash"].nunique())
print(dups.groupby("phash")["label"].nunique().value_counts().rename("labels per group"))
print(dups.head(20))

df.to_csv("data/manifest_with_hash.csv", index=False)