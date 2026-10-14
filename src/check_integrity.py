import pandas as pd
from PIL import Image

df = pd.read_csv("data/manifest.csv")

bad = []
for p in df["filepath"]:
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.convert("RGB").load()
    except Exception as e:
        bad.append((p, str(e)))

print("Unreadable files:", len(bad))
for p, e in bad:
    print(p, "->", e)

if bad:
    bad_paths = {p for p, _ in bad}
    df = df[~df["filepath"].isin(bad_paths)]
    df.to_csv("data/manifest.csv", index=False)
    print("Removed from manifest. New total:", len(df))