from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
EXTS = {".jpg", ".jpeg", ".png"}

# Find the folder that directly contains the class folders
# (handles the case where the zip extracted into one extra folder)
root = RAW
subdirs = [d for d in root.iterdir() if d.is_dir()]
if len(subdirs) == 1:
    root = subdirs[0]

rows = []
for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
    for f in sorted(class_dir.iterdir()):
        if f.suffix.lower() in EXTS:
            rows.append({"filepath": f.as_posix(), "label": class_dir.name})

df = pd.DataFrame(rows)
df.to_csv("data/manifest.csv", index=False)

print(df["label"].value_counts())
print("Total:", len(df), "| Classes:", df["label"].nunique())