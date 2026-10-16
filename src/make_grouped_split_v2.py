import re
import numpy as np
import pandas as pd

SEED = 42
GAP = 10  # seconds between consecutive shots that still count as one group
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}

df = pd.read_csv("data/manifest_dedup.csv")   # must be the threshold-2 version (3706 rows)
print("Rows:", len(df))

pat = re.compile(r"(\d{8})[_-](\d{6})")
def get_ts(path):
    m = pat.search(path.split("/")[-1])
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S", errors="coerce")

df["ts"] = df["filepath"].map(get_ts)
print("Images without a timestamp:", df["ts"].isna().sum())

# 1) Build groups within each class
group, gid = {}, 0
for label, sub in df.groupby("label"):
    dated = sub.dropna(subset=["ts"]).sort_values("ts")
    prev = None
    for idx, t in zip(dated.index, dated["ts"]):
        if prev is None or (t - prev).total_seconds() > GAP:
            gid += 1
        group[idx] = gid
        prev = t
    for idx in sub.index[sub["ts"].isna()]:
        gid += 1
        group[idx] = gid
df["group"] = df.index.map(group)

sizes = df.groupby("group").size()
print("Groups:", len(sizes), "| largest:", sizes.max())
print(df.groupby("label")["group"].apply(lambda s: s.value_counts().max()).rename("largest group per class"))

# 2) Allocate groups to splits per class, largest first, to the split furthest below target
df["split"] = ""
for label, sub in df.groupby("label"):
    g = sub.groupby("group").size()
    order = g.sample(frac=1, random_state=SEED).sort_values(ascending=False, kind="stable")
    n = len(sub)
    have = {k: 0 for k in TARGET}
    for g_id, size in order.items():
        deficit = {k: TARGET[k] * n - have[k] for k in TARGET}
        best = max(deficit, key=deficit.get)
        df.loc[df["group"] == g_id, "split"] = best
        have[best] += size

assert (df.groupby("group")["split"].nunique() == 1).all(), "A group spans splits!"
assert (df["split"] != "").all()

print(pd.crosstab(df["label"], df["split"], margins=True)[["train", "val", "test", "All"]])
df[["filepath", "label", "split"]].to_csv(
    "data/manifest_grouped_10s.csv",
    index=False
)