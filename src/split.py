import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("data/manifest_dedup.csv")

train, rest = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED)
val, test = train_test_split(
    rest, test_size=0.50, stratify=rest["label"], random_state=SEED)

train = train.assign(split="train")
val = val.assign(split="val")
test = test.assign(split="test")

out = pd.concat([train, val, test]).sort_values("filepath").reset_index(drop=True)
out.to_csv("data/manifest.csv", index=False)

print(pd.crosstab(out["label"], out["split"], margins=True))