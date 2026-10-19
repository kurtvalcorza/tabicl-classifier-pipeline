# ruff: noqa: E401, E501  -- a one-off data build script, kept compact
"""Build the pinned MAGIC Gamma Telescope split zip (row 96, R96-D1 = C).

Split rule (identical to the 2026-10-19 dataset research, so the fresh test is the one hashed there and never scored):
sklearn train_test_split, stratified, seed 0: test (1500) carved FIRST from all 19,020 rows, then val (1000) from the
rest, then train capped at 3000 (stratified); the 13,520 rows left over are the untouched reserve.
The CSV rows are copied as the original text (dtype=str), so no float is re-formatted. The test rows are written,
never fitted, predicted or scored by this script.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SRC_SHA = "a52d3a378e43df7b678c4c0b2f03a66a058bbfeb27c95ccd037a767e7d172984"
SEED = 0
src, out = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "source digest mismatch"
df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
assert df.shape == (19020, 11), df.shape
df = df.rename(columns={"class": "target"})
y = df["target"]
idx = np.arange(len(df))
rest, test = train_test_split(idx, test_size=1500, stratify=y, random_state=SEED)
tr, va = train_test_split(rest, test_size=1000, stratify=y.iloc[rest], random_state=SEED)
tr, reserve = train_test_split(tr, train_size=3000, stratify=y.iloc[tr], random_state=SEED)
parts = {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(test), "reserve": np.sort(reserve)}
sets = {k: set(v.tolist()) for k, v in parts.items()}
for a in sets:
    for b in sets:
        if a < b:
            assert not sets[a] & sets[b], (a, b)
assert sum(map(len, sets.values())) == 19020
ids_sha = {k: hashlib.sha256(",".join(map(str, v.tolist())).encode()).hexdigest() for k, v in parts.items()}
assert ids_sha["test"] == "8847cab4e87c8ff30dd6a27045464f7891b2e14dd39149c5e5648348f190f3e4", ids_sha["test"]
# exact-duplicate feature rows across partitions (near-duplicate check: exact match of all 10 feature strings)
feats = [c for c in df.columns if c != "target"]
key = df[feats].agg(",".join, axis=1)
dup = {}
for a in ("train", "val", "reserve"):
    dup[f"test_vs_{a}"] = int(len(set(key.iloc[parts["test"]]) & set(key.iloc[parts[a]])))
dup["train_vs_val"] = int(len(set(key.iloc[parts["train"]]) & set(key.iloc[parts["val"]])))
dup["within_all_rows"] = int(key.duplicated().sum())
members = {}
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    def put(name, data):
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 19, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, data)
        members[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    for name in ("train", "val", "test"):
        put(f"{name}.csv", df.iloc[parts[name]].to_csv(index=False, lineterminator="\n").encode("utf-8"))
    put("split_row_ids.json", (json.dumps({"source": "UCI 159 static/public/159/data.csv", "source_sha256": SRC_SHA, "row_id": "0-based data row index in the source CSV (header excluded)", "split_rule": "sklearn train_test_split stratified seed 0: test 1500 first, then val 1000, then train capped at 3000; rest = reserve", **{k: v.tolist() for k, v in parts.items()}}, separators=(",", ":")) + "\n").encode())
data = buf.getvalue()
out.write_bytes(data)
rec = {"zip_sha256": hashlib.sha256(data).hexdigest(), "zip_bytes": len(data), "members": members, "row_ids_sha256": ids_sha, "sizes": {k: len(v) for k, v in parts.items()}, "class_counts": {k: y.iloc[v].value_counts().to_dict() for k, v in parts.items()}, "exact_duplicate_feature_rows": dup}
print(json.dumps(rec, indent=1))
