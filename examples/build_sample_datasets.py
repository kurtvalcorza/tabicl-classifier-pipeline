#!/usr/bin/env python3
"""Build and package sample datasets for TabICLv2 Classifier tutorials."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "examples" / "sample-data"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PALMER_SHA256 = "fe894295ccc0a447dc020f5d57798968b95eae3da1a4cbe09a638e6b1bd0ba44"


def build_palmer_penguins() -> Path:
    """Build Palmer Penguins (multiclass classification, 3 species, mixed features).

    Standard Palmer Penguins table contains 344 rows.
    Removing 11 incomplete rows (10 missing measurements + 1 unrecorded sex marker)
    yields exactly 333 complete rows.
    Stratified 60/20/20 partition produces 199 train / 67 val / 67 test.
    """
    print("Fetching Palmer Penguins (OpenML penguins)...")
    dataset = fetch_openml("penguins", version=1, as_frame=True)
    df = dataset.frame.dropna().copy()
    # Filter out unrecorded sex marker ('_')
    df = df[df["sex"].isin(["MALE", "FEMALE"])].reset_index(drop=True)

    assert len(df) == 333, f"Expected 333 rows, got {len(df)}"

    target_col = "species"
    feature_cols = [c for c in df.columns if c != target_col]
    ordered_cols = feature_cols + [target_col]
    df = df[ordered_cols]

    # 60% train, 20% val, 20% test (stratified by species)
    train_df, remainder = train_test_split(
        df, test_size=0.4, random_state=42, stratify=df[target_col]
    )
    val_df, test_df = train_test_split(
        remainder, test_size=0.5, random_state=42, stratify=remainder[target_col]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    assert len(train_df) == 199
    assert len(val_df) == 67
    assert len(test_df) == 67

    out_zip = SAMPLE_DIR / "palmer-penguins.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, part in [("train.csv", train_df), ("val.csv", val_df), ("test.csv", test_df)]:
            info = zipfile.ZipInfo(name, (2026, 9, 7, 23, 15, 42))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, part.to_csv(index=False))

    out_bytes = buf.getvalue()
    out_zip.write_bytes(out_bytes)
    digest = hashlib.sha256(out_bytes).hexdigest()
    assert digest == EXPECTED_PALMER_SHA256, f"Expected {EXPECTED_PALMER_SHA256}, got {digest}"
    print(
        f"[OK] Built {out_zip.name}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test. "
        f"Features: {len(feature_cols)} (SHA-256: {digest})"
    )
    return out_zip


def main() -> None:
    build_palmer_penguins()
    print("All TabICL Classifier sample datasets built successfully.")


if __name__ == "__main__":
    main()
