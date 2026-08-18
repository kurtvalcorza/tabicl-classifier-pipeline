# TabICLv2 Classification Dataset Specification

## Files

A dataset is a directory or zip containing exactly one `train.csv` and optionally one `val.csv` and one `test.csv`. Duplicate candidates are rejected.

All present splits must contain the same columns before configured exclusions.

## Target

- Default target column: `target`
- Missing targets are dropped before row-count and class checks.
- At least 50 usable training rows are required.
- At least two classes are required.
- When the pipeline must create its own validation split, every class must have at least two usable rows.

TabICLv2 supports many-class classification; the validator uses an operational ceiling of 1,000 classes to catch likely high-cardinality mistakes.

## Features

Every column except the target and configured `drop_columns` is passed to TabICL. The initial DIMER profile permits up to 2,000 features.

TabICL accepts pandas DataFrames and performs built-in handling for numeric and categorical columns, missing values, scaling, and ensemble feature shuffling.

## Splitting and sampling

If `val.csv` is absent, the finetuner creates a deterministic stratified holdout using `validation_split` and `seed`.

If training rows exceed `max_train_rows`, the finetuner performs deterministic stratified sampling to the configured cap. The DIMER field range is 300–50,000 rows.

`test.csv`, when supplied, is not used for optimization or early stopping. It is scored only after fine-tuning.

## Archive safety

Default operational guards:

- maximum total uncompressed dataset archive: 1 GiB
- maximum single CSV: 512 MiB
- nested zip files rejected
- path traversal archive members rejected

The limits may be overridden by platform environment variables when the deployment profile is intentionally larger.
