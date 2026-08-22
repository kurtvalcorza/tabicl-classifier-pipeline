# TabICLv2 Classifier — DIMER Pipeline

A DIMER pipeline that fine-tunes [TabICLv2](https://huggingface.co/jingang/TabICL), a pretrained
tabular foundation model, on your own tabular-classification dataset. You supply a table of rows
with one categorical target column. The pipeline validates the table, fine-tunes TabICLv2 on a
GPU, and produces a saved model artifact plus a holdout score.

TabICL is [BSD-3-Clause](https://opensource.org/license/bsd-3-clause) licensed, so the trained
pipeline can be enabled and served without a usage restriction on the model itself. The training
data carries its own licence — see [Data licence governs the served model](#data-licence-governs-the-served-model).

For platform-administrator setup and operations — resource profiles, weights delivery, network
egress, base-model handoff, and the release gate — see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## The model: TabICLv2

TabICL is a tabular foundation model from the [Soda team at Inria](https://team.inria.fr/soda/),
released with open weights under BSD-3-Clause. Like TabPFN and Mitra, it is an **in-context
learner**: it reads a table of labelled examples as context and predicts on new rows, with no
gradient update required for a base prediction. TabICLv2 (arXiv
[2602.11139](https://arxiv.org/abs/2602.11139)) is the successor to the original TabICL (ICML
2025, arXiv [2502.05564](https://arxiv.org/abs/2502.05564)), designed to be faster and to scale
to larger tables.

TabICL's architecture processes a table in two stages — a column-wise pass that builds a
distribution-aware embedding per feature, then a row-wise pass and an in-context head that
attends over the labelled support set to classify each query row. See the
[model card](weights/README.md) once the weights are staged locally, or the upstream
[`jingang/TabICL`](https://huggingface.co/jingang/TabICL) repository, for provenance and license.

### Checkpoints

TabICLv2 ships classifier and regressor checkpoints in a single Hugging Face repository. This
pipeline uses the classifier.

| Checkpoint | Target type | Hugging Face id | File |
|---|---|---|---|
| Classifier (this pipeline) | categorical | [`jingang/TabICL`](https://huggingface.co/jingang/TabICL) | `tabicl-classifier-v2-20260212.ckpt` |
| Regressor | numeric | [`jingang/TabICL`](https://huggingface.co/jingang/TabICL) | `tabicl-regressor-v2-20260212.ckpt` |

The classifier checkpoint is pinned by Hugging Face **revision**
`4dcd344ece2c00be9e831fdd35bed57b5ad83e19` and hard-verified by **SHA-256**
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` at fine-tune time, so every
run starts from identical weights (see [DEPLOYMENT.md → Base model handoff](DEPLOYMENT.md)).

### Applicability

TabICLv2 is designed for tabular classification and supports **many-class** targets. Because it
is an in-context learner, its accuracy depends on whether the features carry signal about the
class. Treat published benchmark results as evidence of strong performance where signal exists,
not as a guarantee on any table. The validator applies an operational ceiling of **1,000
classes** to catch likely high-cardinality mistakes (e.g. an id column selected as the target),
not because the model caps there.

### Fine-tuning is GPU-only

This component is intentionally a **fine-tuner**, not a zero-shot server. It adapts the
pretrained weights to the uploaded table through `tabicl.FinetunedTabICLClassifier`, which
requires CUDA. **If CUDA is unavailable the run fails** with a clear message rather than silently
downgrading a fine-tuning request into zero-shot inference. The effective device and provenance
are recorded in `result.json`.

Because TabICL remains an in-context model *after* fine-tuning, the served artifact carries both
the fine-tuned checkpoint and the training context — see [Outputs](#outputs).

---

## When to use this pipeline

Use this pipeline for tabular classification: predicting a categorical label from a row of
features. Risk tiers, demand bands, churn/no-churn, quality grades, and any row-per-record
categorical prediction fit here. Do not use it for images — for vision tasks use the Image
Classification, Object Detection, or Segmentation pipelines. For a numeric target, use the
[TabICLv2 regressor pipeline](https://github.com/kurtvalcorza/tabicl-regressor-pipeline).

---

## Repositories

The pipeline is two deployable containers, one repository each, plus this umbrella repository for
the contract and docs. Each `Dockerfile` sits at its repository root.

| Component | Repository | Runs on |
|---|---|---|
| Validator | [`tabicl-classifier-dataset-validator`](https://github.com/kurtvalcorza/tabicl-classifier-dataset-validator) | CPU |
| Fine-tuner | [`tabicl-classifier-finetuner`](https://github.com/kurtvalcorza/tabicl-classifier-finetuner) | CUDA GPU |
| Umbrella (this repo) | `tabicl-classifier-pipeline` | Docs + contracts |

```
tabicl-classifier-dataset-validator/    (CPU)
├── Dockerfile
├── validate.py          DIMER-facing entrypoint (delegates to validator.py)
├── validator.py         validation implementation
├── conftest.py
├── requirements.txt
└── tests/

tabicl-classifier-finetuner/            (GPU)
├── Dockerfile           torch 2.8 / CUDA 12.8; bakes the pinned checkpoint
├── train.py             DIMER-facing entrypoint
├── dimer-pipeline.json  preprocessing + fine-tuning fields
├── requirements.txt
└── tests/
```

DIMER builds each repository from its root and launches the container by the portal naming
convention: `validate.py` for the validator and `train.py` for the fine-tuner. The validator's
tested logic lives in `validator.py`; `validate.py` is a thin entrypoint that delegates to it.

Keep `dimer-pipeline.json` at the fine-tuner repository root. It defines the preprocessing and
fine-tuning fields end users see. Without it, the workbench preprocessing step renders empty and
the fine-tuning step stays locked.

The authoritative dataset contract ([`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](TABULAR_CLASSIFICATION_DATASET_SPEC.md))
and deployment notes ([`DEPLOYMENT.md`](DEPLOYMENT.md)) live in this umbrella repository, not in
the container repositories.

---

## Creating the pipeline

Prerequisites: portal access as AI Engineer, and both repositories reachable by the portal's
GitHub App.

1. Open **AI Engineer → New Pipeline** and set these fields:

   | Field | Value |
   |---|---|
   | Pipeline Name | `TabICLv2 Tabular Classification` |
   | Description | `Fine-tune the TabICLv2 tabular foundation model for classification using your own tabular dataset. Supports binary and many-class classification, dataset validation, configurable preprocessing, evaluation, and export of the trained model.` |
   | Task Type | `Custom / Other` |
   | Base Model | `jingang/TabICL` (classifier checkpoint) |
   | Validator repository | `https://github.com/kurtvalcorza/tabicl-classifier-dataset-validator` |
   | Fine-tuner repository | `https://github.com/kurtvalcorza/tabicl-classifier-finetuner` |

2. Build both images (the fine-tuner build needs Hugging Face egress to bake the checkpoint — see
   [DEPLOYMENT.md → Network](DEPLOYMENT.md)).
3. Run the smoke test with a small dataset.
4. Enable the pipeline **only after** the on-platform serving check in the release gate passes.

### Portal implementation notes

- **`Custom / Other` is the correct portal card** for tabular pipelines; DIMER has no native
  tabular task type. The pipeline declares its own task identity: the fine-tuner image sets
  `DIMER_TASK_TYPE=tabular_classification` and relies on that baked fallback, treating any value
  the platform sends as an override.
- **`dimer-pipeline.json` stays at the fine-tuner repository root.** The portal reads it there to
  render the preprocessing and fine-tuning fields.
- **Field-to-runtime mapping.** `datasetPreprocessing` keys are passed to the fine-tuner as
  preprocessing arguments; `modelFinetuning` keys as hyperparameters. Every declared manifest key
  maps one-to-one to runtime behaviour.
- **`model_id` is not declared in `dimer-pipeline.json`.** The DIMER **Base Model** field is
  authoritative for the checkpoint that loads.

---

## The dataset

### Format

A zip of CSV files. The full contract is in
[`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](TABULAR_CLASSIFICATION_DATASET_SPEC.md).

```
dataset.zip
├── train.csv          (required)   one row per example; one categorical target column
├── val.csv            (optional)   same columns as train; a stratified holdout is split if absent
└── test.csv           (optional)   scored after fine-tuning, never used for selection
```

The target column is named `target` by default; change it with the `target_column` preprocessing
field. Its distinct values are the class labels. Every other column, except those listed in
`drop_columns`, is a feature; features may be numeric or categorical. Rows with a missing target
are dropped before the row-count and class checks. Duplicate `train.csv`/`val.csv`/`test.csv`
candidates in the archive are rejected so the validator and fine-tuner cannot resolve different
files.

### How to build a dataset

TabICL consumes a feature table, not raw records. Convert a time series or transaction log
(`entity, date, value`) into a training table by engineering one row per `(entity, date)`:

- **features** — history and context at that point: lags, rolling means and standard deviations,
  calendar fields, and any known covariates (promotions, holidays, weather, stock status);
- **target** — the categorical label to predict, e.g. a demand band a chosen number of days
  ahead, or a binary event flag.

### Row, feature, and class ceilings

The initial DIMER profile caps training at **50,000 rows** (`max_train_rows` range 300–50,000)
and **2,000 features** for operational predictability; the 1,000-class ceiling is a validator
guard, not a model limit. At least **50 usable training rows** and **2 classes** are required,
and — when the pipeline must create its own holdout — every class needs at least two usable rows.
Sampling and holdout creation are **stratified by class**. Archive guards: ≤1 GiB uncompressed
total, ≤512 MiB per CSV, nested zips and path-traversal members rejected. These limits are
overridable by platform environment variables for an intentionally larger profile.

### Data licence governs the served model

The model is BSD-3-Clause, but a served pipeline is also bound by the licence of the data it was
trained on. A model fine-tuned on non-commercial data — for example CC BY-NC — may not be
appropriate to expose as a hosted service. Confirm the licence of any corpus before you enable a
pipeline built from it.

---

## Configurable fields

Preprocessing (`datasetPreprocessing`):

| Field | Default | Purpose |
|---|---|---|
| `target_column` | `target` | Name of the categorical column to predict |
| `drop_columns` | — | Comma-separated columns to exclude from features (ids, raw dates) |
| `max_train_rows` | `10000` | Stratified cap on training rows (range 300–50,000); larger tables are sampled to it |
| `validation_split` | `0.2` | Stratified holdout fraction when the zip has no `val.csv` (range 0.05–0.4) |

Fine-tuning (`modelFinetuning`):

| Field | Default | Purpose |
|---|---|---|
| `time_limit_seconds` | `1800` | Wall-clock fine-tuning budget (range 60–7,200) |
| `seed` | `0` | RNG seed for splitting, sampling, and TabICL; pin it for reproducible runs |
| `epochs` | `30` | Maximum fine-tuning epochs; early stopping may stop sooner (range 1–100) |
| `learning_rate` | `1e-5` | AdamW learning rate (range 1e-7–1e-3) |
| `weight_decay` | `0.01` | AdamW weight decay (range 0–0.1) |
| `patience` | `8` | Non-improving epochs tolerated before early stopping (range 1–30) |
| `eval_metric` | `accuracy` | Validation metric for best-checkpoint selection (`accuracy`, `roc_auc`, `log_loss`) |
| `n_estimators_finetune` | `2` | Ensemble members per fine-tuning meta-batch (range 1–8) |
| `n_estimators_validation` | `2` | Ensemble members for end-of-epoch validation (range 1–8) |
| `n_estimators_inference` | `8` | Ensemble members used by the final fitted classifier (range 1–16) |

---

## Outputs

A successful run writes the trained artifact and a `result.json` describing the run:

```
tabicl_classifier/
├── artifact.json            complete inference contract (schema, inference block, encoders, digests)
├── training_context.parquet the in-context support table used at inference time
└── checkpoints/
    └── best.ckpt            the fine-tuned TabICLv2 checkpoint
```

`result.json` records validation accuracy / log-loss / AUC where available, optional test
metrics, the input **dataset digest**, the `tabicl` package version, the base checkpoint
identifier, and the SHA-256 of the fine-tuned checkpoint. Its provenance fields record the
resolved base-model revision, SHA-256, source (`dimer-provided` / `pinned-baked` /
`pinned-download`), and whether the base matched the pinned default (`baseMatchesPinned`).

TabICL remains an in-context model after downstream fine-tuning, so **inference needs both the
fine-tuned checkpoint and the training context**. A fresh-environment loader reconstructs the
exact scored model from `artifact.json` plus its referenced files, then fits and predicts:

```python
from tabicl import TabICLClassifier
import pandas as pd

ctx = pd.read_parquet("training_context.parquet")
model = TabICLClassifier(model_path="checkpoints/best.ckpt", allow_auto_download=False)
model.fit(ctx.drop(columns=[target_column]), ctx[target_column])
pred = model.predict(X_new)
```

The fine-tuner performs this reload-and-predict smoke check against `best.ckpt` before it reports
success. When the dataset zip includes a `test.csv`, it is scored after fitting and does **not**
influence checkpoint selection.

---

## Reproducibility

TabICL fine-tuning is stochastic: two runs on identical data can differ unless the seed is fixed.
`seed` is a first-class field and seeds the split, the sampling, and TabICL itself. GPU kernel
autotuning can still leave small residual variation, so runs are reproducible in ranking but not
guaranteed byte-identical. Starting from identical weights is guaranteed by the pinned revision +
SHA-256 baked into the image.

---

## Resource profile

Each fine-tuning run executes as a Kubernetes job under a **GPU** profile — there is no CPU
fallback. Start with a single modern CUDA GPU and the conservative default row/feature limits,
then profile representative workloads before increasing `max_train_rows`, the estimator counts,
or `epochs`. TabICL holds the training table in memory as in-context context, so its footprint
grows with rows and features; set the production GPU/RAM profile from measured peak usage during
the release gate (see [DEPLOYMENT.md → Pre-enable checklist](DEPLOYMENT.md)).

The fine-tuner image is built on **`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`**, which ships
sm_120 (Blackwell) kernels alongside sm_70–sm_100. Do not downgrade to cuda12.4/torch2.6 — that
build lacks sm_120 and dies with *"no kernel image is available"* on RTX 50-series / B200 GPUs.

---

## Provenance and traceability

### How this pipeline was authored

The validator, fine-tuner, configuration, and documentation in this repository were drafted with
AI assistance (Anthropic Claude, via Claude Code) and are pending human review before production
deployment. The following were verified by execution, not only generated: both container scripts
byte-compile; `dimer-pipeline.json` validates against the field schema; the validator's unit-test
suite covers its check set (usable-row and class thresholds, stratified split feasibility,
duplicate-split and archive-safety rejection, the wrong-pipeline guidance, and the crash-path
callback/metadata contract); and the base weights' SHA-256 is verified before fitting.

Not yet verified, and requiring human sign-off: the DIMER portal image build, the on-platform GPU
smoke test, the resource-profile request, and the platform's **inference-serving integration**
for this custom TabICL artifact format. Treat the generated code as a reviewed draft, not audited
production code, and do not production-enable until the end-to-end serving check in
[DEPLOYMENT.md](DEPLOYMENT.md) passes on-platform.

### Model lineage

| Field | Value |
|---|---|
| Base model | [`jingang/TabICL`](https://huggingface.co/jingang/TabICL) (classifier checkpoint) |
| Checkpoint file | `tabicl-classifier-v2-20260212.ckpt` |
| Pinned weights revision | `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` |
| Weights SHA-256 | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |
| Licence | BSD-3-Clause |
| Origin | [Qu et al. (2025, ICML)](https://arxiv.org/abs/2502.05564); [TabICLv2 (2026)](https://arxiv.org/abs/2602.11139); Soda team, Inria |
| Framework | `tabicl[finetune]==2.1.1` |

Pinning the revision (fine-tuner `Dockerfile`) makes every run start from identical weights.
Without it, `tabicl` could fetch a moved `main` at runtime and the model could change between
builds.

### Data lineage

A trained model inherits the provenance and licence of the table it was fine-tuned on. Each
dataset should carry its source, its licence, and — for a derived table — the transformation that
produced it. Because the served artifact embeds the training context, treat it with the same
data-governance controls as the source training dataset.

### Per-run record

Every fine-tuning run writes a `result.json` that serves as the run's provenance record: the base
model and its resolved revision/SHA-256/source, the target and dropped columns, the seed, time
budget, and eval metric, the training device, row counts, the resulting scores, the fine-tuned
checkpoint SHA-256, the training-context SHA-256, and the input dataset digest. Paired with the
container image tag, this forms a chain from data to served model.

---

## References

- Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2025).
  [*TabICL: A Tabular Foundation Model for In-Context Learning on Large Data*](https://arxiv.org/abs/2502.05564).
  International Conference on Machine Learning (ICML) 2025.
- Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2026).
  [*TabICLv2: A better, faster, scalable, and open tabular foundation model*](https://arxiv.org/abs/2602.11139).
- [TabICL source code](https://github.com/soda-inria/tabicl), Soda team, Inria (BSD-3-Clause).
- [`jingang/TabICL`](https://huggingface.co/jingang/TabICL) model card and checkpoints, Hugging Face.
