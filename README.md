# TabICLv2 Classifier — DIMER Pipeline

A DIMER pipeline that fine-tunes **TabICLv2** for tabular classification using `tabicl.FinetunedTabICLClassifier`.

## Repositories

| Component | Repository | Runtime |
|---|---|---|
| Validator | `tabicl-classifier-dataset-validator` | CPU |
| Fine-tuner | `tabicl-classifier-finetuner` | CUDA GPU |
| Umbrella | `tabicl-classifier-pipeline` | Docs + contracts |

This umbrella repository holds the authoritative dataset contract, DIMER field manifest (`dimer-pipeline.json`), and deployment/serving notes. Deployable source lives in the two sibling repositories above.

## Dataset contract

Upload a zip containing:

```text
dataset.zip
├── train.csv      required
├── val.csv        optional; otherwise a stratified holdout is created
└── test.csv       optional; scored after fine-tuning
```

The target column defaults to `target`. All other non-dropped columns are features. Missing-target rows are excluded. Duplicate `train.csv`, `val.csv`, or `test.csv` candidates are rejected so the validator and fine-tuner cannot resolve different files.

The initial DIMER deployment caps training at 50,000 rows and 2,000 features for operational predictability. Sampling and holdout creation are stratified by class.

## Fine-tuning

The pipeline pins `tabicl[finetune]==2.1.1` and the TabICLv2 classifier checkpoint identifier `tabicl-classifier-v2-20260212.ckpt`.

Default training configuration:

- 30 epochs, with early stopping
- AdamW learning rate `1e-5`
- weight decay `0.01`
- 8-epoch patience
- accuracy for best-checkpoint selection
- 2 fine-tune ensemble members, 2 validation members, 8 inference members
- 1,800-second wall-clock budget

This component is intentionally a **fine-tuner**: if CUDA is unavailable, the run fails instead of silently switching to zero-shot inference.

## Artifact

A successful run writes:

```text
tabicl_classifier/
├── artifact.json
├── training_context.parquet
└── checkpoints/
    └── best.ckpt
```

TabICL remains an in-context model after downstream fine-tuning, so inference needs both the fine-tuned checkpoint and the training context. The finetuner reloads `best.ckpt`, re-fits the context, and performs a prediction smoke test before reporting success.

Equivalent reload sequence:

```python
from tabicl import TabICLClassifier

model = TabICLClassifier(model_path="best.ckpt", allow_auto_download=False)
model.fit(X_context, y_context)
pred = model.predict(X_new)
```

## Result metrics

`result.json` records validation accuracy/log-loss/AUC where available, optional test metrics, dataset digest, package version, base checkpoint identifier, and SHA-256 of the fine-tuned checkpoint.

## DIMER setup

Create a **Custom / Other** pipeline using:

- task identity: `tabular_classification`
- validator repo: `tabicl-classifier-dataset-validator`
- fine-tuner repo: `tabicl-classifier-finetuner`
- root `dimer-pipeline.json` from the fine-tuner repo
- GPU resource profile for fine-tuning

The platform still needs an end-to-end inference-serving check for this custom TabICL artifact format before production enablement.

## Upstream

TabICLv2 is the ICML 2026 tabular foundation model from the Soda team at Inria. The upstream implementation is BSD-3-Clause licensed and exposes explicit downstream fine-tuning through `FinetunedTabICLClassifier`.
