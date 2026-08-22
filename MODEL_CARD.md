# TabICLv2 Classifier — DIMER Model Card

## Base model

- Model family: TabICLv2
- Checkpoint identifier: `tabicl-classifier-v2-20260212.ckpt`
- Hugging Face repository: [`jingang/TabICL`](https://huggingface.co/jingang/TabICL)
- Pinned revision: `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`
- Weights SHA-256: `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`
- Task: tabular classification
- Upstream package pinned by this pipeline: `tabicl[finetune]==2.1.1`
- Upstream project license: BSD 3-Clause for TabICL's core tabular implementation

## Adaptation mode

This DIMER pipeline performs real downstream fine-tuning through
`tabicl.FinetunedTabICLClassifier`. The upstream fine-tuning wrapper adapts the pretrained
weights to the uploaded table and supports validation-based best-checkpoint selection using
accuracy, ROC-AUC, or log-loss. TabICLv2 supports **many-class** classification; the fine-tuner
loads it with `support_many_classes=True` and `allow_auto_download=False`. The dataset validator
applies an operational ceiling of 1,000 classes to catch likely high-cardinality mistakes, not
because the model caps there.

Fine-tuning is GPU-only in this DIMER implementation. Lack of CUDA is a run failure rather than
an automatic change to zero-shot inference.

## Artifact

A trained artifact contains both:

1. `checkpoints/best.ckpt` — fine-tuned TabICLv2 weights;
2. `training_context.parquet` — the support/context table used when reloading the in-context
   classifier.

TabICL remains an in-context learner after downstream fine-tuning, so inference reloads the
fine-tuned checkpoint and then fits the inference wrapper on the saved support/context table
before predicting new rows. `artifact.json` is the complete inference contract: target and
feature-column ordering, the `inference` block (`nEstimators`, `randomState`, device,
`allowAutoDownload`, `supportManyClasses`), the persisted `categoricalEncoders`, and component
`digests`. The training job verifies that `best.ckpt` can be loaded by `TabICLClassifier`, fitted
on the saved training context, and used for prediction before declaring success.

## Evaluation

Validation metrics: accuracy, log-loss, and ROC-AUC where available. The best-checkpoint
selection metric is set by `eval_metric` (`accuracy`, `roc_auc`, or `log_loss`; default
`accuracy`).

When `test.csv` is present, it is scored only after fine-tuning and best-checkpoint selection; it
does not influence optimization or early stopping.

## Provenance

Each successful `result.json` records the base checkpoint identifier with its resolved HF
revision, SHA-256, and source (`dimer-provided` / `pinned-baked` / `pinned-download`), whether
the base matched the pinned default (`baseMatchesPinned`), the TabICL version, the uploaded
dataset SHA-256, the fine-tuned checkpoint SHA-256, and the training-context SHA-256. Pinning the
revision + SHA-256 (fine-tuner `Dockerfile`) makes every run start from identical weights.

## Production status

Repository implementation is not equivalent to DIMER production validation. Production enablement
still requires an on-platform GPU build, a fine-tuning smoke test, persistence of both checkpoint
and support context, and end-to-end inference-serving verification — see
[DEPLOYMENT.md](DEPLOYMENT.md).
