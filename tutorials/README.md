# TabICLv2 Classifier standalone Colab tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/tabicl-classifier-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabicl-classifier-pipeline/blob/main/tutorials/tabiclv2_classifier_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jingang%2FTabICL-ffcc4d?style=flat)](https://huggingface.co/jingang/TabICL)
[![Upstream](https://img.shields.io/badge/Upstream-soda--inria%2Ftabicl-181717?style=flat&logo=github&logoColor=white)](https://github.com/soda-inria/tabicl)
[![arXiv](https://img.shields.io/badge/arXiv-2602.11139-b31b1b.svg)](https://arxiv.org/abs/2602.11139)

These notebooks make the model usable outside DIMER Workbench while preserving the repository's pinned checkpoint identity and serving-artifact contract.

## Notebook specification

These release-grade tutorials conform to **DIMER Notebook Specification v1.0**. The normative profile is declared here and in each notebook's `metadata.dimer.notebook_profile` field.

| Notebook | Profile | Badge | Purpose |
|---|---|---|---|
| [`tabiclv2_classifier_colab.ipynb`](tabiclv2_classifier_colab.ipynb) | `E2E` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabicl-classifier-pipeline/blob/main/tutorials/tabiclv2_classifier_colab.ipynb) | End-to-end tutorial: checkpoint → data → evaluation → optional fine-tuning → inference → portable bundle → fresh reload |
| [`tabiclv2_classifier_artifact_inference_colab.ipynb`](tabiclv2_classifier_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabicl-classifier-pipeline/blob/main/tutorials/tabiclv2_classifier_artifact_inference_colab.ipynb) | Consume a previously produced trusted bundle in a fresh runtime, reconstruct serving state, and score new data without gradient fine-tuning |

## Main tutorial

- exact pinned `tabicl-classifier-v2-20260212.ckpt` acquisition and SHA-256 verification
- DIMER ZIP/direct `.ckpt` of the same pinned base checkpoint, or pinned upstream source; fine-tuned serving bundles use the artifact-inference notebook
- Breast Cancer Wisconsin sample or BYOD CSV/pre-split data
- pretrained in-context evaluation with accuracy, balanced accuracy, weighted F1, log loss, and ROC-AUC where defined
- optional `FinetunedTabICLClassifier` CUDA fine-tuning
- companion classical tree baselines (LightGBM and Random Forest) with holdout leaderboard and device latency
- in-memory post-hoc probability blending with strict label alignment and generalization assessment
- holdout-only artifact selection; independent test is evaluation only
- many-class-compatible `predict()` / `predict_proba()` output
- DIMER-style bundle export: checkpoint + training context + manifest
- fresh reload prediction/probability equivalence smoke test

## Artifact contract

TabICL remains an in-context learner after downstream fine-tuning. A deployable bundle therefore requires at minimum:

```text
checkpoints/best.ckpt
training_context.parquet
artifact.json
```

The inference-only notebook verifies the archive/member bounds, expected file set, internal digests and recorded sizes, reconstructs the support context, and calls the ordinary TabICL estimator with `allow_auto_download=False`.

## Trust boundary

ZIP path checks prevent traversal and symlink extraction, but they do not make model deserialization trustworthy. Load only artifacts you produced yourself or obtained from a trusted source. When a known ZIP SHA-256 is available, compare it before extraction/loading.

The portable artifact also embeds the labelled training context. Treat the exported ZIP with the same dataset licence, access-control, retention, and disclosure requirements as the source training data.

## Runtime and reproducibility

The tutorials pin their directly installed packages. PyTorch is treated as part of the supported runtime substrate rather than reinstalled after import, so the notebooks print the effective Python, PyTorch, TabICL, accelerator, and principal package versions used by each run. Explicit seeds control the tutorial's splits and model-level stochastic settings; CUDA kernels, hardware, library internals, and wall-clock latency can still introduce run-to-run variation, so reproducibility claims do not imply bitwise identity across hardware.

## Release execution evidence

Static/CI checks validate notebook JSON/code structure, pinned checkpoint identity, task semantics, CSV-header safeguards, artifact contract, profile declarations, security markers, badges, and AI provenance. Static validation is not execution evidence.

Pull-request CI also executes the committed default `E2E` path in a clean Jupyter kernel and then executes the `ARTIFACT-INFERENCE` notebook in a **separate fresh kernel**, supplying the exported artifact and a separate CSV through a Colab-compatible upload harness. The workflow uploads the executed notebooks and `notebook-release-evidence.json`, which records the tested revision, environment, package/runtime identity, artifact SHA-256, fresh-kernel boundary, and outcomes. Optional CUDA gradient fine-tuning is not claimed by the CPU release job and still requires an accelerator when explicitly enabled.

A notebook known to fail in current Google Colab is not release-ready even if the clean-kernel CI job passes; Colab-specific failures must be fixed before release.

## AI provenance

These tutorials were developed with substantial AI assistance using **GPT-5.6 Sol High**, via **OpenAI / ChatGPT**, under Agent Relay role **Builder**, with maintainer direction and review. Attribution is provenance, not sign-off or independent verification.
