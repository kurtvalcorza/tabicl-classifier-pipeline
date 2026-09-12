---
license: bsd-3-clause
model_card_spec: "1.0"
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - tabular-foundation-model
  - in-context-learning
  - tabicl
base_model: jingang/TabICL
---

# TabICLv2 Classifier

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jingang%2FTabICL-ffcc4d?style=flat)](https://huggingface.co/jingang/TabICL)
[![GitHub](https://img.shields.io/badge/GitHub-soda--inria%2Ftabicl-181717?style=flat&logo=github&logoColor=white)](https://github.com/soda-inria/tabicl)
[![arXiv](https://img.shields.io/badge/arXiv-2602.11139-b31b1b.svg)](https://arxiv.org/abs/2602.11139)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


###### Description

TabICLv2 Classifier packages the `tabicl-classifier-v2-20260212.ckpt` checkpoint from `jingang/TabICL` at Hugging Face revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`, a pretrained tabular foundation model developed by Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan of the Soda team at Inria, run through the `tabicl==2.1.1` reference implementation. The model is a three-stage Transformer for tables — a column-wise encoder that embeds each feature distribution, a row-wise encoder that builds one representation per observation, and a dataset-wise in-context-learning Transformer that attends from the labelled support rows to the query rows and reads off class logits. Version 2 introduces a redesigned synthetic-data prior and long-context improvements over the original TabICL; it was pretrained on synthetic classification tasks with up to 10 classes and supports more classes downstream through mixed-radix ensembling.

At inference the model conditions on the labelled training table as in-context support and emits a class-probability vector per query row. Adaptation happens through in-context conditioning by default and, in this pipeline, through gradient fine-tuning on the operator's table (`tabicl-classifier-finetuner/train.py`, `early_stopping=True`, learning rate 1e-5 by default). What this repository adds is the DIMER composition around those weights: the pipeline contract (`dimer-pipeline.json`, `DIMER_CONTRACT.md`), the dataset specification, the Colab artifact-inference tutorial, and the release conformance record; the validator and fine-tuner workers it composes live in the sibling `tabicl-classifier-dataset-validator` and `tabicl-classifier-finetuner` repositories. The upstream checkpoint is not modified by this repository.

#### Intended Use and Limitations

The use cases below are the ones envisioned during development; the limits are the ones the workers enforce.

###### Primary Intended Uses

Supervised classification of tabular data where each observation is one row of numerical and categorical predictor columns and one categorical target. The pipeline takes a `train.csv` (optionally `val.csv`/`test.csv`) with a declared target column and produces a fine-tuned TabICLv2 artifact, per-row class labels and class probabilities, and holdout metrics.

Concrete application domains envisioned during development: binary, multiclass, and many-class classification for risk categorisation, quality or condition grading, event and fault classification, churn and demand-class prediction, and scientific classification represented as feature tables, where the operator wants strong performance with little or no task-specific hyperparameter search. The pipeline is meant to play the role of a strong zero-configuration baseline or a fine-tuned production model inside DIMER. Enforced ceilings: at most 2,000 feature columns (`MAX_FEATURES`), a stratified training cap of `max_train_rows` between 300 and 50,000 (default 10,000), and at most 1,000 classes (`MAX_REASONABLE_CLASSES`). Pretraining covered roughly 48,000 rows and 100 features, so larger tables are extrapolation the upstream authors report works but this pipeline does not separately validate.

###### Primary Intended Users

Machine-learning researchers, data scientists, machine-learning engineers, software developers, and scientific researchers building predictive systems from structured datasets. The envisioned deployment setting is internal enterprise or research use through the DIMER platform, where the fine-tuner runs as a CUDA worker and the validator as a CPU worker — not a public-facing service.

The pipeline assumes its users understand dataset provenance, holdout evaluation, leakage, class imbalance, and distribution shift, and know that `predict()` is an argmax over class probabilities that have not been calibrated for their domain, that a holdout metric on a few hundred rows has wide variance, and that fine-tuning needs a CUDA GPU and will fail without one rather than fall back. A user who cannot tell a stratified holdout from an in-sample score is outside the assumed competency.

###### Out-of-scope use cases

- **Capability boundaries:** regression or continuous-value prediction (the sibling `tabicl-regressor-pipeline` does that); image, audio, video, or natural-language inputs; unsupervised clustering; causal-effect estimation; generative modelling; raw time-series forecasting without tabular feature construction.
- **Input boundaries:** more than 2,000 feature columns (refused); fewer than 50 usable training rows (`MIN_TRAIN_ROWS`) or fewer than 10 evaluation rows (`MIN_EVAL_ROWS`); a single class, or more than 1,000 classes (refused); any class with fewer than 2 rows when a stratified split is needed (refused); archives over 1 GiB uncompressed or a single CSV over 512 MiB (refused); fine-tuning on a host without CUDA (refused with a clear error).
- **Decision boundaries:** autonomous high-impact decisions — health, safety, criminal justice, credit, employment, housing — without application-specific validation and a human decision-maker; treating upstream benchmark rankings as a guarantee on a new dataset.

#### Factors

TabICLv2's behaviour varies with the structure of the table it is given, not with a physical capture condition; the three subsections below say what that means for groups, instruments, and environment.

###### Groups

This pipeline is not human-centric by construction: TabICLv2 was pretrained on synthetic tasks rather than any fixed human population, so no demographic group — age, sex, gender, ethnicity, disability, nationality, socioeconomic status — is an intrinsic evaluation group of the foundation model, and the pretraining corpus is not group-audited because it contains no people. Demographic fairness or subgroup parity has therefore **not** been established for the checkpoint, and the pipeline measures no subgroup metric.

Where the operator's downstream table describes people, the obligation transfers to the operator: identify the relevant groups in their own data, compute per-group accuracy, log loss, and ROC-AUC on the holdout split, and check for disparate error rates before deployment. The validator result records `classNames` and row counts, not any demographic structure.

###### Instrumentation

TabICLv2 consumes an abstract tabular representation rather than a raw sensor stream; the upstream pretraining did not depend on any real acquisition hardware. The instrument does not disappear because a table sits between it and the model: the operator's rows are produced by whatever systems fed the CSV — transactional databases, ETL pipelines, sensors, survey instruments — and their sampling rate, resolution, calibration, and encoding of missing values determine feature quality.

Instrument error reaches the model as feature error. Drift, miscalibration, or a changed collection procedure between training and inference is not detectable by this pipeline; the validator checks schema, row and class counts, and archive safety, not whether a column's meaning has changed. Operators should document the instrumentation of downstream datasets separately.

###### Environment

**Operating environment.** Fine-tuning is GPU-only: the fine-tuner normalises `DIMER_TRAIN_DEVICE` to a CUDA device and raises if CUDA is unavailable — there is no CPU fine-tune path. Inference on the saved artifact runs on CPU or GPU through `tabicl==2.1.1`, batched at `PREDICT_BATCH_ROWS` to bound memory. The validator is CPU-only. Precision follows the reference implementation's defaults.

**Data environment.** The reported behaviour assumes the inference rows are drawn from the same distribution as the training table: same feature semantics, same encoding, same class prevalence. Performance degrades, without warning from the pipeline, under geographic, temporal, institutional, or population shift, and with feature count beyond the roughly 100 features and 48,000 rows of the pretraining regime. Robustness to arbitrary distribution shift has not been established.

#### Metrics

Metrics are chosen for a probabilistic multiclass classifier whose intended use spans balanced, imbalanced, and many-class tables.

###### Performance Measures

The fine-tuner scores the fine-tuned model on the held-out split in `_classification_metrics` (`tabicl-classifier-finetuner/train.py`) and writes `accuracy`, `logLoss`, and — for binary targets — `rocAuc`, or `rocAucOvr` for multiclass, into the result artifact, with `rocAucError` recorded when AUC is undefined (a class absent from the holdout). The headline metric is the DIMER hyperparameter `eval_metric` (default `accuracy`).

Why these: accuracy captures discrete correctness and is the right summary when classes are balanced and error costs similar; log loss captures probability quality and penalises confident mistakes, which matters whenever the class probabilities are used operationally; ROC-AUC captures ranking quality independent of any threshold and is the informative one for imbalanced problems, with the one-vs-rest form extending it to many classes. Reading only accuracy hides both calibration and imbalance failures, which is why all three are written. Upstream, the authors report that untuned TabICLv2 surpasses RealTabPFN-2.5 on TabArena and TALENT with accuracy as the principal metric; that is a published relative ranking, not a number this pipeline measures or claims.

###### Decision thresholds

The default decision rule is an implicit `argmax`: `predict()` returns the class with the highest predicted probability, and this pipeline ships that rule unchanged. No acceptance threshold on accuracy, log loss, or ROC-AUC was set during development, because the pipeline is domain-agnostic and the tolerable error rate is a property of the deployment.

No probability cutoff is applied, and none is shipped, because the emitted probabilities are not calibrated for the operator's domain. Calibrating and thresholding are the deployment owner's responsibility: set the operating point from the asymmetric cost of false positives against false negatives and the class prevalence on held-out data, and revisit it when either changes. For a screening use where a missed positive is the expensive error, the threshold on the positive-class probability belongs below 0.5; where a false alarm is expensive, above it.

###### Approaches to uncertainty and variability

The pipeline's reported metrics come from a single stratified holdout split of the operator's table (`validation_split`, default 0.2, used when `val.csv` is absent). No dispersion is reported alongside the point value: one split, one fine-tuning run, no confidence interval. Operators who need one should repeat the run across seeds or use cross-validation on their own side.

Sources of run-to-run variability: the stratified holdout, the stratified training cap, TabICL's internal feature-permutation ensembling (`n_estimators_finetune`, `n_estimators_validation`, `n_estimators_inference`, defaults 2/2/8), and gradient fine-tuning with early stopping; all are driven by the DIMER `seed` hyperparameter, which the fine-tuner passes to the split, the cap, and TabICL's `random_state`. Non-deterministic CUDA kernels can still produce small run-to-run differences. The class probabilities are raw ensemble-averaged softmax outputs and have not been calibrated; a caller who needs calibrated probabilities must fit a calibrator on their own holdout data.

#### Ethical considerations and biases

No external ethics board reviewed this pipeline, and no clearance testing with a specific group took place; the subsections record what the developers considered and what the repository actually does.

###### Data

TabICLv2 was pretrained exclusively on synthetic tasks, so the pretraining data do not consist of personally identifiable, health, biometric, financial, or classified records — known from the upstream disclosure, which ends at the description of the synthetic prior; the generated tables are not published. What this repository distributes: the pipeline contract, dataset specification, the Colab tutorial, and conformance records; it does **not** distribute the checkpoint (the fine-tuner downloads it at the pinned revision and verifies its SHA-256 before use, and `weights/` is gitignored) and ships no sample data of its own.

Operators may fine-tune or evaluate TabICLv2 on sensitive real-world tables. The pipeline does not audit the operator's data for personal, sensitive, or proprietary attributes — the validator checks structure, not content — so the legality, privacy, consent, access control, and governance of downstream data remain with the application developer and data owner.

###### Human Life

The pipeline is not intended for decisions in health care, physical safety, criminal justice, legal rights, employment, credit, insurance, education access, or public benefits, and it has not been validated for any of them. The only validation performed is the contract and notebook conformance testing in `scripts/` and the DIMER holdout evaluation on the operator's own table; no clinical, regulatory, or independent domain validation has been carried out by the developers or by any external body, and upstream benchmark rankings are not evidence of suitability.

Where such a use is foreseeable — a triage classifier on a clinical feature table, for example — it would be admissible only with independent domain validation on that operator's population, a human decision-maker between the prediction and the action, subgroup evaluation, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented in the composed workers, each inspectable in the named code:

- **Supply-chain integrity:** the base checkpoint is downloaded with `hf_hub_download(..., revision=BASE_MODEL_REVISION)` at `4dcd344e…`, its SHA-256 is computed and compared with `BASE_MODEL_SHA256` (`bdc7dbd5…`), and a mismatch raises unless the checkpoint was DIMER-provided, in which case the digest and `matches_pinned: false` are recorded in provenance rather than enforced; `tabicl` is pinned to 2.1.1.
- **Input integrity:** the validator rejects archives over 1 GiB uncompressed, fewer than 50 training or 10 evaluation rows, more than 2,000 features, a single class, or more than 1,000 classes; the fine-tuner re-applies the feature limit, refuses a single CSV over 512 MiB, and refuses a stratified split when any class has fewer than 2 rows.
- **Statistical mitigations:** the training cap is stratified so every class survives, and a cap below the class count is raised as an error rather than silently dropping classes; prediction is batched to bound memory.
- **Reproducibility:** `seed` drives the split, the cap, and TabICL's `random_state`; the result artifact records the checkpoint digest, revision, source, dataset digest, and the effective hyperparameters.
- **Refusals:** fine-tuning refuses any non-CUDA device rather than silently training on CPU; the validator writes `classNames` on every result because DIMER requires it.

###### Risks and harms

- **Overconfidence outside the training distribution** (model-intrinsic): the probabilities are uncalibrated and carry no out-of-distribution signal; borne by whoever the operator's decision affects; likely under normal use as the deployment drifts; magnitude set by what the classification gates.
- **Amplification of input bias** (model-intrinsic): a table whose labels encode a historical disparity yields a classifier that reproduces it; borne by the data subjects in the disadvantaged group; realised whenever such a table is used without subgroup evaluation.
- **Many-class decomposition error** (model-intrinsic): above 10 classes the mixed-radix ensemble combines several sub-problems, and errors in one view propagate to the final label; borne by the operator with a many-class target.
- **Small-sample variance** (model-intrinsic): a holdout metric on a few hundred rows can move by several points between seeds; borne by the operator who ships on one split.
- **Automation bias** (use-context): a numerically precise probability displaces human judgement; borne by the data subject; likely wherever the score is surfaced without its uncertainty.
- **Undetected leakage** (use-context): a feature derived from the target inflates the holdout score and collapses in production; the validator does not detect it; borne by the operator and downstream users.
- **Benchmark over-generalisation** (use-context): reading the upstream ranking over RealTabPFN-2.5 as an expected accuracy; borne by whoever sets expectations from it.

###### Use cases

Distinct from the capability and decision boundaries listed under *Out-of-scope use cases*, the developers consider the following uses prohibited even where the model would produce a numerically plausible label:

- surveillance, biometric or demographic profiling, or social scoring of individuals;
- unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access, including classification on a target that proxies a protected attribute;
- deceptive, manipulative, or predatory applications, including presenting an uncalibrated class probability as a certified risk estimate;
- criminal-justice, medical-diagnosis, or legal-rights determinations without the validation and oversight described under *Human Life*;
- any use that violates the BSD-3-Clause terms of the upstream `jingang/TabICL` checkpoint and `tabicl` code, or the terms of the DIMER deployment.

---

## Model Details

- **Model name:** TabICLv2 Classifier
- **Model family:** TabICLv2
- **Checkpoint identifier:** `tabicl-classifier-v2-20260212.ckpt`
- **Code repository:** [soda-inria/tabicl](https://github.com/soda-inria/tabicl)
- **Hugging Face repository:** [jingang/TabICL](https://huggingface.co/jingang/TabICL)
- **Developers:** Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan; Soda team, Inria
- **Task:** tabular classification
- **Learning paradigm:** in-context learning
- **Checkpoint version date:** 12 February 2026
- **Associated paper first posted:** 11 February 2026
- **Reference implementation:** `tabicl`
- **Known-compatible library version used by this repository:** `tabicl[finetune]==2.1.1`
- **License:** BSD 3-Clause

## Checkpoint Provenance

This card documents the following released TabICLv2 classification checkpoint.

- **Checkpoint:** `tabicl-classifier-v2-20260212.ckpt`
- **Hugging Face repository:** [`jingang/TabICL`](https://huggingface.co/jingang/TabICL)
- **Pinned Hugging Face revision:** `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`
- **Direct download URL:** [`https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt`](https://huggingface.co/jingang/TabICL/resolve/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt)
- **SHA-256:** `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`

The checkpoint hash and revision should be retained when exact model-version provenance is required. The classifier and regressor are separate pretrained checkpoints and should not be treated as interchangeable.

## Input

The model operates on a supervised tabular task consisting conceptually of:

- a labelled support or training table;
- feature columns describing each observation;
- categorical target labels for the support examples; and
- one or more unseen rows requiring prediction.

Features may represent numerical or categorical variables through the preprocessing provided by the reference implementation.

TabICL differs from conventional estimators in that the labelled support examples remain part of the inference context. Calling `fit()` primarily establishes this support context; task adaptation occurs during the forward-pass prediction process.

## Output

TabICLv2 Classifier produces categorical predictions for unseen rows. Depending on the estimator interface, outputs may include predicted class labels and class-probability estimates.

For problems containing more than 10 classes, TabICLv2 can use mixed-radix ensembling to construct several simplified classification views, each containing no more than 10 classes, and combine them into predictions for the original class space.

The model's internal 10-class embeddings and prediction head therefore **do not represent a hard downstream limit of 10 classes**.

## Architecture

TabICLv2 processes a table through three principal Transformer stages:

1. **Column-wise embedding**
2. **Row-wise interaction**
3. **Dataset-wise in-context learning**

Its asymptotic runtime complexity for a table with `n` rows and `m` columns is approximately `O(n² + n m²)`.

### Repeated Feature Grouping

Instead of embedding each column entirely independently, TabICLv2 repeatedly groups related feature positions using circular shifts. The published configuration uses the feature-position pattern `(0, 1, 3)`. This preserves the effective number of features while giving the model multiple local views of feature relationships.

### Target-Aware Embedding

Target information from labelled support rows is injected before the final ICL stage. For classification, target-aware embeddings are learned lookup embeddings for up to 10 classes. This allows the earlier representation stages to condition directly on the relationship between features and labels.

### Column-Wise Transformer

- 3 induced self-attention blocks
- 128 inducing vectors
- model dimension: 128
- 8 attention heads

### Row-Wise Transformer

- 3 Transformer layers
- model dimension: 128
- 8 attention heads
- 4 learnable `[CLS]` tokens

The four `[CLS]` outputs are concatenated to produce the row representation.

### Dataset-Wise ICL Transformer

- 12 Transformer layers
- model dimension: 512
- 8 attention heads

Test examples attend to the labelled context and use it to generate task-specific predictions.

### Prediction Head

The classification prediction head is a two-layer MLP with hidden dimension 1024 and output dimension 10. The output dimension reflects the native classification subproblem size, not the maximum number of classes that can be handled by the full inference system.

### Other Architectural Characteristics

The classifier uses pre-norm LayerNorm with learnable weights and biases, GELU activations, feed-forward expansion factor of 2×, rotary positional embeddings in the row Transformer, Query-Aware Scalable Softmax (QASSMax) in key attention stages, and standard residual initialization rather than TabICLv1's zero-initialized residual branches.

QASSMax is designed to preserve useful attention sharpness as the number of context examples increases and contributes to TabICLv2's ability to generalize beyond the sequence lengths used during pretraining.

## Many-Class Classification

TabICLv2 is pretrained on classification problems containing at most 10 classes. To support larger class spaces, it introduces **mixed-radix ensembling**.

For a problem with more than 10 classes, the original label is decomposed into multiple lower-cardinality views. Each view contains at most 10 possible values and can therefore be handled by the pretrained classifier. The resulting view predictions are recombined to recover the original many-class prediction.

This mechanism enables classification beyond the native 10-class pretraining regime without retraining the foundation model itself. Performance on very high-cardinality problems remains application dependent and should be evaluated separately.

## Pretraining Data

TabICLv2 was pretrained entirely on **synthetically generated tabular datasets**.

The synthetic-data generator is based on a graph-structured structural causal model prior and produces diverse combinations of numerical variables, categorical variables, nonlinear relationships, neural-network-derived functions, tree-based functions, discretization functions, random graph structures, feature interactions, and different target-generation mechanisms.

Approximately **35 million synthetic datasets** are processed across the three-stage pretraining curriculum. No real-world benchmark datasets are used as the model's main pretraining corpus.

Synthetic pretraining reduces direct exposure to identifiable real-world records but does not imply that downstream use is free of privacy, bias, or fairness risks.

## Pretraining Procedure

TabICLv2 uses a three-stage curriculum that progressively increases dataset size.

### Stage 1

- 500,000 steps
- 1,024 samples per synthetic dataset
- approximately 30–90% assigned to the training context
- maximum learning rate: `8e-4`

### Stage 2

- 40,000 steps
- 400–10,240 samples per dataset, sampled approximately log-uniformly
- approximately 80% training context
- maximum learning rate: `1e-4`

### Stage 3

- 10,000 steps
- 400–60,000 samples per dataset, sampled approximately log-uniformly
- approximately 80% training context
- maximum learning rate: `2e-5`

Common characteristics include batch size 64, up to 100 features, Muon optimizer, cosine learning-rate scheduling, gradient clipping, automatic mixed precision, and eight attention heads throughout the principal attention modules.

The paper reports approximately **24.5 H100 GPU-days of pretraining compute per model** across the three stages.

## In-Context Learning and Fine-Tuning

Zero-shot or untuned in-context learning is the default TabICL operating paradigm. The support dataset is supplied as context, and the pretrained Transformer performs learning as part of prediction.

TabICLv2 additionally supports explicit downstream fine-tuning through the upstream fine-tuning implementation. A fine-tuned derivative should be treated as a separate model version because its parameters have been adapted to a particular downstream dataset.

Published TabICLv2 benchmark claims for the base model should not automatically be attributed to downstream fine-tuned derivatives, and vice versa.

## Evaluation

TabICLv2 was evaluated by its developers on major real-world tabular benchmark collections including **TabArena** and **TALENT**.

The authors report that the untuned TabICLv2 model surpasses RealTabPFN-2.5 on these evaluations despite RealTabPFN-2.5 using substantial downstream tuning, ensembling, and fine-tuning.

For TALENT classification tasks, **accuracy** is used as the principal classification metric. Supplementary metrics include ROC-AUC, log loss, and ranking-based aggregate measures.

The paper separately evaluates many-class classification datasets containing more than 10 classes and reports strong performance from TabICLv2's native mixed-radix approach.

There is no single intrinsic "accuracy percentage" that applies to all uses of this foundation model. Performance varies materially by dataset.

## Scalability

TabICLv2 was designed to improve long-context scalability over TabICL and contemporary tabular foundation models.

The official implementation reports good empirical performance on datasets ranging from hundreds to tens of thousands of training observations and useful generalization to substantially larger tables.

The authors also demonstrate million-scale inference using Query-Aware Scalable Softmax, selective attention computation, CPU offloading, and disk offloading.

The paper reports processing a table with approximately one million samples and 500 features within about 450 seconds under the specified high-end hardware and offloading setup. These results demonstrate technical scalability, not a guaranteed runtime or accuracy level for arbitrary infrastructure.

## Reproducibility

### Exact Checkpoint Identity

For strict model-version reproduction, preserve:

- **Checkpoint:** `tabicl-classifier-v2-20260212.ckpt`
- **Revision:** `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`
- **SHA-256:** `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`

### Released Pretraining Recipe Caveat

The upstream project now publishes TabICLv2 pretraining scripts derived from the original private pretraining codebase. The maintainers state that these scripts were cross-checked against the released checkpoints but have not yet been fully validated through an end-to-end reproduction of the original pretraining run.

Consequently, the released checkpoint itself should remain the canonical artifact for this model version.

### Weight-Decay Implementation Detail

The paper describes cautious weight decay as part of the training methodology. The current upstream reproduction scripts explicitly keep:

```text
use_cautious_wd = False
```

because cautious weight decay was not wired into Muon during the reference runs that produced the released checkpoints.

For checkpoint reproduction, the released scripts' documented reference behaviour should therefore be distinguished from the higher-level methodological description in the paper.

### Recommended Reproduction Record

For reproducible downstream use, record at minimum the exact checkpoint file and SHA-256, Hugging Face revision, `tabicl` package version, random seed, preprocessing, support/train split, inference estimator count, fine-tuning configuration if applicable, and hardware/software environment.

## Limitations

1. Performance remains dataset dependent.
2. Pretraining used up to approximately 100 features, so larger feature counts represent extrapolation.
3. The largest pretraining tasks contained approximately 48,000 labelled training examples, although the architecture can technically process much larger contexts.
4. Classification pretraining used at most 10 classes; many-class support relies on an additional decomposition mechanism.
5. Strong aggregate benchmark rankings do not guarantee strong performance on a particular dataset.
6. General demographic fairness has not been established.
7. Robustness to arbitrary distribution shift has not been established.
8. Probability calibration must be evaluated for the downstream application.
9. High-impact application suitability cannot be inferred from general tabular benchmarks.
10. Exact reproduction of the original pretraining run has not yet been demonstrated using the currently published reproduction scripts.

## License

The core TabICL tabular implementation and released TabICLv2 checkpoints are distributed under the **BSD 3-Clause License**.

The upstream repository also contains code with separate licensing associated with forecasting functionality. This card concerns the TabICLv2 tabular classification checkpoint and not the forecasting extension.

Downstream datasets and fine-tuned derivatives may carry additional licensing or governance obligations.

## Model Ownership and Attribution

TabICLv2 was developed by Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan of the Soda team at Inria. Upstream source code is maintained at [soda-inria/tabicl](https://github.com/soda-inria/tabicl) and base model checkpoints are hosted on Hugging Face at [jingang/TabICL](https://huggingface.co/jingang/TabICL).

Downstream integrations and fine-tuned derivatives should distinguish their modifications from the upstream TabICLv2 checkpoint.

## Citation

Cite the TabICLv2 paper, the original TabICL foundation paper, and the upstream repository:

### Papers

- **TabICLv2 (2026):**  
  Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2026). *TabICLv2: A better, faster, scalable, and open tabular foundation model.* ICML 2026. arXiv:2602.11139. https://doi.org/10.48550/arXiv.2602.11139

- **TabICL (2025):**  
  Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2025). *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data.* ICML 2025. arXiv:2502.05564. https://doi.org/10.48550/arXiv.2502.05564

### Upstream Repository

- **TabICL Codebase:**  
  Soda team, Inria. *TabICL: Open Tabular Foundation Models* [Software]. GitHub. https://github.com/soda-inria/tabicl

### BibTeX

```bibtex
@article{qu2026tabiclv2,
  title={{TabICLv2}: {A} better, faster, scalable, and open tabular foundation model},
  author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  journal={arXiv preprint arXiv:2602.11139},
  year={2026}
}

@inproceedings{qu2025tabicl,
  title={Tab{ICL}: {A} Tabular Foundation Model for In-Context Learning on Large Data},
  author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  booktitle={International Conference on Machine Learning},
  year={2025}
}

@misc{tabicl_repo,
  author = {Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  title = {Tab{ICL}: Open Tabular Foundation Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/soda-inria/tabicl}}
}

@misc{tabicl_hf,
  author = {Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  title = {{TabICL}: Open Tabular Foundation Models},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/jingang/TabICL}}
}
```

## Evaluation Status

### Established by the Upstream Work

The upstream evidence establishes binary classification, multiclass classification, native many-class handling, in-context learning, optional downstream fine-tuning, strong performance on TabArena and TALENT, generalization beyond the largest synthetic pretraining contexts, efficient large-table inference, and open checkpoint availability.

### Application-Dependent or Not Generally Established

The upstream evidence does not establish universal accuracy on a specific downstream dataset, demographic fairness, subgroup parity, probability calibration, adversarial robustness, domain-specific safety, operational service levels, or suitability for high-impact decision-making.

These properties must be evaluated for the specific downstream model and application.
