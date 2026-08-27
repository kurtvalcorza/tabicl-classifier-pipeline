---
license: bsd-3-clause
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - tabular-foundation-model
  - in-context-learning
  - tabicl
base_model: jingang/TabICL
---

# TabICLv2 Classifier

## Description

TabICLv2 Classifier is a pretrained tabular foundation model developed by Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan of the Soda team at Inria.

It performs supervised classification on structured or tabular data using **in-context learning (ICL)**. Rather than fitting a conventional model from randomly initialized parameters for every new dataset, TabICLv2 processes labelled training examples and unseen examples jointly through a pretrained Transformer architecture and performs task adaptation during inference.

TabICLv2 is the successor to the original TabICL model. Version 2 introduces a redesigned synthetic-data prior, architectural improvements for long-context generalization and efficiency, and an optimized pretraining procedure using the Muon optimizer.

The classifier is pretrained on synthetic classification tasks with up to 10 classes but supports downstream classification problems with more than 10 classes through **mixed-radix ensembling**, which decomposes a many-class problem into multiple views containing at most 10 classes each.

## Model Details

- **Model name:** TabICLv2 Classifier
- **Model family:** TabICLv2
- **Checkpoint identifier:** `tabicl-classifier-v2-20260212.ckpt`
- **Hugging Face repository:** `jingang/TabICL`
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
- **Pinned Hugging Face revision:** `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`
- **SHA-256:** `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`

The checkpoint hash and revision should be retained when exact model-version provenance is required. The classifier and regressor are separate pretrained checkpoints and should not be treated as interchangeable.

## Intended Use and Limitations

### Primary Intended Uses

TabICLv2 Classifier is intended for supervised classification of structured or tabular data. Appropriate applications include binary classification, multiclass classification, many-class classification, risk categorization, quality or condition classification, churn and event prediction, scientific classification problems represented as structured feature tables, operational and administrative classification tasks, and other row-per-observation supervised classification problems.

The model is particularly useful when users want strong tabular predictive performance with little or no task-specific hyperparameter optimization.

TabICLv2 was pretrained across dataset sizes extending to approximately 48,000 labelled training samples and up to 100 features. The upstream project reports strong generalization beyond the pretraining regime, including substantially larger datasets and datasets with considerably more features. Such use beyond the pretraining distribution should nevertheless be regarded as extrapolation and independently evaluated.

### Primary Intended Users

The model is primarily intended for machine-learning researchers, data scientists, machine-learning engineers, software developers, scientific researchers, and practitioners building predictive systems from structured datasets.

Users should understand dataset provenance, evaluation methodology, leakage risks, class imbalance, distribution shift, and the consequences of prediction errors.

### Out-of-Scope Use Cases

TabICLv2 Classifier is not intended for regression or continuous-value prediction, raw image/audio/video/natural-language modelling, unsupervised clustering, causal-effect estimation, generative modelling, direct time-series forecasting without conversion into an appropriate supervised prediction problem, or autonomous high-impact decision-making without domain-specific validation and appropriate human oversight.

Strong benchmark results do not guarantee satisfactory performance on a particular downstream dataset.

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

## Factors

### Groups

TabICLv2's pretraining data are synthetic rather than a fixed population of human subjects. No demographic groups such as age, sex, gender, ethnicity, disability, nationality, or socioeconomic status are intrinsic evaluation groups of the foundation model. Any human-centred downstream use requires application-specific subgroup evaluation.

### Instrumentation

The model consumes derived tabular features and is not tied to a specific camera, sensor, laboratory device, or other physical instrument. Where features originate from measurement systems, limitations of those instruments belong to the downstream dataset and application documentation.

### Environment

The foundation model is not validated for every geographic, temporal, institutional, or operational setting. Changes in feature distributions, class prevalence, data-collection systems, populations, or operating conditions can materially affect performance.

## Ethical Considerations and Risks

Potential risks include incorrect classifications, amplification of downstream dataset biases, unequal error rates across relevant groups, class-imbalance effects, distribution shift, target or feature leakage, overconfidence in foundation-model benchmark results, inappropriate use of sensitive attributes, and automation bias.

Synthetic pretraining reduces direct memorization risks associated with real pretraining records but does not prevent bias or privacy risks introduced by downstream data.

TabICLv2 has not been established as universally suitable for autonomous decisions in medicine, criminal justice, employment, lending, insurance, public benefits, or other high-impact contexts. Such applications require domain-specific validation, governance, and appropriate human oversight.

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

TabICLv2 was developed by Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan of the Soda team at Inria.

Downstream integrations and fine-tuned derivatives should distinguish their modifications from the upstream TabICLv2 checkpoint.

## Citation

Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2026). *TabICLv2: A better, faster, scalable, and open tabular foundation model.* ICML 2026. arXiv:2602.11139. https://doi.org/10.48550/arXiv.2602.11139

## Evaluation Status

### Established by the Upstream Work

The upstream evidence establishes binary classification, multiclass classification, native many-class handling, in-context learning, optional downstream fine-tuning, strong performance on TabArena and TALENT, generalization beyond the largest synthetic pretraining contexts, efficient large-table inference, and open checkpoint availability.

### Application-Dependent or Not Generally Established

The upstream evidence does not establish universal accuracy on a specific downstream dataset, demographic fairness, subgroup parity, probability calibration, adversarial robustness, domain-specific safety, operational service levels, or suitability for high-impact decision-making.

These properties must be evaluated for the specific downstream model and application.
