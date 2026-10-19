# Pre-registration: MAGIC Gamma Telescope fine-tuning grid and fresh-test reading rule

- **Date:** 2026-10-19. Written before any grid run on this data. It is committed alone, before the code that implements it.
- **Notebook:** `tutorials/tabiclv2_classifier_colab.ipynb` (row 96 of the 2026-10-15 tutorial triage; findings R96-M1..M3, R96-m1..m12).
- **Owner decisions:** R96-D1 = C (switch the default to a harder pinned table: MAGIC Gamma Telescope); R96-D2 = A (the earlier
  Breast Cancer 114-row test stays only as a labelled earlier-run reference); 2026-10-19: teach the null result on MAGIC.
- **Status of the fresh test at the time of writing:** never fitted, predicted or scored by anyone. Its row ids were only hashed.

## 1. What was looked at before writing this (validation only)

A CPU room check on the **validation** partition (2026-10-19, uncommitted working tree, CPU, `tabicl` 2.1.1):

| Model | Val log loss | Val accuracy | Val ROC-AUC |
|---|---|---|---|
| Pretrained TabICLv2, `n_estimators=8` | 0.2634 | 0.903 | 0.9463 |
| Random forest (dataset research, CPU) | 0.325 | 0.873 | – |

Three fine-tunes (`max_data_size` 500, so 6 AdamW steps per epoch; no LR scheduler, so no zero-LR step; 3 epochs; learning
rates 1e-5, 5e-5 and 1e-4) never beat tabicl's own validation baseline (log loss 0.2701 at `n_estimators_validation=2`). Every
epoch was worse: 0.286 / 0.334 / 0.303 at 5e-5 and 0.296 / 0.361 / 0.287 at 1e-4. tabicl restored the pretrained weights each time
(parameter-delta L2 of the restored model = 0.0). **The expected selection is therefore the pretrained model.** The owner ruled
on 2026-10-19 to teach this null result rather than switch datasets.

## 2. Data (fixed)

| Setting | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset 159, `https://archive.ics.uci.edu/static/public/159/data.csv`, DOI 10.24432/C52C8B, CC BY 4.0 |
| Source SHA-256 | `a52d3a378e43df7b678c4c0b2f03a66a058bbfeb27c95ccd037a767e7d172984` (1,496,488 bytes; 19,020 rows × 10 numeric features + `class` g/h) |
| Prep | `class` renamed to `target`; rows copied as the original text |
| Split rule | `sklearn.model_selection.train_test_split`, stratified on `target`, `random_state=0`: the fresh test (1,500) is carved **first** from all rows, then validation (1,000) from the rest, then train capped at 3,000 (stratified); the remaining 13,520 rows are the reserve |
| Row-id SHA-256 (sha256 of the comma-joined sorted 0-based row indices) | train `009082b8…10cd`, val `22dbeb10…17b3`, **fresh test `8847cab4e87c8ff30dd6a27045464f7891b2e14dd39149c5e5648348f190f3e4`**, reserve `c1b30483…08b3` |
| Archive | `examples/sample-data/magic-gamma.zip`, SHA-256 `2c6108e6d3e53cd9fcc8417df6cc8018ea2faa70a3e9b8e58fd3ef4e1463497d` (train.csv `0a5f1692…a246`, val.csv `36bc2340…b0b4`, test.csv `62c3eff6…2b51`, split_row_ids.json `96fba963…6a30`); fetched by the notebook at an immutable commit and digest-checked |
| Class counts | train 1,945 g / 1,055 h; val 648 / 352; fresh test 973 / 527 |
| Reserve | 13,520 rows, never used by the notebook. They are the source for a future fresh test if this one is ever exposed. |
| Exact-duplicate feature rows | 3 fresh-test rows have an exact feature duplicate in train, 1 in validation (7 in the reserve; 1 train–val pair; 115 duplicated rows in the whole source). **Handling:** kept, so the split stays exactly as hashed; the report adds a sensitivity readout of the primary endpoint on the 1,496 rows without those 4 (secondary, pre-registered here). |

## 3. Model and fine-tuning grid (fixed)

Common to every config: checkpoint `jingang/TabICL` revision `4dcd344e…` (`tabicl-classifier-v2-20260212.ckpt`), `FinetunedTabICLClassifier`
from `tabicl` 2.1.1, AdamW, `weight_decay` 0.01, `grad_clip` 1.0, **`max_data_size` 500** (3,000 train rows → 6 chunks, so 6 optimizer
steps per epoch), **`use_lr_scheduler=False`** (constant LR, so no warmup and no zero-LR first step), `finetune_ctx_query_ratio` 0.2,
`amp=True` on CUDA (fp16 autocast, float32 weights and optimizer state), early stopping on the **validation** partition with
`eval_metric="log_loss"`, `min_delta` 1e-4, `n_estimators_inference` 8, `random_state` 42. The candidate of each config is the
checkpoint tabicl writes as `best.ckpt` (the pretrained weights when no epoch beat the baseline), re-scored at `n_estimators=8`.

| Config | Role | `learning_rate` | `epochs` | `patience` | `n_estimators_finetune` / `_validation` | `time_limit` (s) |
|---|---|---|---|---|---|---|
| C1 | production-like (`dimer-pipeline.json` defaults except `max_data_size` 500, no warmup, `eval_metric` log_loss) | 1e-5 | 30 | 8 | 2 / 2 | 1800 |
| C2 | larger steps | 5e-5 | 10 | 3 | 2 / 2 | 600 |
| C3 | smaller steps | 1e-6 | 10 | 3 | 2 / 2 | 600 |

Inference for every model: `n_estimators=8`, `random_state=42`, support = the 3,000 train rows. Probabilities are renormalised
row-wise after the maximum row-sum deviation is recorded.

**Determinism (pre-registered settings, recorded in the report):** `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA initialises,
`torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`,
`torch.use_deterministic_algorithms(True)` (no `warn_only`), seeded `random` / `numpy` / `torch` (seed 42), no DataLoader workers. An op
with no deterministic CUDA kernel raises and stops the run; it is named, and no run of record is made until it is resolved. The
condition stage prints a **grid hash** (SHA-256 of the grid's per-epoch validation values, selections and parameter deltas, rounded to
6 decimals) so two masked runs can be compared.

**Grid time estimate (T4, inferred):** 6 steps per epoch at about 1–3 s per step plus a validation pass per epoch: C1 stops after
9 epochs when no epoch improves (up to 30 otherwise), C2 and C3 after 4. Under 15 minutes in total, below the 55-minute threshold,
so the grid runs on the default path.

## 4. Selection rule (fixed, validation only)

1. Score the pretrained model and each config's candidate on the 1,000 validation rows at `n_estimators=8`.
2. A config is eligible only if its validation log loss is lower than the pretrained model's by **more than 0.005** (the practical
   tolerance: about 2 % of the pretrained value, about 50× tabicl's `min_delta`, and below what changes any decision a learner would
   make with these probabilities).
3. Among eligible configs, the lowest validation log loss wins; a tie within 1e-4 goes to the config that took fewer optimizer steps.
4. Otherwise the **pretrained model is kept** (expected).

The fresh test plays no part in any of this.

## 5. Reading rule (fixed)

- **Primary endpoint:** fresh-test log loss difference **Δ = LL(selected) − LL(pretrained)** over the 1,500 fresh-test rows.
- **Interval:** paired bootstrap 95 % percentile interval over rows, `seed` 20261019, `n_boot` 10,000.
- **Verdict:**
  - if the selected model **is** the pretrained model: **"same model (Δ = 0)"** — no difference was measured, because none can exist;
  - "improved" if the interval's upper bound is < 0 **and** Δ ≤ −0.005;
  - "worse" if the interval's lower bound is > 0;
  - otherwise "no detectable difference".
- **Secondary:** fresh-test accuracy (Wilson 95 % interval) and ROC-AUC of the selected and pretrained models; the paired accuracy
  difference with its bootstrap interval; the primary endpoint on the 1,496 rows without the 4 duplicated rows.
- **Exploratory readout (disclosed, validation only):** for the config whose **last-epoch** checkpoint has the lowest validation log
  loss, that checkpoint's validation log loss and accuracy and its parameter-delta L2 against the pretrained weights. It shows what
  the weights did when tabicl did not keep them. It never touches the fresh test.
- **Power (planning estimates, inferred from the dataset research's CPU proxies):** a single accuracy on 1,500 rows has a Wilson
  half-width of about ±0.017; a paired accuracy difference about ±0.011 at 5 % discordance; a paired log-loss difference about
  ±0.017 (pessimistic proxy: random forest vs gradient boosting). Fine-tuned and pretrained TabICL should differ less.

## 6. Score-once rule (SCORE_ONCE_RULE.md, 2026-10-17)

- The fresh test is scored **only** when `preregistered` is true. `preregistered` is computed from **all** of: data source and
  archive digest, split row-id hashes, the grid table and common settings above, the selection tolerance, inference settings,
  bootstrap seed and size, determinism flags, BYOD off, fine-tuning on, and a **Tesla T4** CUDA device. Anything else is
  exploratory: validation only, written to `*_exploratory_<n>.json`, labelled `not scored (exploratory)`.
- Before scoring, the fresh-test stage recomputes `preregistered`, checks that the grid is complete and matches this table, that the
  selected model's checkpoint digest equals the one the condition stage recorded, and that the fresh-test member digest matches.
- `DIMER_MASK_FRESH_TEST=1` skips the fresh test entirely (the member is never extracted). The mask is re-read from the notebook
  kernel's environment on every stage call and printed before any data is fetched.
- The first pre-registered scoring in a runtime is labelled **"pre-registered evaluation (this runtime)"**; a later identical scoring
  in the same runtime writes `_reproduction_<n>` and is labelled "reproduction". `RECORDED_EVALUATION` stays `None` until the first
  hosted T4 run is recorded in `docs/release-verification.md`; after that, every identical run in any runtime is a reproduction.
- A look counts only after scoring completes and the report is written. Report and ledger are written atomically. A failed or
  partial grid, a failed digest check or an interrupted write is not a look. An empty or corrupt ledger never crashes a run; it is
  set aside and the scoring is labelled "reproduction" (never claimed as the first evaluation).
- Fixer CPU checks run with the fresh test **masked**.

## 7. Disclosures

- **Benchmark exposure:** MAGIC is in TALENT (300 datasets), which the TabICLv2 paper (arXiv 2602.11139) evaluated on. It is not in
  TabArena or OpenML-CC18. Membership in the 60 TabPFNv2 development datasets was **not checked**.
- **Pretraining** is on synthetic tables only, so benchmark membership is evaluation exposure, not training leakage.
- **MAGIC rows are Monte Carlo simulations** of telescope events, not field measurements.
- **Earlier results:** all earlier recorded numbers of this notebook (2026-10-10 Colab T4, 2026-10-11 CPU drive) are on the Breast
  Cancer sample and its 114-row test, under the previous recipe. They remain an earlier-run reference, labelled as such; that test is an
  exposed development test and no setting here was chosen on it.
