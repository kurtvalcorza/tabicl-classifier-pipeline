# TabICLv2 Classifier Sample Datasets

This directory provides and documents sample datasets for the [TabICLv2 Classifier Colab Tutorial](../../tutorials/tabiclv2_classifier_colab.ipynb).

TabICLv2 is an in-context learning tabular foundation model that ingests training rows as in-context conditioning context (`training_context.parquet`). The tutorial's default dataset is MAGIC Gamma Telescope (section 0, added 2026-10-19); the three earlier samples remain selectable:

| Dataset | Modality / Task | Rows (Train / Val / Test) | Features | Source & License |
|---|---|---|---|---|
| **MAGIC Gamma Telescope** (default) | Binary (`g` gamma / `h` hadron) | 3,000 / 1,000 / **1,500 fresh test** (+13,520 unused reserve; 19,020 total) | 10 numeric | UCI 159 (CC BY 4.0) |
| **Breast Cancer Wisconsin** | Binary sanity check (`malignant` / `benign`) | 341 / 114 / 114 (569 total) | 30 numeric | scikit-learn / UCI (CC0 1.0 Universal) |
| **Wine Recognition** | 3-class chemical classification (integer IDs `0, 1, 2`) | 106 / 36 / 36 (178 total) | 13 numeric | scikit-learn / UCI (CC BY 4.0) |
| **Palmer Penguins** | 3-class morphological classification | 199 / 67 / 67 (333 total) | 6 mixed (2 categorical strings, 4 numeric) | OpenML 42585 (CC0 1.0 Universal) |

---

## 0. MAGIC Gamma Telescope (default, 2026-10-19)

- **Purpose:** the default table of the E2E tutorial. Pretrained TabICLv2 is not at a trivial ceiling here (validation accuracy about 0.90, log loss about 0.26 on a CPU development check), and the 1,500-row fresh test resolves differences far finer than the earlier 114-row Breast Cancer test.
- **Source:** Bock, R. (2004). MAGIC Gamma Telescope [Dataset]. UCI Machine Learning Repository, dataset 159, https://doi.org/10.24432/C52C8B, `https://archive.ics.uci.edu/static/public/159/data.csv`, SHA-256 `a52d3a378e43df7b678c4c0b2f03a66a058bbfeb27c95ccd037a767e7d172984`. Licence: CC BY 4.0 (attribution required; this card is the attribution). The rows are Monte Carlo simulations of telescope events, not field measurements, and contain no personal data.
- **Archive:** `magic-gamma.zip` (SHA-256 `2c6108e6d3e53cd9fcc8417df6cc8018ea2faa70a3e9b8e58fd3ef4e1463497d`), built by `tools/build_magic_sample.py` from the pinned source: `train.csv`, `val.csv`, `test.csv` (target column `target`, original text copied unchanged) and `split_row_ids.json` (0-based source row ids of every partition, including the reserve).
- **Split:** stratified `train_test_split`, seed 0: the 1,500-row **fresh test** is carved first from all rows, then validation (1,000), then train capped at 3,000; the 13,520 rows left are the **reserve**, never used by the notebook and kept as the source of a future fresh test. Row-id SHA-256: train `009082b81a485027433be0fcb0de99855a480b425ebbb9e70bd37242307210cd`, val `22dbeb1022a76e2cab51dd7a2df9869efa71430867d975f21ac59369e84d17b3`, test `8847cab4e87c8ff30dd6a27045464f7891b2e14dd39149c5e5648348f190f3e4`, reserve `c1b30483d5024fa6f94893a824dcbacf60ca942b941faa790edbb541c97208b3`.
- **Fresh test:** scored only under the pre-registered settings (`docs/magic-finetune-preregistration.md`); see `docs/release-verification.md` for the recorded evaluation and its reproductions.
- **Duplicates:** 3 test rows have an exact feature duplicate in train and 1 in validation; they are kept and handled by a pre-registered sensitivity readout.
- **Benchmark exposure:** MAGIC is in the TALENT benchmark, which the TabICLv2 paper evaluated on; it is not in TabArena or OpenML-CC18. TabICLv2 is pretrained on synthetic tables only.

---

## 1. Breast Cancer Wisconsin (Diagnostic)

- **Status:** the earlier default; kept as a labelled earlier-run reference (its 114-row test was scored in the 2026-10-10 and 2026-10-11 runs and is an exposed development test).
- **Purpose:** Quick sanity benchmark. Pretrained TabICLv2 scores ~0.97–0.98 out-of-the-box, making it ideal for verifying GPU execution, in-context conditioning, and export mechanics.
- **Source:** Built into `sklearn.datasets.load_breast_cancer` (zero network downloads).
- **Target:** `target` — binary diagnosis (`0` = malignant, `1` = benign).
- **Features:** 30 continuous numeric attributes computed from digitized images of fine needle aspirates (cell nuclei radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension).
- **Split:** 60% train / 20% holdout / 20% test (stratified by class, seed 42).

---

## 2. Wine Recognition

- **Purpose:** Multiclass benchmark demonstrating 3-class probability calibration (`predict_proba`), multi-class ROC-AUC (`ovr`), log loss, and label alignment with classical tree baselines.
- **Source:** Built into `sklearn.datasets.load_wine` (zero network downloads).
- **Target:** `target` — 3 wine cultivars stored as integer class IDs `0`, `1`, `2` (corresponding to cultivar display semantics `class_0`, `class_1`, `class_2`).
- **Features:** 13 continuous numeric attributes derived from chemical analysis (alcohol, malic acid, ash, alcalinity, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, proline).
- **Split:** 60% train / 20% holdout / 20% test (stratified by class, seed 42).

---

## 3. Palmer Penguins

- **Purpose:** Demonstrates TabICLv2 on mixed tabular data with string/categorical columns, verifying the `fit_encoder()` and `apply_encoder()` ordinal mapping pipeline alongside 3-class species prediction.
- **Archive:** `palmer-penguins.zip` (3.9 KB). Contains `train.csv`, `val.csv`, and `test.csv` (SHA-256: `fe894295ccc0a447dc020f5d57798968b95eae3da1a4cbe09a638e6b1bd0ba44`).
- **Target:** `species` — 3 penguin species (`Adelie`, `Chinstrap`, `Gentoo`).
- **Features:** 6 mixed features:
  - Categoricals (strings): `island` (`Biscoe`, `Dream`, `Torgersen`), `sex` (`FEMALE`, `MALE`).
  - Numerics: `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`.
- **Split:** 60% train (199 rows) / 20% val (67 rows) / 20% test (67 rows) (stratified by species, seed 42, 11 incomplete rows removed from 344 total: 10 missing measurements + 1 unrecorded sex marker, leaving 333 complete rows).
- **Provenance & License:** Gorman KB, Williams TD, Fraser WR (2014) *Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis)*. PLoS ONE 9(3): e90081. Distributed under **CC0 1.0 Universal (Public Domain Dedication)** via OpenML (Dataset 42585).

### Immutable tutorial asset

The final notebook loader retrieves `palmer-penguins.zip` from immutable repository revision `169e60fa8d956aa389c144ac9c7988b92db84a79` and verifies the SHA-256 above before extraction. That revision is retained by the durable branch `anchors/sample-data-20260911`, so deleting the feature branch after merge does not break the published tutorial asset.

---

## How Bundled Datasets Were Built

The bundled archive was generated deterministically using [`../build_sample_datasets.py`](../build_sample_datasets.py):

```bash
python examples/build_sample_datasets.py
```
