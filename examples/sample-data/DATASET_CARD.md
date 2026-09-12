# TabICLv2 Classifier Sample Datasets

This directory provides and documents sample datasets for the [TabICLv2 Classifier Colab Tutorial](../../tutorials/tabiclv2_classifier_colab.ipynb).

TabICLv2 is an in-context learning tabular foundation model that ingests training rows as in-context conditioning context (`training_context.parquet`). The tutorial supports three complementary sample datasets:

| Dataset | Modality / Task | Rows (Train / Val / Test) | Features | Source & License |
|---|---|---|---|---|
| **Breast Cancer Wisconsin** | Binary sanity check (`malignant` / `benign`) | 341 / 114 / 114 (569 total) | 30 numeric | scikit-learn / UCI (CC0 1.0 Universal) |
| **Wine Recognition** | 3-class chemical classification (integer IDs `0, 1, 2`) | 106 / 36 / 36 (178 total) | 13 numeric | scikit-learn / UCI (CC BY 4.0) |
| **Palmer Penguins** | 3-class morphological classification | 199 / 67 / 67 (333 total) | 6 mixed (2 categorical strings, 4 numeric) | OpenML 42585 (CC0 1.0 Universal) |

---

## 1. Breast Cancer Wisconsin (Diagnostic)

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
