# Release verification

`tutorials/tabiclv2_classifier_colab.ipynb` (`E2E`) and `tutorials/tabiclv2_classifier_artifact_inference_colab.ipynb`
(`ARTIFACT-INFERENCE`) are **release candidates** until the exact notebook revisions have executed top-to-bottom in a
clean supported runtime. Unit tests, JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are
necessary checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is the durable
release-gate record for both notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for each of the two notebooks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, each named in `tutorials/README.md` with its profile, the notebook-spec version and
  the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`, `standalone: true` and `generated_from`
  (repository, module commit, module path, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary path
  (the previous pair's `git+https://github.com/kurtvalcorza/...` install and lock URL are gone); one cell tagged
  `embedded_module` equal to `src/tabicl_classifier_pipeline/api.py` after the generator's documented rewrite (the
  `DEFAULT_WEIGHTS_DIR` line); the inline `MANIFEST` equal to the committed `weights/tabicl-classifier-v2/dimer-base-manifest.json`
  and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output for its template; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same identity
  string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls — E2E: `stage_missing_files`, `verify_snapshot`,
  `TabICLClassificationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, n_estimators=8, random_state=42)`, `validate_inputs`
  (with the too-few-rows rejection probe), `prepare_classification_table`, `fit_categorical_encoder`, `majority_class_baseline`,
  `fit`, `compare_metric`, `create_finetuned_classifier` / `fine_tune_classifier` behind the CUDA guard, LightGBM / Random Forest
  on the same partitions, `evaluation_report` with the independent test, `read_inference_csv` + inference-mode
  `validate_inputs`, the bundle export with `sha256_file` digests and the payload allowlist, `safe_extract_zip` +
  `verify_artifact_bundle` on the fresh reload and the `rtol=1e-5, atol=1e-7` equivalence check; companion:
  `safe_extract_zip`, `validate_artifact_runtime`, the base-model identity/digest comparison, `verify_artifact_bundle`,
  reconstruction on the bundled checkpoint with `allow_auto_download=False`, inference-mode `validate_inputs` with a
  rejection probe, `apply_categorical_encoder`, `predict`, a `not-measurable` `evaluation_report` — the ceiling prints,
  the four exports per notebook, the learner-facing statements (uncalibrated probabilities / argmax rule, class coverage, dropped rows counted, BYOD privacy
  boundary, fine-tuning gate/disk usage/reproducibility boundaries, member-by-member extraction, trust boundary, no artifact
  created in the companion), and the three gated-off form parameters (`USE_BYOD`, `RUN_FINE_TUNING`, `RUN_NEW_DATA_INFERENCE`);
  forbidden patterns (credential-in-URL, own-repository clone/install, a `git+https://` dependency without a 40-hex SHA,
  a mutable `revision='main'`, `worker.run(` / `worker_cli(` / `subprocess.run([` outside the install cell, direct
  `from tabicl import` / `TabICLClassifier(` / `FinetunedTabICLClassifier(` / `zipfile.ZipFile(` / `hf_hub_download(` /
  `sklearn.metrics` use **outside the carried module cell**, `trust_remote_code=True`, `pickle.load`, `torch.load(`,
  `extractall(`; in the companion also `load_breast_cancer(`, `load_wine(`, `check_stratifiable(`, `majority_class_baseline(`, `shutil.make_archive(`, any fine-tuning call);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and the `## Checkpoint Provenance` section.

CI also runs `ruff`, `tools/build_notebook.py --check` for both templates, `scripts/validate_colab_tutorial.py` (the
repository's own spec-1.1 checks: exact build backend, lock files, fine-tune gate, hardening markers), the two
`scripts/test_*.py` classification suites (now against the package functions) and the offline unit suite (`tests/`:
`test_snapshot_helpers.py`, `test_role_helpers.py`, `test_notebook_parity.py`, `test_companion_parity.py`; injected
downloader, no weights, no model). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or GPU runtime, Python 3.11+ | The runtime the tutorials are written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle kernel, Python 3.11+ image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebooks are standalone). For the companion the shim also places the E2E run's `outputs/tabiclv2_classifier_artifact.zip` and a separately generated unlabelled CSV, and sets `ARTIFACT_ZIP_PATH` / `NEW_DATA_PATH` to them |
| GitHub Actions `release-notebook-execution` (previous pair) | ubuntu, Python 3.13, `nbclient` | Executed the previous repository-installing pair through real kernels (`scripts/execute_notebook_release_paths.py`, `/content` workspace, `DIMER_ARTIFACT_PATH` / `DIMER_INFERENCE_CSV_PATH`); it must be re-pointed at the standalone pair (`outputs/` workspace, `ARTIFACT_ZIP_PATH` / `NEW_DATA_PATH` form parameters) before it counts again |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a new CPU (or CUDA) runtime with **no repository checkout** and a clean model cache;
3. run it top-to-bottom without editing implementation cells (form parameters at their defaults: `DATA_SOURCE = 'Sample: Breast Cancer'`,
   `USE_BYOD = False`, `RUN_FINE_TUNING = False`, `EVAL_METRIC = 'accuracy'`, `RUN_NEW_DATA_INFERENCE = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`;
   `torch==2.14.0` is now installed from the pins rather than assumed runtime-provided, and `huggingface-hub` is the lock's 1.30.0);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `TabICLClassificationPipeline`, the helpers and the bundle safety functions) with no import of the repository package;
   - pinned `jingang/TabICL` acquisition at the immutable revision through the package: the inline `MANIFEST` is asserted
     against the module identity and written to `weights/tabicl-classifier-v2/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)`
     reports the one manifest entry (`tabicl-classifier-v2-20260212.ckpt`, 114,324,594 bytes) on a clean runtime, `verify_snapshot`
     returns the manifest dict, and `from_pretrained(...)` reports `source == 'local-snapshot'`;
   - the Breast Cancer sample split 60/20/20 with stratification and its CSV SHA-256 printed (`train=341, holdout=114, test=114`, two classes); the ceilings (`MIN_TRAIN_ROWS` 50, `MIN_EVAL_ROWS` 2, `MIN_CLASSES` 2,
     `MAX_TRAIN_ROWS` 50,000, `MAX_FEATURES` 2,000) surfaced; `validate_inputs` writes `outputs/tabiclv2_classifier_input_manifest.json`
     (three accepted tables, one recorded rejection finding from the too-few-rows probe); dropped/unseen counts printed;
   - `fit` (in-context conditioning) and `classification_metrics` on holdout and independent test; `majority_class_baseline`;
     the fine-tuning gate skipped; LightGBM / Random Forest fitted on the same rows; the blend chosen on holdout accuracy with probabilities aligned to the TabICLv2 class order;
   - `evaluation_report` writes `outputs/tabiclv2_classifier_evaluation_report.json` with verdict `sample-sanity`, the five
     metric ids, the independent-test block and the majority-class baseline;
   - eight held-out rows scored into `outputs/tabiclv2_classifier_predictions.csv` (inference upload skipped);
   - the bundle `outputs/tabiclv2_classifier_artifact.zip` written, extracted with `safe_extract_zip` into `outputs/artifact-reload/`,
     `verify_artifact_bundle` passes, the rebuilt classifier's predictions equal the in-memory model's within `rtol=1e-5, atol=1e-7`;
     `outputs/tabiclv2_classifier_result.json` written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. in a **second** clean runtime, run the exact companion notebook revision with `ARTIFACT_ZIP_PATH` pointing at a copy of the
   E2E run's bundle and `NEW_DATA_PATH` at a separately generated unlabelled CSV (or supply both through the upload dialog);
   verify the archive checks, `validate_artifact_runtime`, the base-model identity comparison and `verify_artifact_bundle` pass
   before reconstruction, that `validate_inputs(..., target_column=None, ...)` writes `outputs/tabiclv2_classifier_artifact_inference_input_manifest.json`
   with one recorded rejection finding, and that the `not-measurable` evaluation report, `..._predictions.csv` and `..._result.json` are written;
7. verify the exports exist and the interpretation sections match the observed paths;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, tabicl, device), model identifier and immutable
   revision, whether the model cache was clean, outcome, produced outputs, and any warning or applicable `SHOULD` deviation in the table below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook file (verify with `git rev-parse <commit>:tutorials/<notebook>`). Wall
times, when recorded, are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Standalone E2E default sample path | | pending — queued to the GPU lane |
| | | | Standalone ARTIFACT-INFERENCE with an external bundle | | pending — queued to the GPU lane |

## Current status

No clean-runtime execution of the standalone notebooks has been recorded yet; both runs are **pending** and queued to the
GPU lane. Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code
cell, and the offline unit suite passed on the tutorial source at the candidate revision, which is necessary but not
sufficient. The registry status remains **Candidate** until a reviewer confirms a recorded run against the notebook blobs
under review and an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real `hf_hub_download` fetch
into a fresh `weights/tabicl-classifier-v2/` has not been executed); `from_pretrained` constructs the estimator but `tabicl`
deserialises the checkpoint only on the first `fit`; the previous CI kernel executions covered the old repository-installing
notebooks, not this carrier; and the standalone carrier itself — executing the carried module cell in a runtime that has no
repository checkout — has been validated statically only (parity PASS, carrier probe with the package import blocked),
never run. The clean runs will be the first execution of the standalone path, of the staging path, of the extracted helpers
(`validate_inputs`, `prepare_classification_table`, `classification_metrics`, `majority_class_baseline`, `evaluation_report`,
`safe_extract_zip`, `verify_artifact_bundle`) and of `TabICLClassificationPipeline` against the real weights.
