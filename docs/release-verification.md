# Release verification

`tutorials/tabiclv2_classifier_colab.ipynb` (`E2E`) and `tutorials/tabiclv2_classifier_artifact_inference_colab.ipynb`
(`ARTIFACT-INFERENCE`) are **release candidates** until the exact notebook revisions have executed top-to-bottom in a
clean supported runtime. Unit tests, JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are
necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.2. This file is the durable
release-gate record for both notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for each of the two notebooks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, each named in `tutorials/README.md` with its profile, the notebook-spec version and
  the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, mode `GUIDED`, `standalone: true` and
  `generated_from` (repository, generating commit, package module paths and SHA-256, carried-file digests, generator
  `build_notebook.py/3.0-tabular`);
- the standalone carrier and isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): no clone, repository install
  or repository import on the primary path and nothing installed into the notebook kernel; one carrier cell
  (`metadata.dimer.embedded_sources`) whose `CARRIED_FILES` / `CARRIED_BINARY` equal the repository files they come from
  (`src/tabicl_classifier_pipeline/{__init__,api}.py`, the stage runner `tools/tutorial_stages.py` or
  `tools/tutorial_stages_artifact_inference.py`, `tutorials/requirements-colab.lock.txt`, the committed
  `weights/tabicl-classifier-v2/dimer-base-manifest.json`, the licence, and the companion's pinned sample bundle from
  `examples/sample-bundle/`) with matching `CARRIED_HASHES`; the lock pins every `pyproject.toml` runtime pin with hashes;
  the install cell builds a managed-CPython virtual environment with a pinned `uv` (size + SHA-256 checked) and
  `--require-hashes`, keyed on the lock digest, drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP` and sets `MPLBACKEND=Agg`
  for the stages; the four Infrastructure cells are titled and collapsed; every learner cell runs a stage (`run_stage`);
  no "restart the runtime" instruction anywhere; the notebook byte-identical to `tools/build_notebook.py` output;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried package (and repeated in the carried manifest, which the
  `weights` stage checks against the package before fetching), the revision is a 40-hex immutable commit, and the same
  identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific calls in the stage runners — E2E: `stage_missing_files`, `verify_snapshot`,
  `TabICLClassificationPipeline.from_pretrained(weights_dir=..., n_estimators=..., random_state=42)`, `validate_inputs`
  (with the too-few-rows rejection probe), the BYOD checks (missing target column named with the header, numeric columns
  with stray strings refused unless listed in `TEXT_COLUMNS`, the dropped missing-target count carried into the input
  manifest), `prepare_classification_table`, `fit_categorical_encoder`, `majority_class_baseline`, `fit`, the bounded
  `create_finetuned_classifier` / `fine_tune_classifier` run (3 epochs, 300 s, patience 2) with holdout-only selection
  (`compare_metric`, `MIN_SELECTION_HOLDOUT_ROWS`) and a recorded skip on CPU, LightGBM / Random Forest on the same
  partitions with holdout error counts and the blend's tied-weight range, `evaluation_report` with the independent test,
  `read_inference_csv` + inference-mode `validate_inputs`, the bundle export with digests, the payload allowlist and the
  printed ZIP SHA-256, `safe_extract_zip` + `verify_artifact_bundle` on the fresh reload and the `rtol=1e-5, atol=1e-7`
  equivalence check; companion: the trusted-digest check (`EXPECTED_ZIP_SHA256`) before `safe_extract_zip`,
  `validate_artifact_runtime`, the base-model identity check including the pretrained checkpoint binding
  (`digests.checkpointSha256 == BASE_MODEL_SHA256`), `verify_artifact_bundle`, reconstruction on the bundled checkpoint with
  `allow_auto_download=False`, inference-mode `validate_inputs` with a rejection probe, the numeric-feature check,
  `apply_categorical_encoder`, `predict`, a `not-measurable` `evaluation_report` — the ceiling prints, the four exports per
  notebook, the learner-facing statements, the form-parameter defaults (`USE_BYOD`, the BYOD path fields,
  `RUN_NEW_DATA_INFERENCE`, `RUN_ACTIVITY` off; the companion's `ARTIFACT_ZIP_PATH`, `UPLOAD_ARTIFACT`,
  `EXPECTED_ZIP_SHA256`, `NEW_DATA_PATH`, `UPLOAD_NEW_DATA` empty/off), and no quality `assert` in the stage runners;
  forbidden patterns (credential-in-URL, own-repository clone/install, a `git+https://` dependency without a 40-hex SHA, a
  mutable `revision='main'`, direct `from tabicl import` / `TabICLClassifier(` / `FinetunedTabICLClassifier(` /
  `hf_hub_download(` / `sklearn` use **in the notebook's own cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`,
  `extractall(`; in the companion's runner also `load_breast_cancer(`, `load_wine(`, `check_stratifiable(`,
  `majority_class_baseline(`, `shutil.make_archive(`, any fine-tuning call);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and the `## Checkpoint Provenance` section.

CI also runs `ruff`, `tools/build_notebook.py --check` for both templates, `scripts/validate_colab_tutorial.py` (the
repository's own checks: exact build backend, release lock files and their agreement with the pins, the carried hash lock,
the fine-tune gate and its bounds, hardening markers), the two `scripts/test_*.py` classification suites and the offline
unit suite (`tests/`: snapshot, role and weight-fact helpers, `test_notebook_parity.py`, `test_companion_parity.py`, and
`test_notebook_review_fixes.py`, which execs the notebooks' own cells with stand-ins and runs the model-free stages; no
weights, no model). `tools/build_sample_bundle.py --check` reproduces `examples/sample-bundle/` byte for byte. These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab T4 GPU runtime (CPU runs everything except fine-tuning); any kernel Python — the stages run on the isolated environment's CPython 3.12.12 | The runtime the tutorials are written for; a clean one-pass **Run all** here is promotion evidence |
| Kaggle CLI kernel | Kaggle GPU kernel, any image Python | Reproducible clean-room executor of the same class; the notebook is pushed verbatim (no repository checkout is needed — the notebooks are standalone, and neither notebook's default path opens an upload dialog). For the companion's executor path, set `ARTIFACT_ZIP_PATH` to the E2E run's `outputs/tabiclv2_classifier_artifact.zip`, `EXPECTED_ZIP_SHA256` to the digest that run printed, and `NEW_DATA_PATH` to its `outputs/tabiclv2_classifier_new_rows.csv` |
| GitHub Actions `release-notebook-execution` (previous pair) | ubuntu, Python 3.13, `nbclient` | Executed the previous repository-installing pair through real kernels (`scripts/execute_notebook_release_paths.py`); it must be re-pointed at the standalone pair before it counts again |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a new **T4 GPU** runtime with **no repository checkout** and a clean model cache;
3. run it with one **Run all** and no restart, without editing implementation cells (form parameters at their defaults:
   `DATA_SOURCE = 'Sample: Breast Cancer'`, `USE_BYOD = False`, `RUN_FINE_TUNING = True`, `FINE_TUNE_EPOCHS = 3`,
   `EVAL_METRIC = 'accuracy'`, `RUN_NEW_DATA_INFERENCE = False`, `RUN_ACTIVITY = False`), then re-run the export cell
   once to confirm a second pass reuses the environment and reproduces the bundle digest;
4. verify that Section 2 reports the carried-file verification and the isolated environment's versions (CPython 3.12.12,
   `torch` 2.14.0, `tabicl` 2.1.1) with CUDA visible, and that the exported result records `NOTEBOOK_SOURCE` equal to
   `metadata.dimer.generated_from`;
5. verify every default-path stage completes:
   - pinned `jingang/TabICL` acquisition at the immutable revision: the carried manifest is checked against the package
     identity, `stage_missing_files` reports the one manifest entry (`tabicl-classifier-v2-20260212.ckpt`) on a clean
     runtime, and `verify_snapshot` returns its digest;
   - the Breast Cancer sample split 60/20/20 with stratification (`train=341, holdout=114, test=114`, two classes); the
     ceilings surfaced; `outputs/tabiclv2_classifier_input_manifest.json` with one recorded rejection finding;
   - in-context conditioning and the pretrained holdout/test metrics; the **fine-tuning stage ran** (epochs, the
     candidate's holdout metrics, the selection basis, `best.ckpt` kept only); LightGBM 0.9649 (4 errors) and the random
     forest 0.9561 (5 errors) on the holdout; the blend's tied-weight range;
   - `outputs/tabiclv2_classifier_evaluation_report.json` with verdict `sample-sanity`;
   - eight held-out rows scored into `outputs/tabiclv2_classifier_predictions.csv`, and their unlabelled copy
     `outputs/tabiclv2_classifier_new_rows.csv`;
   - the bundle `outputs/tabiclv2_classifier_artifact.zip` written with its SHA-256 printed and
     `matches_pinned_sample_context` reported, reloaded in a fresh process with identical labels and equivalent
     probabilities;
6. in a **second** clean runtime, run the exact companion notebook revision with its defaults (the pinned sample bundle and
   rows; no upload) and confirm `trusted_digest: verified`, `mode: pretrained`, the checkpoint binding, 341 context rows, and
   the `not-measurable` report; then set `ARTIFACT_ZIP_PATH` to a copy of the E2E bundle, `EXPECTED_ZIP_SHA256` to the
   digest the E2E run printed and `NEW_DATA_PATH` to its `..._new_rows.csv`, and confirm the same checks pass before
   reconstruction;
7. verify the exports exist and the interpretation sections match the observed paths;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, tabicl, device), model identifier and
   immutable revision, whether the model cache was clean, `restarted: false`, outcome, produced outputs and metrics, and any
   warning or applicable `SHOULD` deviation in the table below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook file (verify with `git rev-parse <commit>:tutorials/<notebook>`). Wall
times, when recorded, are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `b1705c6` / `f1b597a0b6ad` | Kaggle T4 (`kurtvalcorza/dimer-nb2-tabiclv2-classifier` v1) | Default sample path | 251.4 s | **PASSED** — 9/9 ok code cells executed cleanly, 5 files, 110 MB staged |
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