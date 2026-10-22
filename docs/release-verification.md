# Release verification

`tutorials/tabiclv2_classifier_colab.ipynb` (`E2E`) is a **release candidate** until the exact notebook revision has
executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook
Specification 3.0. This file is the durable release-gate record.

**Notebook Spec 3.0 fold (2026-10-22).** The notebook now declares Notebook Spec 3.0 and closes with the external-artifact
section (Section 10, §19.1). The artifact-inference companion `tutorials/tabiclv2_classifier_artifact_inference_colab.ipynb`,
its template, its stage runner, `tests/test_companion_parity.py`, `tools/build_sample_bundle.py` and the Breast Cancer sample
bundle `examples/sample-bundle/` were deleted (KND11; the bundle was assembled by a script that ran no model and described an
earlier default, so it could not meet SART9/SART10). The section's default artifact is the release asset
`tabiclv2_classifier_artifact.zip` of tag `sample-artifact-magic-v1` (SHA-256
`2dc82d55161382602fa7a5d1e217e3a33b6c6188ba6ddf4f7ff6b087bf67114c`, 102,793,443 bytes): the deterministic export of the recorded
hosted T4 run of `383ca04` listed under *Recorded executions* (SART8). The section's own hosted evidence with that pin (the
second pass of §32.2) is recorded under *Fresh-test evaluation* (the `1bca212`, `d4172f5` and `1344dca` records); the notebook stays
Candidate for the reasons under *Current status*. Text below that describes the companion, its sample bundle or both notebooks describes revisions before the fold
and is kept as history (REL14); the companion's execution evidence is not evidence for the migrated notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks the notebook (the companion-specific items below applied before the Notebook Spec 3.0 fold; the same checks now bind the external-artifact section, plus the pinned URL and digest, the digest check before extraction, the SART10 contract check and the `outputs/external_artifact/` exports):

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

CI also runs `ruff`, `tools/build_notebook.py --check`, `scripts/validate_colab_tutorial.py` (the
repository's own checks: exact build backend, release lock files and their agreement with the pins, the carried hash lock,
the fine-tune gate and its bounds, hardening markers), the two `scripts/test_*.py` classification suites and the offline
unit suite (`tests/`: snapshot, role and weight-fact helpers, `test_notebook_parity.py`, `test_external_artifact_section.py`, and
`test_notebook_review_fixes.py`, which execs the notebooks' own cells with stand-ins and runs the model-free stages; no
weights, no model). These are
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
   `DATA_SOURCE = 'Sample: MAGIC Gamma Telescope'`, `USE_BYOD = False`, `RUN_FINE_TUNING = True`, `GRID_EPOCHS_CAP = 0`,
   `GRID_LEARNING_RATE = 0.0`, `RANDOM_STATE = 42`, `RUN_NEW_DATA_INFERENCE = False`, `RUN_ACTIVITY = False`; `DIMER_MASK_FRESH_TEST`
   unset only for the run that is to score the fresh test — see *Fresh-test evaluation* below), then re-run the export cell
   once to confirm a second pass reuses the environment and reproduces the bundle digest;
4. verify that Section 2 reports the carried-file verification and the isolated environment's versions (CPython 3.12.12,
   `torch` 2.14.0, `tabicl` 2.1.1) with CUDA visible, and that the exported result records `NOTEBOOK_SOURCE` equal to
   `metadata.dimer.generated_from`;
5. verify every default-path stage completes:
   - pinned `jingang/TabICL` acquisition at the immutable revision: the carried manifest is checked against the package
     identity, `stage_missing_files` reports the one manifest entry (`tabicl-classifier-v2-20260212.ckpt`) on a clean
     runtime, and `verify_snapshot` returns its digest;
   - (the 2026-10-10 record below used the earlier Breast Cancer default, `train=341, holdout=114, test=114`; from 2026-10-19 the default is the
     pinned MAGIC split, `train=3000, holdout(validation)=1000`, fresh test 1500 read only by `fresh_test`); the
     ceilings surfaced; `outputs/tabiclv2_classifier_input_manifest.json` with one recorded rejection finding;
   - in-context conditioning and the pretrained holdout/test metrics; the **fine-tuning stage ran** (epochs, the
     candidate's holdout metrics, the selection basis, `best.ckpt` kept only); on the earlier Breast Cancer default (Colab T4,
     2026-10-10, blob `051c81e9`) LightGBM 0.9649 (4 errors) and the random forest 0.9561 (5 errors) on the holdout; the blend's
     tied-weight range;
   - `outputs/tabiclv2_classifier_evaluation_report.json` with verdict `sample-sanity`;
   - eight held-out rows scored into `outputs/tabiclv2_classifier_predictions.csv`, and their unlabelled copy
     `outputs/tabiclv2_classifier_new_rows.csv`;
   - the bundle `outputs/tabiclv2_classifier_artifact.zip` written with its SHA-256 printed, reloaded in a fresh process
     with identical labels and equivalent probabilities;
6. in the same run, confirm the closing external-artifact section (Section 10): the pinned URL and SHA-256 printed before the
   download, `trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`, `zip_sha256` `2dc82d55…`, `mode: pretrained` with the
   checkpoint bound to the base, `data_contract` matching the MAGIC default path, 3,000 context rows, eight MAGIC rows scored,
   and `outputs/external_artifact/tabiclv2_classifier_external_predictions.csv` and `..._external_result.json` written. A
   tampered or wrong pin must stop the run before extraction;
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
| 2026-09-14 | `b1705c6` / `f1b597a0b6ad` | Kaggle T4 (`kurtvalcorza/dimer-nb2-tabiclv2-classifier` v1) | Default sample path of an **earlier** E2E blob (spec 1.1 carrier, in-kernel install) | 251.4 s | **PASSED** — 9/9 ok code cells, 5 files, 110 MB staged. Kaggle image, runtime package versions and the `restarted` flag were not recorded and no metrics were captured. History only: not evidence for the current blobs (REL14) |
| 2026-10-10 | `f59e233` / `051c81e9d8fd` of `tabiclv2_classifier_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/tabiclv2_classifier_colab_f59e233_colab-cli-t4.ipynb`, sha256 `a3f0364b49fe22d64986ff4bbf59f1dced25a48798f93df87f88b92449deb19c` (`…_run_summary.json` `ef448653dd85…`, `…_exec.log` `9b3074714ee3…`) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (15,360 MiB, reported by cell 1), via the workspace `colab-cli-serial-test-suite`; isolated uv CPython 3.12.12 environment from the carried hash lock (90 packages; torch 2.14.0, tabicl 2.1.1, numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, lightgbm 4.7.0). Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/11–11/11) | E2E default path incl. the bounded CUDA fine-tune: Breast Cancer sample 341 / 114 / 114, device cuda; checkpoint fetched from the Hub at revision `4dcd344ece2c` and digest-verified (`bdc7dbd5…`) | 185.7 s session wall | **PASSED — one pass, no restart, 0 errors, 11/11 code cells.** Too-few-rows probe rejected; holdout: TabICL pretrained accuracy 0.9825, ROC AUC 0.9993; LightGBM 0.9649; random forest 0.9561; majority class 0.6228; independent test TabICL 0.9912 (all equal to the CPU lock-only drive). Fine-tune requested (3 epochs / 300 s) and ran on cuda, stopping early after about 2.7 s (progress bar: two epoch updates, holdout accuracy flat at 0.9825); its best holdout 0.9825 tied the pretrained model, so `recommended_for_export` was `pretrained` and the export ZIP carries the 110,368,038-byte base checkpoint; the 331,004,321-byte fine-tuned checkpoint was neither exported nor reloaded. Fresh-process reload PASSED: labels identical, max abs probability difference 0.0. Wine activity not run (`RUN_ACTIVITY` off). Six scikit-learn `y_prob values do not sum to one` UserWarnings. Carried provenance names `fba80c6` (generation-time HEAD, not `f59e233`). Status stays **Candidate** |
| 2026-10-10 | `f59e233` / `cb92a9e84376` of `tabiclv2_classifier_artifact_inference_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/tabiclv2_classifier_artifact_inference_colab_f59e233_colab-cli-t4.ipynb`, sha256 `6bbc9b036a81175cf8bd4e93ebc575a415aacfd944e2a6350f046e2a82c6ed81` (`…_run_summary.json` `44bcf5e26b4f…`, `…_exec.log` `3c1663ee88dc…`) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`; same 90-package isolated uv environment. Not a browser Run all; order from the CLI log (1/9–9/9) | Companion default path: the in-repo pinned sample bundle (no release, no upload dialog), checkpoint fetched from the Hub and digest-verified | 128.6 s session wall | **PASSED — one pass, no restart, 0 errors, 9/9 code cells.** No upload dialog (no output asks for one); `artifact.json` verified against the pinned sample record; checkpoint `bdc7dbd5e4ff21f8…` verified and bound (identical to the pinned base); reconstructed on cuda from 341 context rows; missing-column probe rejected; 8-row labels and probabilities identical to the E2E run's (all eight rows are `benign`, so the label match is a weak check). Optional activity not run. Carried provenance `fba80c6`. Status stays **Candidate** |

### Local lock-only drives (not clean-runtime evidence)

These are development checks of the current source, run with the notebooks' own install cell (the hash lock, CPython
3.12.12, `--require-hashes --only-binary :all:`) and every code cell driven in order by a plain-CPython stand-in kernel,
GPU hidden (`CUDA_VISIBLE_DEVICES=-1`), real pinned checkpoint (`bdc7dbd5…`, staged from a local copy, so the Hub download
was not exercised). They are not hosted runs and do not change the status.

| Date (UTC) | Source | Path | Wall | Result |
|---|---|---|---|---|
| 2026-10-11 | E2E `051c81e9d8fd` (generated at `fba80c6`) | defaults + `RUN_ACTIVITY` (Wine) on CPU | 186 s (environment reused; first build 408 s) | 11/11 cells. Fine-tuning skipped with the recorded CPU reason. Holdout (114 rows): TabICLv2 accuracy 0.9825 (2 errors), ROC-AUC 0.9993, log loss 0.0409; LightGBM 0.9649 (4 errors); random forest 0.9561 (5 errors); majority 0.6228. Independent test: TabICLv2 0.9912, LightGBM 0.9737. Blend ties over weights 0.50–1.00. `matches_pinned_sample_context: True`; fresh-process reload PASS (max prob. diff 1.1e-10). Wine activity: 36 holdout rows, accuracy 0.9722 vs majority 0.3889 |
| 2026-10-11 | E2E fine-tuning branch, CPU stand-in for CUDA | the `condition` stage with `torch.cuda.is_available` patched for that stage only and `device='cpu'`, 1 epoch | 59 s | branch completes with the locked `tabicl` 2.1.1: `best.ckpt` only (331 MB), `selection: holdout:accuracy`, candidate holdout = pretrained, export stays `pretrained`. API check only; the T4 run is the first real execution |
| 2026-10-11 | companion `cb92a9e84376` | defaults (pinned sample bundle and rows) | 21 s | 9/9 cells, no dialog, `trusted_digest: verified`, checkpoint bound to the base; labels identical to the E2E run on the same eight rows, max prob. diff 1.1e-10 |
| 2026-10-11 | companion (previous blob, same stages) | the E2E ZIP by `ARTIFACT_ZIP_PATH` + its printed digest + its `new_rows.csv`, `RUN_ACTIVITY` | 66 s | 9/9 cells, `trusted_digest: verified (EXPECTED_ZIP_SHA256)`, `sample_kind: BYOD`, same predictions |

**Local-path redaction (2026-10-19, R96-m4).** The two 2026-10-10 `_exec.log` files carried the executor's local home directory in 2 lines each. That prefix was replaced by `<runner>/` in a separate commit; nothing else changed. SHA-256 before → after: `tabiclv2_classifier_colab_f59e233_colab-cli-t4_exec.log` `9b3074714ee35eb2830cc335c6fdd1e9e1ff8ccadfac161010d8220a56cf65a4` → `8dc1cfa361052293d3f042536759dee8a58b31d9f57889b505c360910e5d9821`; `tabiclv2_classifier_artifact_inference_colab_f59e233_colab-cli-t4_exec.log` `3c1663ee88dcf934dc2486ed0f6f0f87e86aea3401557daa2ab623e7fec3b4d5` → `fa82c03ad0cf79c5204776477c466c08cc1ff71494173777353233fd5a37e6b3`. The executed notebooks and run summaries carried no local path and are unchanged.

On a generic Linux kernel the `report` stage needs the system OpenMP library (`libgomp.so.1`, Debian/Ubuntu `libgomp1`)
for LightGBM; Colab and Kaggle images have it. The local drives supplied it from the locked `torch` wheel.

## Fresh-test evaluation (MAGIC, pre-registered 2026-10-19)

The settings, grid, selection and reading rule are fixed in [`magic-finetune-preregistration.md`](magic-finetune-preregistration.md)
(committed alone as `64c968c` before any implementing code). The 1,500-row fresh test is scored only by the `fresh_test` stage, only
when every pre-registered setting holds and the device is a Tesla T4; any other run writes `fresh-test/*_exploratory_<n>.json` and
reads nothing.

- **Evaluation of record (2026-10-20):** commit `9ea0888`, notebook blob `947feb162abbf8126a8579188268024da3ca3ba8`, Colab CLI
  sequential execution on a fresh hosted Tesla T4 (PASS, 12/12 code cells, 0 errors, 450.5 s wall), determinism flags on,
  `sdpa_backend` MATH, `grid_hash` `4b1bf5366b121dd2365c6c9b02408e1b825c247bb6636fee6d8fe35697b20692`. Selection: **pretrained kept**
  (no config beat validation log loss 0.2634 by more than 0.005). Fresh test (1,500 rows), labelled "pre-registered evaluation (this
  runtime)": primary endpoint (log-loss difference, selected − pretrained) **Δ = 0.0, 95 % CI [0, 0], verdict "same model (Δ = 0)"**
  (bootstrap seed 20261019, 10,000 resamples); accuracy 0.8993, Wilson 95 % [0.8831, 0.9136]; ROC AUC 0.95; without the 4 rows
  duplicated in train/val (n = 1,496) the same verdict. Artifact reload: labels identical, max abs probability difference 0.0.
  On the T4, C3 (lr 1e-6) ran all 10 epochs (60 optimizer steps taken, 0 AMP-skipped) and its weights changed (parameter delta
  0.0592); its exploratory, validation-only readout ends at log loss 0.2628 against the pretrained 0.2634 (printed difference
  −0.0005), far inside the 0.005 tolerance, so it was not selected. This differs from the CPU development run below, where C3
  stopped after 3 epochs with its weights restored. Evidence: [`execution-evidence/2026-10-20-9ea0888/`](execution-evidence/2026-10-20-9ea0888/)
  (executed notebook, `run_summary.json`, `exec.log`, the suite log, `fresh_test_report_transcription.json`). The executed
  notebook's cell sources equal blob `947feb16` (39 cells compared). **Gap:** the report file
  `outputs/fresh-test/tabiclv2_classifier_fresh_test_report.json` was not downloaded before the VM was released; the
  transcription holds every field the stage printed, and the unprinted fields (checks, determinism record, row digests, power,
  validation, notebook source) are not recoverable. `RECORDED_EVALUATION` in `tools/tutorial_stages.py` names this run; every later
  run of the pre-registered settings is a **reproduction** and is listed here.
- **Masked determinism check, attempt 1 (FAILED, history):** two masked hosted T4 runs at `maskcheck` `ae8d6ad` (blob `f5f0c1bb`), 96m1 and 96m2, printed identical flags but grid hashes `d7ad485b…` and `577e6953…`; C1's second training step already differed (0.3355 vs 0.3351). Both kept the pretrained model; 0 AMP-skipped steps; the fresh test was not read. Amendment 2 (`c718438`) pre-registers the MATH attention backend, and the check restarts.
- **Masked determinism check, attempt 2 (PASSED):** two masked hosted T4 runs (`DIMER_MASK_FRESH_TEST=1`) under amendment 2 on the
  throwaway branch `fix/tabicl-tutorial-review-2026-10-17-maskcheck2` (`b66a30f`, blob `652c0015`, never merged), 96m3 and 96m4 (96m4
  reused 96m3's run directory, so its suite log is its record). Both printed `grid_hash` `4b1bf536…` and the same 27 printed result
  lines apart from timings (`seconds`, `setup_seconds`, latency), run ids and the artifact ZIP SHA-256 (`753f347c…` vs `b24167ed…`,
  explained below). The run of record's grid rows and exploratory readout are byte-identical to 96m4's once `seconds` is removed.
  Hash table: `execution-evidence/2026-10-20-9ea0888/masked_determinism_check.json`; logs under `masked-determinism/` there.
- **Why the artifact ZIP digest differs between deterministic runs.** `stage_export` at `9ea0888` builds the ZIP with
  `shutil.make_archive`, which stamps each member with its file-system modification time; the members are written during the run,
  so identical members give different ZIP bytes. The members themselves matched in every masked run: `checkpoint_bytes`
  110,368,038 (the pinned base checkpoint, as the pretrained model was exported) and `training_context_sha256` `ccbad0fa…` printed
  identically; the manifest is built from those digests and the carried `source.json`. Because the companion notebook checks
  `EXPECTED_ZIP_SHA256` against the digest printed by **the same** E2E run that wrote the ZIP, the guidance stays correct; a digest
  copied from another run (or from this record, `ef7a13f5…`) would not match. The export now writes the ZIP deterministically
  (sorted members, fixed timestamp and permissions); see the code commit after this record.
- **Reproduction 1 (hosted Colab T4, 2026-10-20, `fb16f81` / blob `baf03478`): results reproduced, grid hash not.** The first
  run of the record-pinned notebook in a fresh runtime (account 1, started 2026-10-20T12:55:58Z, wall 370.2 s, exit 0, input
  unchanged; executed cell sources equal blob `baf03478`, 39 cells). Evidence:
  [`execution-evidence/2026-10-20-fb16f81/`](execution-evidence/2026-10-20-fb16f81/). It is labelled `reproduction`, selects
  `pretrained`, and reproduces every reported result: fresh-test log-loss difference 0.0, 95 % CI [0, 0], verdict
  **same model (Δ = 0)**; accuracy 0.8993 (Wilson [0.8831, 0.9136]); the same verdict without the 4 duplicated rows; artifact
  reload identical. It prints `matches_record: False`, `differs: ['grid_hash']`: `grid_hash` `2aa9e0cd…` against the record's
  `4b1bf536…`. The validation curves of the fine-tuned configurations differ in the fourth decimal (C1 epoch-1 step losses agree
  at step 1, 0.2437, then 0.3341 vs 0.3344 at step 2; C3 last-epoch validation log loss 0.2628 vs 0.2627), with the same torch
  2.14.0 / tabicl 2.1.1 / 90 locked packages, the MATH attention backend active and every determinism flag set. Three earlier
  hosted runs (96m3, 96m4, the record) agreed exactly; this one did not. It was first read as a residual non-deterministic step in
  tabicl's fp16 fine-tuning on the T4; PR #30 then attributed it to the stage-runner revision, and reproductions 4 and 5 (same
  revision, two grid hashes) show that run-to-run variation does occur (issue #28, revised finding below). The selection cannot flip on a difference of
  this size: the exported system is the pretrained checkpoint, which fine-tuning never touches, and the closest configuration is
  about 0.0006 better on validation against the 0.005 tolerance. Owner ruling (2026-10-20): disclose and merge; the remaining op
  is tracked as a follow-up. The artifact ZIP (`2dc82d55…`) is the first written by the deterministic exporter.
- **Reproduction 2 and the producing run of the sample artifact (hosted Colab T4, 2026-10-21, `383ca04` / blob `baf03478`).**
  The default path of `tutorials/tabiclv2_classifier_colab.ipynb` at `main` `383ca04` (blob `baf03478`), run by the Colab CLI
  (`colab exec -f`, 0.7.4) on a fresh hosted T4 session, started 2026-10-21T22:56:07Z, wall 363.3 s, exit 0, input unchanged,
  12/12 code cells, verdict PASS. Evidence:
  [`execution-evidence/2026-10-21-383ca04/`](execution-evidence/2026-10-21-383ca04/). It is labelled `reproduction` (the
  second in a new runtime; the runtime-local report file is named `..._fresh_test_reproduction_1.json`), selects `pretrained`
  and reproduces every reported result: fresh-test difference 0.0, 95 % CI [0, 0], **same model (Δ = 0)**; accuracy 0.8993
  (Wilson [0.8831, 0.9136]). Like reproduction 1 it prints `matches_record: False`, `differs: ['grid_hash']` (`2aa9e0cd…`,
  the same value as reproduction 1; a different stage-runner revision from the record, issue #28); the verdict is unchanged. Its deterministic export
  `outputs/tabiclv2_classifier_artifact.zip` (SHA-256 `2dc82d55161382602fa7a5d1e217e3a33b6c6188ba6ddf4f7ff6b087bf67114c`,
  102,793,443 bytes) is published unchanged as the release asset of tag `sample-artifact-magic-v1` (target `383ca04`), whose
  notes name this run (SART8). It is the pinned default artifact of the external-artifact section (Section 10) from the
  Notebook Spec 3.0 fold on; it is not evidence for the migrated notebook, which needs its own hosted run.
- **Reproduction 3 and the first hosted run of the migrated notebook (hosted Colab T4, 2026-10-21, `1bca212` / blob
  `822549fb`, Notebook Spec 3.0).** The default path of the folded E2E notebook on PR #29, run by the Colab CLI (`colab exec -f`)
  on a fresh hosted T4 session, started 2026-10-21T23:20:35Z, wall 397.8 s, exit 0, input unchanged; executed cell sources
  equal blob `822549fb` (48 cells), verdict PASS. Evidence:
  [`execution-evidence/2026-10-21-1bca212/`](execution-evidence/2026-10-21-1bca212/). The fresh test is labelled
  `reproduction`, selects `pretrained` and repeats every reported result; like reproductions 1 and 2 it differs from the record
  only in `grid_hash` (`2aa9e0cd…`; a different stage-runner revision, issue #28). **Section 10 (external artifact) ran end to end on the default path:** it downloaded the
  pinned release asset of tag `sample-artifact-magic-v1`, verified SHA-256
  `2dc82d55161382602fa7a5d1e217e3a33b6c6188ba6ddf4f7ff6b087bf67114c` before extraction, confirmed the checkpoint binding,
  rebuilt the classifier from the files alone, scored the sample MAGIC rows and wrote
  `outputs/external_artifact/tabiclv2_classifier_external_predictions.csv` and `..._external_result.json` (printed in the
  executed notebook; the files stayed on the released VM). This is the second pass of §32.2 for this pipeline.
- **Reproduction 4: first hosted run of the revision-aware record check (hosted Colab T4, 2026-10-22, `d4172f5` / blob
  `f21920ab`; PR #30, issue #28).** Colab CLI `colab exec -f` on a fresh T4, started 2026-10-22T11:11:21Z, wall 523.6 s, exit 0,
  input unchanged; executed cell sources equal the blob (48 cells); verdict PASS. Evidence:
  [`execution-evidence/2026-10-22-d4172f5/`](execution-evidence/2026-10-22-d4172f5/). The fresh test is labelled `reproduction`,
  selects `pretrained` and repeats every reported result (difference 0.0, CI [0, 0], **same model (Δ = 0)**, accuracy 0.8993).
  The record comparison printed `stage_runner_sha256` record `cc604098…` vs this run `ee759496…`, `same_stage_runner: False`,
  `grid_hash: 'not comparable across revisions (…)'` and **`matches_record: True`, `differs: []`**; this run's grid hash was
  `2aa9e0cd…`. Section 10 verified the pinned asset `2dc82d55…` as before.
- **Reproduction 5: optional-activity run (hosted Colab T4, 2026-10-22, `edbe94d` / committed blob `f21920ab`,
  `RUN_ACTIVITY = True`).** A run copy of committed blob `f21920ab` with one override, `RUN_ACTIVITY = False` → `True` (run copy
  blob `a69defdd91f5`, SHA-256 `18c889577b602df042fb54f779579a3a84f2e56dfe810e0a28f42e1bd9458625`; it differs from the committed
  blob only in that assignment literal and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by Colab CLI
  0.7.5 `colab exec -f` (a scripted run, not a browser Run all) on a fresh T4, started 2026-10-22T13:24:55Z, wall 480.6 s, exit 0,
  sequential, input (the run copy) unchanged, 15/15 code cells, 0 errors, verdict PASS. Evidence:
  [`execution-evidence/2026-10-22-edbe94d-activity/`](execution-evidence/2026-10-22-edbe94d-activity/). The fresh test was not masked (`DIMER_MASK_FRESH_TEST = unset`); it is
  labelled `reproduction`, selects `pretrained` and repeats every reported result (difference 0.0, CI [0, 0], **same model
  (Δ = 0)**, accuracy 0.8993), with `same_stage_runner: False` against the record and **`matches_record: True`, `differs: []`**.
  The export reloaded with `labels_identical: True`, and Section 10 verified the pinned asset `2dc82d55…`. The Section 11 Wine
  activity printed `'activity_sample': 'sklearn-wine'`, 36 holdout rows (`one_row_resolution` 0.0278), TabICL pretrained holdout
  accuracy 0.9722 / balanced accuracy 0.9762 / log loss 0.0518 / ROC AUC 1.0 against the majority-class baseline's accuracy
  0.3889, and `{'canonical_outputs_unchanged': True}`. **Grid hash:** this run printed `grid_hash` `4b1bf536…`, while the
  `d4172f5` run of the same blob and the same carried runner (`ee759496…`) printed `2aa9e0cd…` (both values are in the committed
  executed notebooks). This is the observation that reopened issue #28: the grid hash varies run to run at an identical revision
  (see *Issue #28* below).
- **Reproduction 6: first hosted run of the informational grid-hash check, default path (hosted Colab T4, `1344dca` / blob
  `515f8954`; PR #33, issue #28).** Colab CLI `colab exec -f` (a scripted run, not a browser Run all) on a fresh T4, started
  2026-10-22T22:07:41Z (2026-10-23 local, hence the directory date), wall 425.8 s, exit 0, input unchanged, 15/15 code cells,
  0 errors, verdict PASS. Evidence: [`execution-evidence/2026-10-23-1344dca/`](execution-evidence/2026-10-23-1344dca/). The
  fresh test was not masked (`DIMER_MASK_FRESH_TEST = unset`); it is labelled `reproduction`, selects `pretrained` and repeats
  every reported result (difference 0.0, CI [0, 0], **same model (Δ = 0)**, accuracy 0.8993, ROC AUC 0.95). The record
  comparison printed `stage_runner_sha256` record `cc604098…` vs this run `a0130b1e…`, `same_stage_runner: False`, `grid_hash`
  `{'informational': …, 'record': '4b1bf536…', 'this_run': '4b1bf536…', 'equal': True}` and **`matches_record: True`,
  `differs: []`**. The export reloaded with `labels_identical: True` (maximum probability difference 0.0), and Section 10
  verified the pinned asset `2dc82d55…` before extraction.
- **Reproduction 7: optional-activity run of the same blob (hosted Colab T4, `1344dca` / committed blob `515f8954`,
  `RUN_ACTIVITY = True`).** A run copy of committed blob `515f8954` with one override, `RUN_ACTIVITY = False` → `True` (run copy
  blob `12894013ec29`, SHA-256 `d5b5e63ae32908175674ad8f7f3da12e99e630978c3b88ac20af3a3ea4a2e039`; it differs from the committed
  blob only in that assignment literal, in cell 45 of 48, and is committed beside the evidence as `…_activity_run_copy.ipynb`),
  run by Colab CLI `colab exec -f` on a fresh T4, started 2026-10-22T22:15:47Z (2026-10-23 local), wall 440.0 s, exit 0,
  input (the run copy) unchanged, 15/15 code cells, 0 errors, verdict PASS. Evidence:
  [`execution-evidence/2026-10-23-1344dca-activity/`](execution-evidence/2026-10-23-1344dca-activity/). The fresh test was not
  masked; it is labelled `reproduction`, selects `pretrained` and repeats every reported result (difference 0.0, CI [0, 0],
  **same model (Δ = 0)**, accuracy 0.8993), with `same_stage_runner: False` against the record, `grid_hash`
  `{'informational': …, 'record': '4b1bf536…', 'this_run': '2aa9e0cd…', 'equal': False}` and **`matches_record: True`,
  `differs: []`**. The export reloaded with `labels_identical: True`, and Section 10 verified the pinned asset `2dc82d55…`. The
  Section 11 Wine activity printed `'activity_sample': 'sklearn-wine'`, 36 holdout rows (`one_row_resolution` 0.0278), TabICL
  pretrained holdout accuracy 0.9722 / balanced accuracy 0.9762 / log loss 0.0518 / ROC AUC 1.0 against the majority-class
  baseline's accuracy 0.3889, and `{'canonical_outputs_unchanged': True}`.
- **Fresh-test looks so far:** eight, the evaluation of record and reproductions 1–7 above. No CPU development look was taken: the
  fixer's CPU checks ran with the fresh test masked; the masked hosted runs (96m1–96m4) did not read it.

### Issue #28: the fp16 grid hash varies on the T4; it is reported, never compared (2026-10-23)

**Finding (revised 2026-10-23).** On a hosted Tesla T4 the fp16 fine-tuning grid hash has at least two outcomes for an identical
revision. Two hosted runs of the same committed notebook blob `f21920ab`, carrying the same stage runner (`ee759496…`), printed
different grid hashes: `d4172f5` (reproduction 4) printed `2aa9e0cd…` and `edbe94d` (reproduction 5, the optional activity runs
after the grid) printed `4b1bf536…`. In both runs, and in every run in the table below, the selection outcome and every fresh-test
value matched the record exactly. The first version of this section (PR #30) concluded that the hash was bit-exact per runner
revision; reproduction 5 contradicts that, so that conclusion is withdrawn. The cause of the variation is not established
(cuBLAS algorithm selection is plausible, not shown). The runs so far: runner digest = SHA-256 of the carried `tutorial_stages.py` (UTF-8, LF), as stamped in the notebook's
`NOTEBOOK_SOURCE['files']['tutorial_stages.py']`; it was checked against `git show <rev>:tools/tutorial_stages.py`.

| Run | Date (UTC) | Commit / notebook blob | Carried runner (source rev, SHA-256) | `grid_hash` |
|---|---|---|---|---|
| 96m3 (masked) | 2026-10-20 | `b66a30f` / `652c0015` | `15c1ae4`, `cc604098…` | `4b1bf536…` |
| 96m4 (masked) | 2026-10-20 | `b66a30f` / `652c0015` | `15c1ae4`, `cc604098…` | `4b1bf536…` |
| Evaluation of record | 2026-10-20 (from 12:31) | `9ea0888` / `947feb16` | `15c1ae4`, `cc604098…` | `4b1bf536…` |
| Reproduction 1 | 2026-10-20 (from 12:55) | `fb16f81` / `baf03478` | `bc7c606`, `9375a44d…` | `2aa9e0cd…` |
| Reproduction 2 | 2026-10-21 | `383ca04` / `baf03478` | `bc7c606`, `9375a44d…` | `2aa9e0cd…` |
| Reproduction 3 | 2026-10-21 | `1bca212` / `822549fb` | `d7426dc`, `6b64c594…` | `2aa9e0cd…` |
| Confound breaker (masked) | 2026-10-22 (about 10:46–10:53) | `b66a30f` / `652c0015` | `15c1ae4`, `cc604098…` | `4b1bf536…` |
| Reproduction 4 | 2026-10-22 (from 11:11) | `d4172f5` / `f21920ab` | `d4172f5`, `ee759496…` | `2aa9e0cd…` |
| Reproduction 5 (activity) | 2026-10-22 (from 13:24) | `edbe94d` / `f21920ab` | `d4172f5`, `ee759496…` | `4b1bf536…` |
| Reproduction 6 | 2026-10-22 (from 22:07) | `1344dca` / `515f8954` | `210c75c`, `a0130b1e…` | `4b1bf536…` |
| Reproduction 7 (activity) | 2026-10-22 (from 22:15) | `1344dca` / `515f8954` | `210c75c`, `a0130b1e…` | `2aa9e0cd…` |

Reproductions 4 and 5 carry the same blob and the same runner and print different grid hashes; that pair is the evidence for the
revised finding. Reproductions 6 and 7 repeat the observation on blob `515f8954`: the same blob and runner printed `4b1bf536…` and
`2aa9e0cd…`, and both runs passed with `matches_record: True`, `differs: []`, because the check now reports the grid hash as
informational and never compares it. That is the fix working: under PR #30's rule a record refreshed on runner `a0130b1e…` would
have failed one of the two on the hash alone. The paragraphs from *Confound breaker* to *Effect size* are the PR #30 analysis, kept as history; its revision
explanation does not account for reproductions 4 and 5.

The 96m3/96m4 suite logs carry blob `652c0015`; its `notebook_source` is `15c1ae4`, and `b66a30f` differs from `15c1ae4` only by the
inserted mask cell. **Confound breaker.** Until 2026-10-22 revision and time were fully confounded: every `4b1bf536…` run was on
2026-10-20 up to the record, and every `2aa9e0cd…` run came after it, so a Colab image change could not be excluded. The
2026-10-22 run of the unchanged `b66a30f` notebook on a fresh hosted T4 printed `4b1bf536…` again, two days later, under the
current image. That rules out an image change as the cause. A diagnostic run on the same day recorded the GPU stack: driver
580.82.07, torch CUDA 13.0, cuDNN 92400, Tesla T4 (sm 7.5). It was a throwaway diagnostic revision with the grid in fp32, so it is
not a reproduction, and its grid hash is not comparable.

**What changed between the groups (PR #30 analysis, superseded by reproductions 4 and 5).** From `15c1ae4` to `bc7c606` only the stage runner changes (and template prose). The added
code (the record, `printed_reading`, `compare_with_record`, the deterministic ZIP writer) runs only in `fresh_test` and `export`,
both after the grid. Seeds, determinism settings, grid, data, encoding, the lock and `src/` are unchanged. **Mechanism (inferred,
not proven):** the size and content of the runner module change the process's allocation history and so the memory layout or
alignment of buffers used during the grid. That can change which cuBLAS kernel runs, or how a SIMD reduction groups its terms, and
fp16 training then amplifies the difference. No changed op, seed or RNG use was found before or inside training. The `d7426dc`
runner (a large change) still prints `2aa9e0cd…`, so not every code change moves the hash.

**Effect size.** About 3e-4 in validation log loss (for example C3's last epoch, 0.2628 vs 0.2627). The pretrained model is selected
in every run, and every fresh-test result is identical in every run (Δ = 0.0, accuracy 0.8993, ROC AUC 0.95, the same sensitivity
verdict).

**Consequence for the record check (revised 2026-10-23).** `compare_with_record` never gates on `grid_hash`, whatever the stage
runner. It prints `grid_hash` as `{'informational': …, 'record': <record's hash>, 'this_run': <this run's hash>, 'equal': True|False}`
and never adds it to `differs`. The reproducibility claim is the exact match of the pre-registered selection outcome, the device and
every fresh-test value, which all must match for `matches_record: True`. `RECORDED_EVALUATION` still names the runner that produced
the record (`stage_runner_sha256` `cc604098aa235653c2ef5b36cffbca37332539a9ac74e1b52912e1ff71184823`, unchanged); both digests and
`same_stage_runner` are printed for information and also never gate. Under PR #30's rule a record refreshed on runner `ee759496…`
would have failed reproduction 4 or 5 on the grid hash alone; under this rule both read `matches_record: True`, as do reproductions
1–3.

**Pre-registration.** No amendment in this change. The amendment-2 check (two masked runs at one revision with identical grid
hashes) was met by 96m3 and 96m4 at `b66a30f` before any run scored the fresh test. Reproductions 4 and 5 show that such agreement
is not guaranteed on the T4, so the grid hash is not evidence of reproducibility on its own.

**Local-path redaction of the reproduction 7 (optional-activity) evidence.** Local paths were replaced by `<local-run-dir>/` before
the first commit. The executed notebook was checked against `run_summary.json`'s `executed_sha256` before any edit.

| File (under `execution-evidence/2026-10-23-1344dca-activity/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_classifier_colab_1344dca_colab-cli-t4_activity.ipynb` | `63d35d62b9248be40f62fab8afef419c5578b923ea222a635761eab2e5187f5c` | unchanged (no local path) |
| `tabiclv2_classifier_colab_1344dca_colab-cli-t4_activity_exec.log` | `c6e48ec7dc1e6699aaf77d9b6b03b74b80484ff2e064ca9e2af385e292932746` | `674b9d3bb9dacd8c2ff73aaa0039b509234c9c199913e7f9be0398a236d02aeb` (2 replacements) |
| `tabiclv2_classifier_colab_1344dca_colab-cli-t4_activity_run_summary.json` | `6586d062e2aceb37f9b88416d39469d782f35989b0e3feb61e0cd23418977c0f` | unchanged (no local path) |
| `tabiclv2_classifier_colab_1344dca_colab-cli-t4_activity_run_copy.ipynb` | `d5b5e63ae32908175674ad8f7f3da12e99e630978c3b88ac20af3a3ea4a2e039` | unchanged (no local path) |

**Local-path redaction of the reproduction 6 evidence.** Local paths were replaced by `<local-run-dir>/` before the first commit.
The executed notebook was checked against `run_summary.json`'s `executed_sha256` before any edit.

| File (under `execution-evidence/2026-10-23-1344dca/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_classifier_colab_1344dca_colab-cli-t4.ipynb` | `1458d1a7c418e6a380589516b02601a2f809be612b86aabcab2a540dff45f69b` | unchanged (no local path) |
| `tabiclv2_classifier_colab_1344dca_colab-cli-t4_exec.log` | `25663ae5f888d57ac5afea5231b19ba2301f787df94dcbb991d68d3b27d98d78` | `1f0701b11144a3976c449467f52d510c3c7437aafd0bc417422fcd119cf848b9` (2 replacements) |
| `tabiclv2_classifier_colab_1344dca_colab-cli-t4_run_summary.json` | `133b05458e2eafabd03e9898841baa626c699b1a7716204e9d8efbfe856fcd2a` | unchanged (no local path) |

**Local-path redaction of the reproduction 5 (optional-activity) evidence.** Local paths were replaced by `<local-run-dir>` before the
first commit.

| File (under `execution-evidence/2026-10-22-edbe94d-activity/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_classifier_colab_edbe94d_colab-cli-t4_activity.ipynb` | `589449a1b829e13fd21c3656c1dcf669d6959cca6c210c04afe027bd75dfa78b` | unchanged (no local path) |
| `tabiclv2_classifier_colab_edbe94d_colab-cli-t4_activity_exec.log` | `3dcb60b33bdf177b1d812d5a591672865547db72c139b4a2b7a8b8ebf65c3df2` | `89424dfb2576485670930f98fd0b87d17d6c3b81ee0979d315c160d4117e8a69` (2 replacements) |
| `tabiclv2_classifier_colab_edbe94d_colab-cli-t4_activity_run_summary.json` | `c590f8c3ba197c8cb907dac3489660e8e5b1b07404dcc5cb518380beee4ca104` | unchanged (no local path) |
| `tabiclv2_classifier_colab_edbe94d_colab-cli-t4_activity_run_copy.ipynb` | `18c889577b602df042fb54f779579a3a84f2e56dfe810e0a28f42e1bd9458625` | unchanged (no local path) |

**Local-path redaction of the reproduction 4 evidence.** Local paths were replaced by `<local-run-dir>/` before the first commit.

| File (under `execution-evidence/2026-10-22-d4172f5/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_classifier_colab_d4172f5_colab-cli-t4.ipynb` | `d622d5c1a62f92b156c39cc4a835e9fd32a9be502e8d5ae27f5f55fb829b2cca` | unchanged (no local path) |
| `tabiclv2_classifier_colab_d4172f5_colab-cli-t4_run_summary.json` | `021c4c56520981821ab9637eacdad37267a7cc02f3645d936e5c14988bb1c502` | unchanged (no local path) |
| `tabiclv2_classifier_colab_d4172f5_colab-cli-t4_exec.log` | `34d58d712543cddb9cb71eeb2a729e25a692cb8e2c32d8de1ec0bb41d89efd55` | `e11b9f84b7459a7266f63215b965254eebba74321d6caae46bad86bc2d041f86` |
| `tabiclv2_classifier_colab_d4172f5_colab-cli-t4_suite.log` | `f47918d5facc93a2313a118f23dcd4b526d29d69b64b311fade9a754ed0229f4` | `e7ec4a629c09bd90215d224e28c21e2247b4d3d83e13db68a49928399b8d1ca2` |

**Local-path redaction of the reproduction 3 evidence.** Local paths were replaced by `<local-run-dir>/` before the first commit.

| File (under `execution-evidence/2026-10-21-1bca212/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_classifier_colab_1bca212_colab-cli-t4.ipynb` | `9703397cc8e3ceaf45f5b80258e396c66ebdfa0c1a28084f42e698ae4f352a9e` | unchanged (no local path) |
| `tabiclv2_classifier_colab_1bca212_colab-cli-t4_run_summary.json` | `5e31302c63f219c658d1f663baf1a5f06d54b8f571b9f20c3ebea96dbfdad27a` | unchanged (no local path) |
| `tabiclv2_classifier_colab_1bca212_colab-cli-t4_exec.log` | `00cc231d30ce86fa7527ae1a1591646e7ead49299ba1c1461edd37c0753add3b` | `153da19297dde235c9b7b8863924e69e46faf76245d84fba3f5665bdee29565e` (2 replacements) |
| `tabiclv2_classifier_colab_1bca212_colab-cli-t4_suite.log` | `037acf788819871be31f08b74be34fa8afb8225edec5d8ee22aaa09760968cb4` | `aeffd16c551f0d329b0ad569b157b601bc7f8b1f60ff88b28cc81318229e206a` (4 replacements) |

**Local-path redaction of the reproduction 2 evidence.** Local paths were replaced by `<local-run-dir>/` before the first
commit. The 102,793,443-byte artifact ZIP is not committed here; it is the release asset.

| File (under `execution-evidence/2026-10-21-383ca04/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_classifier_colab_383ca04_colab-cli-t4.ipynb` | `4dd5ebc255d3b6319be31d27cba68241d1ffd2798e0e70cc0ee9fc4c6af398c6` | unchanged (no local path) |
| `tabiclv2_classifier_colab_383ca04_colab-cli-t4_exec.log` | `b4f51655caa5ac961b065cccdf32e0436ac32a2ce85ff4d2316636cf72dd1dbd` | `df81c9a2c9b1aa405d30df11a5f23de003f4ce9ece0f00ce9c76071e4ca58072` |
| `tabiclv2_classifier_colab_383ca04_colab-cli-t4_run_summary.json` | `3705679994220e272ca2c8df59340a6910631d0de6e410a7f37bc6e372ec98b9` | unchanged (no local path) |
| `tabiclv2_classifier_colab_383ca04_colab-cli-t4_fresh_test_reproduction_1.json` | `32e10e087964b11cad3a04d296441fc07294b837d6cb9522db6bb52c0003be89` | unchanged (no local path) |
| `tabiclv2_classifier_colab_383ca04_colab-cli-t4_suite.log` | `bb50d38b0d8a9d19563680d4ffb9f98063ed5a48d8ee72efc79cbb22dd8514a8` | `a992ed43af67a6514c9db2065b5ed9bb90673a95d5de62e3061ac679893177ea` |

**Local-path redaction of the reproduction evidence.** Local paths were replaced by `<local-run-dir>/` before the first commit.

| File (under `execution-evidence/2026-10-20-fb16f81/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_classifier_colab_fb16f81_colab-cli-t4.ipynb` | `da2513c2a2a4acfb840605d6bea0782577a0da986b3eccc8ee1eb6a1de807d0b` | unchanged (no local path) |
| `tabiclv2_classifier_colab_fb16f81_colab-cli-t4_run_summary.json` | `90c9eb7a463313dd134cdf8678abd42582f2f2194d9b29ab29aeecad78b2fad0` | unchanged (no local path) |
| `tabiclv2_classifier_colab_fb16f81_colab-cli-t4_exec.log` | `e3a3cd18bc5d9bbda8f22dfc2603fe9a00b3771b297eb0b2703dfdb7d32e21b1` | `e36f4c883b506b9adc5cac7a0d8eb992bf46549fe9661dbbb921bf795f5b1a3d` (2 replacements) |
| `tabiclv2_classifier_colab_fb16f81_colab-cli-t4_suite.log` | `5a9cc6d9a158961dcfe6d60f2e9c890b8baa3edebbf18d6d3ea6ac358649f8ab` | `372fd48d5459ba9a92e7165340120ee3d95aff81fc383c13fb320d6a4d283a6a` (4 replacements) |

**Local-path redaction of the 2026-10-20 evidence.** Local run-directory paths were replaced by `<local-run-dir>/` before the first
commit. SHA-256 captured → committed:

| File (under `execution-evidence/2026-10-20-9ea0888/`) | Captured | Committed |
|---|---|---|
| `tabiclv2_classifier_colab_9ea0888_colab-cli-t4.ipynb` | `2ab26d8aca414a00528de09d5e8a9fa352ec22ed4399a6223c8e85ef06a83e79` | unchanged (no local path) |
| `tabiclv2_classifier_colab_9ea0888_colab-cli-t4_run_summary.json` | `e480bebc1ba051860d51ff0fd19f8587a476a57902130bb4c4bf09863ee99bd6` | unchanged (no local path) |
| `tabiclv2_classifier_colab_9ea0888_colab-cli-t4_exec.log` | `83be01078f114de586852f7fabb1dcbe9275e202caa60df03da80e629212cbec` | `1387c7ee2a43127ba5a6595b8357eae211cf331a38bfde4c414f4b8457fc70b2` |
| `tabiclv2_classifier_colab_9ea0888_colab-cli-t4_suite.log` | `ef7aee31c521af8c30ce9739842897d7022262257c4347fc6e4af96142760e4c` | `ac59b8da0d8f74b1a0cd5fa8d296f8fc72dcd2f45be0271a57091f5fac217dda` |
| `masked-determinism/96m1_ae8d6ad_suite.log` | `f6195bd6af3d34dbcf205940d4ce4cb3d1251ffad9aa71deafed345d777b5d0a` | `5c538dccbfaf3fc25a33a5fba59957727f183f0ab68e3da46953de97e87eb5b5` |
| `masked-determinism/96m2_ae8d6ad_suite.log` | `4b0a53774627493c07f4674a5546d18e83a92baf5dfb62c837d38899edbf1d02` | `cf01f467df4e1723a3533a8fce0791f8a2ecb4a477dbda8aaca033632e05c699` |
| `masked-determinism/96m3_b66a30f_suite.log` | `64af358c0a7248b185fa9d9bc2a705dce94a1c7129e0925d269adfb733470380` | `03bd63f25c193a526dd8fbef2b7b7d09c500bd712f00c9e7662ebae20c151629` |
| `masked-determinism/96m4_b66a30f_suite.log` | `832ca219175ba0ec659225877202f6fcfa920edade514f2062fc7f944f888c9c` | `67053fe3c0aea2006d10819ded4b35458058c8f36458e98b767157fcc254a73f` |

**Amendment 1 (2026-10-20, committed alone as `5ce528d`):** the epoch estimate is 8 (C1) / 3 (C2, C3) when no epoch improves,
matching tabicl's patience rule and the CPU development run; optimizer steps are now **counted** (attempted, taken, AMP-skipped) and the
tie rule uses the taken count; a restored config (`weights_changed: false`) is never selected as fine-tuned; the mask does not stop the
pinned archive (which also holds `test.csv`) from being downloaded and cached, only from being extracted or read; and a look marker
outside `outputs/` keeps a later scoring labelled "reproduction" after `outputs/` is deleted.

**Redaction placeholder.** The 2026-10-19 redaction of the two 2026-10-10 exec logs used `<runner>/`; the fleet's shared design names
`<local-run-dir>/`, which new redactions use. The two committed logs are left as recorded (their before/after hashes are above); the
unredacted path remains in the repository's earlier history, which is not rewritten.

### CPU development records (not clean-runtime evidence; fresh test masked)

| Date | Code | What ran | Result |
|---|---|---|---|
| 2026-10-19 | uncommitted working tree on `fix/tabicl-tutorial-review-2026-10-17` (after `9fa714a`) | stages data → validate → condition (`cpu_grid=True`, exploratory) → report → fresh_test (masked, then unmasked) → predict → export → reload on CPU, `DIMER_MASK_FRESH_TEST=1` except for the one unmasked `fresh_test` call | pretrained validation log loss 0.2634, accuracy 0.903 (LightGBM 0.3412 / 0.865, random forest 0.3234 / 0.880). C1 (1e-5): 8 epochs, 48 steps, every epoch 0.2712–0.2762 against tabicl's baseline 0.2701, `weights_changed: false`, last-epoch delta 0.415. C2 (5e-5): 3 epochs, 18 steps, 0.2864 / 0.3340 / 0.3027, delta 0.0 kept / 1.171 last. C3 (1e-6): 3 epochs, 18 steps, 0.2702 / 0.2701 / 0.2700, delta 0.0 kept / 0.031 last. Selection: pretrained kept. CPU grid hash `11cb0310…b19d` (a CPU hash; not comparable with T4). The unmasked `fresh_test` call wrote `not scored (exploratory)` with `settings_differing: [cpu_grid, device]` and read nothing. Reload PASS. 0 "y_prob do not sum to one" warnings. |
| 2026-10-20 | uncommitted working tree after `5ce528d` (round-2 code: counted steps, restored-config rule, look marker) | weights → data → validate → condition (`cpu_grid=True`) → fresh_test, all with `DIMER_MASK_FRESH_TEST=1` | same validation curves and selection as 2026-10-19; **counted** optimizer steps attempted/taken/AMP-skipped: C1 48/48/0, C2 18/18/0, C3 18/18/0 (AMP is off on CPU, so no skips are possible here; the T4 run is where fp16 skips can occur); grid hash `11cb0310…b19d` again (third identical CPU run); fresh test masked, not read. |
| 2026-10-20 | uncommitted working tree after `c718438` (amendment 2 code: MATH SDPA backend, FlashAttention-3 off, warnings surfaced) | weights → data → validate → condition (`cpu_grid=True`) → report → fresh_test, all with `DIMER_MASK_FRESH_TEST=1` | determinism record now includes `sdpa_backend: MATH`, `flash_attn3: false`; every number unchanged to 4 decimals: pretrained validation log loss 0.2634, accuracy 0.903; C1/C2/C3 curves and counted steps as on 2026-10-20 above; pretrained kept; 0 `DETERMINISM WARNING` lines. Grid hash `8db910b8…2a1a` (the backend is now a hash input, so it differs from the earlier CPU hash `11cb0310…` with identical numbers). Fresh test masked, not read. |

## SART7 deviation (2026-10-22)

- **Tag:** `sample-artifact-magic-v1` (asset `tabiclv2_classifier_artifact.zip`).
- **Mutable release:** the GitHub release is not immutable (`gh api repos/kurtvalcorza/tabicl-classifier-pipeline/releases/tags/sample-artifact-magic-v1 --jq .immutable` returns `false`), so SART7 (SHOULD) is not met.
- **Pinned digest:** the notebook pins the asset by `SAMPLE_ARTIFACT_SHA256 = "2dc82d55161382602fa7a5d1e217e3a33b6c6188ba6ddf4f7ff6b087bf67114c"` (`tools/tutorial_stages.py`, carried into the notebook); the GitHub asset digest is the same value.
- **Integrity:** integrity rests on that pinned SHA-256 check. `fetch_pinned_artifact` hashes the downloaded file with `check_trusted_digest` and refuses it before extraction if the digest differs, so a replaced asset is refused, not used.
- **Owner acceptance:** the owner accepted this deviation on 2026-10-22. The release is not re-tagged and the pin is not changed.

## Current status

**Current status (2026-10-23): Candidate.** The current E2E notebook blob is `515f8954` (`tutorials/tabiclv2_classifier_colab.ipynb` at `1344dca`, PR #33). Its default path is covered by reproduction 6 under *Fresh-test evaluation*, evidence in `execution-evidence/2026-10-23-1344dca/`. It ran by Colab CLI `colab exec -f` (scripted, not a browser Run all) on a fresh Colab Tesla T4: exit 0, 15/15 code cells, 0 error outputs, 425.8 s, input unchanged, verdict PASS. In that run the pre-registered fine-tuning grid ran and kept the pretrained model, so the fine-tuned-checkpoint export path was again not exercised; the fresh test reproduced the record (Δ = 0.0, `matches_record: True`, `differs: []`, the grid hash reported as informational); the export reloaded with identical labels; and Section 10 verified the pinned `sample-artifact-magic-v1` asset (`2dc82d55…`) before extraction. The optional-activity gate (the owner's 2026-10-22 ruling that the Section 11 Wine activity runs once on the current blob before promotion) is covered by reproduction 7, the `1344dca` optional-activity run (`RUN_ACTIVITY = True` run copy of this blob, Colab CLI, fresh T4, 440.0 s, 15/15 code cells, PASS, evidence in `execution-evidence/2026-10-23-1344dca-activity/`). The runs of the previous blob `f21920ab` (reproductions 4 and 5, `d4172f5` and `edbe94d`) are history and are not evidence for this blob (REL14). The notebook stays **Candidate** pending only these two for this blob:

- **REL12:** a BYOD run in which one representative input is accepted through the downstream stages and one incompatible input is refused;
- **REL2:** a browser run of the interactive journeys (BYOD upload dialog, optional-activity toggle, own-artifact ZIP); the scripted CLI runs count for the default path and the activity only.

Promotion is an integrator decision; the builder does not change the status.

**History, 2026-10-10 (blobs `051c81e9d8fd` and `cb92a9e84376`, both superseded; not evidence for the current blob, REL14) — one-pass hosted runs of both then-current blobs were recorded** (the 2026-10-10 rows of the clean-runtime table): `tabiclv2_classifier_colab.ipynb` (`051c81e9d8fd`) and `tabiclv2_classifier_artifact_inference_colab.ipynb` (`cb92a9e84376`) at `f59e233`, each by Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 (not a browser Run all), no restart, 0 error outputs. Notes for the reviewer: (a) the CUDA fine-tune ran (requested, 3 epochs / 300 s) and stopped early in about 2.7 s; its best holdout 0.9825 tied the pretrained model, so the pretrained model was exported — the fine-tuned-checkpoint export and reload path (331 MB `best.ckpt`) was **not** exercised on GPU (follow-up); (b) the Wine activity was not run (`RUN_ACTIVITY` off); (c) both carried revision labels read `fba80c6` (generation-time HEAD, not `f59e233`) — follow-up; (d) the companion's 8-row label match is a weak check, because all eight rows are `benign`; (e) the maintainer accepted the in-repo pinned sample bundle (no release; BSD-3, no weights) and keeping the bounded CUDA fine-tune on by default. Both notebooks stayed **Candidate**; the gates then remaining were reviewer confirmation of these runs, the fine-tuned-export path, the BYOD journeys and the optional activities.

Before 2026-10-10: no clean-runtime execution of the current standalone blobs had been recorded; the 2026-09-14 Kaggle row is history for an
earlier blob. Static validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code
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