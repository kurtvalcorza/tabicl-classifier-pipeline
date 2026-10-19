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

- **Evaluation of record:** none yet. The first hosted Tesla T4 run of the pre-registered settings will be recorded here (commit,
  blob, date, `grid_hash`, the primary endpoint with its interval and verdict) and `RECORDED_EVALUATION` in `tools/tutorial_stages.py`
  set to it; every later identical run is a **reproduction** and is listed under it.
- **Before that run:** two **masked** hosted T4 runs (`DIMER_MASK_FRESH_TEST=1`) must print identical grid hashes, validation numbers
  and selections (determinism check). If they differ, no run of record is made.
- **Fresh-test looks so far:** none. No CPU development look was taken: the fixer's CPU checks ran with the fresh test masked.

### CPU development records (not clean-runtime evidence; fresh test masked)

| Date | Code | What ran | Result |
|---|---|---|---|
| 2026-10-19 | uncommitted working tree on `fix/tabicl-tutorial-review-2026-10-17` (after `9fa714a`) | stages data → validate → condition (`cpu_grid=True`, exploratory) → report → fresh_test (masked, then unmasked) → predict → export → reload on CPU, `DIMER_MASK_FRESH_TEST=1` except for the one unmasked `fresh_test` call | pretrained validation log loss 0.2634, accuracy 0.903 (LightGBM 0.3412 / 0.865, random forest 0.3234 / 0.880). C1 (1e-5): 8 epochs, 48 steps, every epoch 0.2712–0.2762 against tabicl's baseline 0.2701, `weights_changed: false`, last-epoch delta 0.415. C2 (5e-5): 3 epochs, 18 steps, 0.2864 / 0.3340 / 0.3027, delta 0.0 kept / 1.171 last. C3 (1e-6): 3 epochs, 18 steps, 0.2702 / 0.2701 / 0.2700, delta 0.0 kept / 0.031 last. Selection: pretrained kept. CPU grid hash `11cb0310…b19d` (a CPU hash; not comparable with T4). The unmasked `fresh_test` call wrote `not scored (exploratory)` with `settings_differing: [cpu_grid, device]` and read nothing. Reload PASS. 0 "y_prob do not sum to one" warnings. |

## Current status

**2026-10-10 — one-pass hosted runs of both current blobs are recorded** (the 2026-10-10 rows of the clean-runtime table): `tabiclv2_classifier_colab.ipynb` (`051c81e9d8fd`) and `tabiclv2_classifier_artifact_inference_colab.ipynb` (`cb92a9e84376`) at `f59e233`, each by Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 (not a browser Run all), no restart, 0 error outputs. Notes for the reviewer: (a) the CUDA fine-tune ran (requested, 3 epochs / 300 s) and stopped early in about 2.7 s; its best holdout 0.9825 tied the pretrained model, so the pretrained model was exported — the fine-tuned-checkpoint export and reload path (331 MB `best.ckpt`) was **not** exercised on GPU (follow-up); (b) the Wine activity was not run (`RUN_ACTIVITY` off); (c) both carried revision labels read `fba80c6` (generation-time HEAD, not `f59e233`) — follow-up; (d) the companion's 8-row label match is a weak check, because all eight rows are `benign`; (e) the maintainer accepted the in-repo pinned sample bundle (no release; BSD-3, no weights) and keeping the bounded CUDA fine-tune on by default. Both notebooks stay **Candidate**; remaining gates: reviewer confirmation of these runs, the fine-tuned-export path, the BYOD journeys and the optional activities.

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