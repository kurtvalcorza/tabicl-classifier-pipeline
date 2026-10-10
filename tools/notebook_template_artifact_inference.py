"""Companion template for tools/build_notebook.py /3 — ARTIFACT-INFERENCE (NOTEBOOK_SPEC 2.2 §19, §25.13).

Generate with ``python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py``. The notebook
carries the same package and pinned checkpoint manifest as the E2E notebook, its own stage runner
(``tools/tutorial_stages_artifact_inference.py``) and the pinned sample bundle of ``examples/sample-bundle/`` (manifest,
support table, eight unlabelled rows; its checkpoint is the pinned base checkpoint staged in Section 3). A user bundle
ZIP (``ARTIFACT_ZIP_PATH`` or an upload) is checked against ``EXPECTED_ZIP_SHA256``. The notebook never creates a bundle.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_e2e_notebook_template", Path(__file__).with_name("notebook_template.py"))
assert _spec and _spec.loader
_e2e_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e2e_module)
BADGES, REPO, ENVIRONMENT, RUNTIME_PREREQ = _e2e_module.BADGES, _e2e_module.REPO, _e2e_module.ENVIRONMENT, _e2e_module.RUNTIME_PREREQ
SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "examples" / "sample-bundle" / "SAMPLE_BUNDLE.json").read_text(encoding="utf-8"))

TEMPLATE = {
    **ENVIRONMENT,
    "stem": "tabiclv2_classifier_artifact_inference",
    "notebook_name": "tabiclv2_classifier_artifact_inference_colab.ipynb",
    "profile": "ARTIFACT-INFERENCE",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages_artifact_inference.py",
    "carried_extra": {
        "sample-bundle/artifact.json": "examples/sample-bundle/artifact.json",
        "sample-bundle/new_rows.csv": "examples/sample-bundle/new_rows.csv",
        "sample-bundle/SAMPLE_BUNDLE.json": "examples/sample-bundle/SAMPLE_BUNDLE.json",
    },
    "carried_binary": {"sample-bundle/training_context.parquet": "examples/sample-bundle/training_context.parquet"},
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime (a T4 GPU is recommended) builds an isolated Python environment from the carried hash-locked requirements without touching the notebook kernel's own packages, then runs each stage in its own process: it stages and digest-verifies the pinned TabICLv2 checkpoint; assembles the carried **sample bundle** — the bundle the E2E notebook exports on its default path, pinned in the repository at `examples/sample-bundle/` (its `artifact.json` SHA-256 `" + SAMPLE["artifact_sha256"] + "` is checked) — whose checkpoint, because it declares `mode: pretrained`, must be and is the verified base checkpoint; validates the bundle (allowlist, sizes, digests, runtime compatibility, base identity) before any deserialisation; rebuilds the in-context classifier from the bundle alone; validates the eight carried unlabelled sample rows, which the bundle never saw, into an input manifest; predicts with class probabilities, reports what cannot be measured, and exports outputs. No upload dialog, repository clone, DIMER worker or service, credential, configuration edit or runtime restart is required (NOTEBOOK_SPEC 2.2 §5, §19). No hosted run of this revision has been recorded yet."
    ),
    "byod": (
        "Your own bundle is the `ARTIFACT_ZIP_PATH` / `UPLOAD_ARTIFACT` branch in Section 4: paste the ZIP digest the E2E notebook printed into `EXPECTED_ZIP_SHA256`, and any other ZIP is refused before it is extracted. Your own unlabelled rows are the `NEW_DATA_PATH` / `UPLOAD_NEW_DATA` branch in Section 6 (a CSV with the bundle's feature columns; extra columns such as an identifier are kept in the output and not passed to the model). Paths work in Colab, Kaggle and Jupyter; the upload dialogs exist only in Colab. Uploads stay inside this runtime; do not upload confidential or restricted data unless you are authorised to process it here."
    ),
    "title": "TabICLv2 Classifier — DIMER artifact inference tutorial (standalone)",
    "badges": [
        badge
        if badge[0] != "Open In Colab"
        else (
            badge[0],
            badge[1],
            f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/tabiclv2_classifier_artifact_inference_colab.ipynb",
        )
        for badge in BADGES
    ],
    "capability": "serving-state reconstruction from an externally produced DIMER-style TabICLv2 classifier bundle (`artifact.json` + checkpoint + training context) and class-probability inference on genuinely new rows under an explicit `argmax` rule",
    "intro": (
        "This notebook consumes a serving bundle produced **outside this execution** — by default the pinned sample bundle, "
        "which the E2E tutorial's default path exports; or your own ZIP from a separate E2E session. It extracts a ZIP only "
        "after every member passed the path, symlink and expanded-size checks, verifies the manifest's payload allowlist, "
        "sizes and SHA-256 digests, validates the artifact/runtime provenance and the base-model identity, rebuilds the "
        "in-context classifier from the bundle alone (bundled checkpoint + training context + recorded inference settings), "
        "accepts genuinely new unlabelled rows, predicts class labels with class probabilities, and exports results. **No "
        "artifact is created here** and no gradient step runs.\n\n"
        "**Trust boundary.** The bundled `checkpoints/best.ckpt` is deserialised by `tabicl` (a PyTorch/Lightning checkpoint), "
        "so loading a bundle means trusting its producer. Two mechanisms narrow that trust. For a bundle that declares `mode: "
        "pretrained`, the checkpoint's SHA-256 must equal the pinned base checkpoint's, verified independently in Section 3 — "
        "anything else is refused. For a fine-tuned bundle, the only binding is a trusted ZIP digest: with "
        "`EXPECTED_ZIP_SHA256` set, a ZIP with any other SHA-256 is refused before extraction. Digests inside `artifact.json` "
        "establish internal consistency, not sender authenticity. Use only bundles from a trusted producer."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** what a TabICLv2 serving bundle contains and why the support rows travel with the checkpoint (Sections 4, 5).\n"
        "2. **Verify** a bundle against a trusted digest, and **distinguish** what binds a pretrained checkpoint from what binds a fine-tuned one (Section 4).\n"
        "3. **Diagnose** a refused bundle or input table from its message (Sections 4, 6).\n"
        "4. **Apply** the rebuilt classifier to new unlabelled rows and read class-ordered, uncalibrated probabilities under an explicit `argmax` rule (Section 7).\n"
        "5. **Explain** why the evaluation report says `not-measurable` here (Section 7).\n"
        "6. **Predict**, run and **explain** the effect of the ensemble size on the same rows in an optional activity (Section 8)."
    ),
    "exclusions": (
        "artifact creation, in-notebook support fitting, fine-tuning, regression, calibrated probabilities, or any "
        "quality claim: without labelled rows nothing is measured, and the exported probabilities are uncalibrated "
        "ensemble outputs with no shipped threshold."
    ),
    "prerequisites": [
        RUNTIME_PREREQ,
        "- **Knowledge:** basic pandas and how to read a printed Python dictionary. The E2E notebook explains conditioning, fine-tuning and the metrics; this notebook's glossary repeats the terms it uses.",
        "- **Bundle:** by default, the carried sample bundle (`examples/sample-bundle/`: `artifact.json`, `training_context.parquet`; produced by `tools/build_sample_bundle.py` with the E2E notebook's own split, encoders and manifest code, no model run, so its `metrics` are `null`) plus the pinned base checkpoint as its `checkpoints/best.ckpt`. Optionally your own ZIP exported by the E2E notebook (`outputs/tabiclv2_classifier_artifact.zip`), by `ARTIFACT_ZIP_PATH` or upload, with the digest that notebook printed. Nothing here manufactures a bundle.",
        "- **Data:** by default, the eight carried unlabelled sample rows (`new_rows.csv`: the first eight holdout rows of the E2E split, which the bundle's support rows do not contain), with their original `row_id`. Optionally one unlabelled CSV with the bundle's feature columns, by `NEW_DATA_PATH` or upload. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who have run, or read, the E2E TabICLv2 tutorial and want to see how a "
                "packaged in-context classifier is reused safely by someone else: checking what was received, rebuilding it, and "
                "scoring new rows. You need to be able to run notebook cells and read short Python; the glossary explains every "
                "term.\n\n"
                "**Running it.** In Colab choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The default "
                "path needs no edit, no upload, no account, no token and no runtime restart: it uses the pinned sample bundle and "
                "sample rows carried in Section 2. Section 2 builds an isolated environment, which takes the longest.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process and stops the "
                "notebook with the stage's own error message if it fails. Stages hand results to each other through files: the "
                "validated bundle directory, the validated rows and JSON records.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–8) are the workflow. *Infrastructure cells* (Sections 1–3) are "
                "collapsed and titled **Infrastructure**; you may run them without studying their implementation.\n\n"
                "**Form controls.** `ARTIFACT_ZIP_PATH`, `UPLOAD_ARTIFACT` and `EXPECTED_ZIP_SHA256` (Section 4); `NEW_DATA_PATH` "
                "and `UPLOAD_NEW_DATA` (Section 6); `RUN_ACTIVITY` and `ACTIVITY_N_ESTIMATORS` (Section 8). Leave them at their "
                "defaults for the first run.\n\n"
                "**Section tags.** **[Concept]**, **[Evaluation practice]**, **[Engineering]** as in the E2E notebook.\n\n"
                "**Predict, then check.** Before Sections 4 and 7 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Verify** | a bundle (sample, or your ZIP with a trusted digest) | digest checks, `safe_extract_zip`, `validate_artifact_runtime`, base-identity binding, `verify_artifact_bundle` | accepted or refused, with provenance printed |\n"
                "| **Rebuild** | the bundled checkpoint, support table and settings | `create_classifier` + `fit` on the bundled context (no gradient step) | a classifier with the producer's class order |\n"
                "| **Validate rows** | unlabelled rows with the bundle's feature columns | `read_inference_csv`, `validate_inputs(target_column=None)`, numeric-type check | an input manifest; refusals name the column and rule |\n"
                "| **Predict** | the validated rows | TabICLv2 reading the bundle's support rows | one probability per class (stored class order), the `argmax` label, a `not-measurable` report |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1–3 | [Engineering] | runtime, carried code, isolated environment, verified checkpoint | versions, digest |\n"
                "| 4. Verify the bundle | [Engineering] | trusted digest, extraction, provenance, base binding | the trust record |\n"
                "| 5. Rebuild the classifier | [Concept] | conditioning on the bundled support rows | classes, device |\n"
                "| 6. Validate the new rows | [Evaluation practice] | input manifest, refusal probe, numeric check | the manifest |\n"
                "| 7. Predict and export | [Concept] | probabilities, labels, `not-measurable` report, provenance | the outputs |\n"
                "| 8. Optional activity | [Concept] | change `n_estimators` (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | every refusal and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | what was and was not shown | your conclusion |"
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Serving bundle** | `artifact.json` + `checkpoints/best.ckpt` + `training_context.parquet`. |\n"
                "| **Support rows / context** | The labelled rows TabICLv2 reads at prediction time; they are part of the model, so the bundle carries them. |\n"
                "| **`mode: pretrained` / `fine-tuned`** | Whether the bundle's checkpoint is the published base checkpoint or a fine-tuned derivative. |\n"
                "| **Trusted digest** | A SHA-256 obtained from the producer through a channel you trust (the E2E export prints it). |\n"
                "| **Internal consistency** | The manifest's recorded sizes and digests match its files; a forger who rewrites both still passes. |\n"
                "| **Code-capable checkpoint** | A file whose loading can execute code; load only verified or trusted files. |\n"
                "| **Safe extraction** | Extracting a ZIP member by member after path, symlink and size checks, never with `extractall`. |\n"
                "| **`n_estimators`** | How many permuted passes are averaged into one prediction. |\n"
                "| **`argmax` rule / uncalibrated** | Predict the most probable class; the probabilities are not guaranteed to match observed frequencies. |\n"
                "| **`not-measurable`** | The evaluation verdict when no labels exist. |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n\n"
                "</details>"
            ),
        ],
    },
    "cells": [
        {
            "md": (
                "## 4. Verify the bundle before any model state is reconstructed · [Engineering]\n\n"
                "The `artifact` stage takes the bundle from one of three places: the carried **pinned sample** (default), "
                "`ARTIFACT_ZIP_PATH` (a ZIP already in the runtime), or, in Colab, the upload dialog (`UPLOAD_ARTIFACT`). For the "
                "sample it checks the carried `artifact.json` against the SHA-256 recorded in `SAMPLE_BUNDLE.json` and places the "
                "verified base checkpoint of Section 3 as `checkpoints/best.ckpt`. For your ZIP it first checks "
                "`EXPECTED_ZIP_SHA256` — the digest the E2E notebook printed — before extracting anything; a mismatch stops here "
                "naming both digests, and without a digest your bundle still runs, with a printed warning.\n\n"
                "Then, before any deserialisation: `safe_extract_zip` extracts member by member after the path, symlink and "
                "expanded-size checks (it never calls `extractall`); `validate_artifact_runtime` checks the artifact format, the "
                "TabICL version and the producer/consumer torch families; the manifest must declare the pinned base checkpoint, "
                "and a `mode: pretrained` bundle's checkpoint digest must **equal** the pinned base digest (the check covers the "
                "checkpoint actually loaded, not only the manifest's declared base); `verify_artifact_bundle` checks the payload "
                "allowlist, sizes and SHA-256 digests. The stage prints the trust record, the provenance, and the support rows' "
                "count and class counts.\n\n"
                "**Predict before running:** the sample bundle's support rows are the E2E notebook's support partition. How many "
                "rows and which classes do you expect?"
            ),
            "code": (
                "ARTIFACT_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "UPLOAD_ARTIFACT = False  # @param {{type:\"boolean\"}}\n"
                "EXPECTED_ZIP_SHA256 = ''  # @param {{type:\"string\"}}\n\n"
                "def upload_one(what, field):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'The upload dialog exists only in Google Colab: set {{field}} to {{what}} in this runtime.') from None\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise RuntimeError(f'The upload was cancelled or empty: no file was received. Run this cell again and choose {{what}}, or set {{field}}.')\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError(f'Upload exactly one file ({{what}}); got {{sorted(uploaded)}}.')\n"
                "    upload_name, payload = next(iter(uploaded.items()))\n"
                "    path = ROOT / 'inputs' / Path(upload_name).name\n"
                "    path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    path.write_bytes(payload)\n"
                "    return str(path)\n\n"
                "artifact_source, zip_path = 'sample', ''\n"
                "if ARTIFACT_ZIP_PATH:\n"
                "    artifact_source, zip_path = 'path', ARTIFACT_ZIP_PATH\n"
                "elif UPLOAD_ARTIFACT:\n"
                "    artifact_source, zip_path = 'upload', upload_one('the bundle ZIP', 'ARTIFACT_ZIP_PATH')\n"
                "run_stage('artifact', source=artifact_source, zip_path=zip_path, expected_zip_sha256=EXPECTED_ZIP_SHA256)"
            ),
        },
        {
            "md": (
                "**What to notice:** `source: 'sample'`, `trusted_digest: 'verified …'`, `mode: 'pretrained'` and `checkpoint_binding: "
                "identical to the pinned base checkpoint`; the producer runtime block, which for the sample says it was an offline "
                "build with no model run; 30 feature columns, classes `benign` and `malignant`, 341 support rows with their class "
                "counts.\n\n"
                "**Checkpoint:** someone replaces `checkpoints/best.ckpt` in a bundle with a different file, rewrites "
                "`digests.checkpointSha256` and `sizes` in `artifact.json`, and leaves `mode: pretrained`. Which check stops it — "
                "and what if they also change `mode` to `fine-tuned`?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "`verify_artifact_bundle` would not: the manifest was rewritten to match the new file. The pretrained binding does: a "
                "`pretrained` checkpoint must have the pinned base digest, verified independently in Section 3, so the swapped file "
                "is refused before `tabicl` deserialises it. If the forger also switches `mode` to `fine-tuned`, that binding no "
                "longer applies — a fine-tuned checkpoint has no public reference digest. Then only `EXPECTED_ZIP_SHA256` stops "
                "it: any change to the ZIP changes its digest. Without a trusted digest a fine-tuned bundle is exactly as "
                "trustworthy as whoever handed it to you.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 5. Rebuild the classifier from the bundle alone · [Concept]\n\n"
                "The bundle's `inference` block records how the producer ran the model: ensemble size, random seed, many-class "
                "support and the categorical encoders fitted on the support split; `classLabels` records the fitted class order, "
                "which the stage checks against the rebuilt `classes_` so probability columns cannot be permuted. The classifier is "
                "built through `create_classifier` on the **bundled** checkpoint (no auto-download, no network fallback) and "
                "conditioned on the bundled training context (`fit` registers the rows; nothing is trained). This is serving-state "
                "reconstruction, not training."
            ),
            "code": "run_stage('reconstruct')",
        },
        {
            "md": "**What to notice:** 341 context rows, 30 features, `classes_` `['benign', 'malignant']`, the device, and the checkpoint binding.",
        },
        {
            "md": (
                "## 6. Supply new unlabelled rows → validate → input manifest · [Evaluation practice]\n\n"
                "By default the `rows` stage uses the eight carried sample rows, which fit only the pinned sample bundle. For your "
                "own rows set `NEW_DATA_PATH` (any runtime) or tick `UPLOAD_NEW_DATA` (Colab). With your own bundle and neither set, "
                "the cell opens the upload dialog in Colab and, elsewhere, stops with a message naming `NEW_DATA_PATH`. The CSV must "
                "contain the bundle's feature columns (order does not matter; extra columns such as `row_id` are preserved in the "
                "output and not passed to the model); duplicate header names, missing features, or pre-existing `prediction` / "
                "`probability_*` columns stop the run. `validate_inputs(..., target_column=None, feature_columns=...)` applies exactly "
                "the checks `read_inference_csv` applies and returns an **input manifest** (schema, row count, extra columns, "
                "missing-value columns), written to `outputs/{stem}_input_manifest.json`; a probe with one feature removed records "
                "the package's own refusal as a finding. Then every column the producer treated as **numeric** must hold numbers: "
                "a value such as `abc` is refused naming the column and the value, instead of reaching the model."
            ),
            "code": (
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n"
                "UPLOAD_NEW_DATA = False  # @param {{type:\"boolean\"}}\n\n"
                "rows_source, rows_path = 'sample', ''\n"
                "if NEW_DATA_PATH:\n"
                "    rows_source, rows_path = 'path', NEW_DATA_PATH\n"
                "elif UPLOAD_NEW_DATA or artifact_source != 'sample':\n"
                "    rows_source, rows_path = 'upload', upload_one('one unlabelled CSV for your bundle', 'NEW_DATA_PATH')\n"
                "run_stage('rows', source=rows_source, path=rows_path)"
            ),
        },
        {
            "md": (
                "**What to notice:** the ceilings and the 30 required features; the input manifest with 8 rows, `extra_columns: "
                "['row_id']`, no missing values, and the `missing-column-probe` finding."
            ),
        },
        {
            "md": (
                "## 7. Predict, report what cannot be measured, and export · [Concept]\n\n"
                "The `predict` stage rebuilds the classifier and scores the rows. `predict_proba` returns one column per fitted class "
                "**in the bundle's stored class order** and `prediction` is the `argmax` label — the package ships no probability "
                "threshold and the probabilities are uncalibrated ensemble outputs; for cost-sensitive decisions threshold the "
                "relevant `probability_<class>` column on your own labelled data. `evaluation_report` is the package's public "
                "evaluation stage and is produced even here: with no labelled rows its verdict is `not-measurable`; its "
                "`sample_kind` is `sample` for the pinned sample bundle and rows and `BYOD` otherwise. It is written to "
                "`outputs/{stem}_evaluation_report.json`. The prediction CSV keeps every input column plus `prediction` and one "
                "`probability_<class>` column per class in class order, and the result JSON records the bundle identity and its "
                "trust record, the compatibility block, the input manifest, the notebook's source, the pinned model identity, "
                "revision and licence, and the runtime identity as a provenance record; it contains no credentials.\n\n"
                "**Predict before running:** these are the first eight holdout rows of the E2E notebook. If you ran it with a "
                "pretrained selection, will the probabilities here equal its Section 8 predictions?"
            ),
            "code": "run_stage('predict')",
        },
        {
            "md": (
                "**What to notice:** eight rows with `row_id`, `prediction` and two probability columns summing to 1; "
                "`verdict: not-measurable`, `sample_kind: sample`.\n\n"
                "**Checkpoint:** why is the verdict `not-measurable`, and what would make it measurable?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The rows arrive without labels, as in deployment, so there is nothing to compare predictions with. A labelled "
                "holdout whose classes all appear in the support rows, scored with `classification_metrics` against "
                "`majority_class_baseline` (what the E2E notebook does), would make it measurable. As for the prediction: with the "
                "same checkpoint (pretrained), support rows, settings and seed, the probabilities match the E2E notebook's up to "
                "floating-point differences between devices.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Optional activity: how much does the ensemble size move the probabilities? · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** with `ACTIVITY_N_ESTIMATORS = 32` instead of the "
                "bundle's 8, will any `prediction` change, and by how much will the probabilities move? **Change:** tick "
                "`RUN_ACTIVITY`. **Run** this cell. **Observe** `labels_changed` and the probability changes. **Explain** what you "
                "see. The activity scores the same rows, writes only to `outputs/activity/`, and stops if any canonical output "
                "changed."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_N_ESTIMATORS = 32  # @param {{type:\"integer\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', n_estimators=ACTIVITY_N_ESTIMATORS)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** `labels_changed`, the largest and mean probability change, and "
                "`canonical_outputs_unchanged: True`. No hosted run of the activity has been recorded, so compare with your own "
                "prediction.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "More permuted passes average out the effect of any one feature or row order, so the probabilities move slightly "
                "toward a steadier value; confident rows keep their label. A label that flips was close to 0.5 — the rows where an "
                "uncalibrated probability deserves the least trust.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1–3 failures (platform, disk, `uv`, Hub download, digest) | as in the E2E notebook | See its Troubleshooting; never remove a pin, a hash or a manifest digest. |\n"
        "| `The run directory … has no carried files, or the isolated environment is gone` | Section 1 run with `NEW_RUN_DIRECTORY` ticked | Run Sections 1–3 again, or *Run all*. |\n"
        "| `Trusted digest mismatch for …zip` | the ZIP is not the one the digest was issued for | Do not proceed; obtain the bundle and digest from the producer again. |\n"
        "| `EXPECTED_ZIP_SHA256 must be 64 hexadecimal characters` | a truncated digest | Paste the full digest the E2E notebook printed. |\n"
        "| `No EXPECTED_ZIP_SHA256 was supplied` (a warning) | your bundle without a digest | It runs, but only internal consistency is checked. |\n"
        "| `ARTIFACT_ZIP_PATH … is not a file` | a wrong path | Point it at `outputs/tabiclv2_classifier_artifact.zip` from the E2E run. |\n"
        "| `Unsafe ZIP member`, `Ambiguous backslash ZIP member`, `ZIP member escapes destination`, `Artifact exceeds … expanded bytes` | a malformed or hostile ZIP | Do not use it; re-export the bundle. |\n"
        "| `expected exactly one artifact.json` | a ZIP of the wrong folder | Zip the bundle directory itself. |\n"
        "| `Unsupported artifactFormat` / `TabICL version does not match` / `torch … incompatible` | a bundle from another format or runtime | Re-export with this repository's E2E notebook. |\n"
        "| `Bundle was not produced on the pinned base checkpoint` | another base model | Use a bundle built on the pinned checkpoint. |\n"
        "| `The bundle declares mode 'pretrained', so its checkpoint must be the pinned base checkpoint` | a swapped checkpoint | Refuse the bundle. |\n"
        "| `Unexpected or missing artifact files`, `Checkpoint size/digest mismatch`, `Training-context size/digest mismatch` | the files do not match the manifest | Re-export; never edit the manifest. |\n"
        "| `Reconstructed class labels disagree with the validated manifest` | a corrupted context or manifest | Re-export the bundle. |\n"
        "| `The pinned sample rows match only the pinned sample bundle` / `upload dialog exists only in Google Colab: set NEW_DATA_PATH` | your bundle with no rows | Set `NEW_DATA_PATH` (or tick `UPLOAD_NEW_DATA` in Colab). |\n"
        "| `Inference CSV missing features` / `duplicate column names` / `already contains prediction/probability output columns` | the rows do not match the bundle | Supply exactly the bundle's feature columns, unscored. |\n"
        "| `numeric feature … has N non-numeric value(s)` | text in a numeric column | Fix the values; leave missing cells empty. |\n\n"
        "## Interpretation and limits\n\n"
        "A successful run proves that the bundle passed the package's path, symlink and expanded-size checks, that its payload "
        "matches the manifest's allowlist, sizes and digests, that it names the pinned base checkpoint — and, for a `pretrained` "
        "bundle, that its checkpoint is byte for byte that base — that the runtime is compatible with the producer's, that an "
        "in-context classifier was rebuilt from the bundle alone with its class order intact, and that schema- and type-"
        "compatible new rows were scored with class probabilities under an explicit `argmax` decision rule — without the "
        "repository being reachable. It does **not** authenticate the producer beyond the channel a trusted digest came "
        "through, or establish predictive quality, robustness, calibration, fairness, or production fitness; the evaluation "
        "report says `not-measurable` because no labels exist here, and the exported probabilities must not be thresholded "
        "without the caller's own calibration data. Never bypass a failed archive, manifest, digest, base-model or schema "
        "check; obtain a correct trusted bundle. The pinned sample bundle was built offline by `tools/build_sample_bundle.py` "
        "(no model run; `metrics: null`); its support table is cross-checked against a real E2E export by "
        "`matches_pinned_sample_context`.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint, validate and reconstruct an external bundle, validate the "
        "supplied inference table, execute the public prediction path and emit the shown machine-readable outputs in the "
        "tested runtime. It does **not** establish benchmark superiority, deployment calibration, safety for "
        "high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three sentences: what the digest checks and the base binding established about the bundle you used; what the "
        "rebuild and the eight predictions show; and what you would need before trusting these probabilities for a decision.\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "The pinned sample bundle's manifest matched its recorded SHA-256, and because it declares `mode: pretrained` its "
        "checkpoint had to be — and was — the base checkpoint verified in Section 3, so every file I loaded is pinned. The "
        "classifier was rebuilt from the bundle's own support rows and settings with its class order intact and scored eight "
        "unseen rows with uncalibrated probabilities; with no labels the report is correctly `not-measurable`. Before using "
        "these probabilities for a decision I would need a labelled, domain-representative test set and calibration data to "
        "choose a threshold.\n\n"
        "</details>\n\n"
        "**Next experiments:** run the E2E notebook, then set `ARTIFACT_ZIP_PATH` to its "
        "`outputs/tabiclv2_classifier_artifact.zip`, paste its printed digest into `EXPECTED_ZIP_SHA256`, and score its "
        "`outputs/tabiclv2_classifier_new_rows.csv` through `NEW_DATA_PATH`; paste a digest with one character changed and read "
        "the refusal; compare probabilities from a pretrained and a fine-tuned bundle of the same split.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        f"- Pinned sample bundle: https://github.com/kurtvalcorza/{REPO}/tree/main/examples/sample-bundle\n"
        f"- E2E companion (produces bundles): https://github.com/kurtvalcorza/{REPO}/blob/main/tutorials/tabiclv2_classifier_colab.ipynb\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/soda-inria/tabicl\n"
        "- TabICLv2 paper: https://arxiv.org/abs/2602.11139"
    ),
}
