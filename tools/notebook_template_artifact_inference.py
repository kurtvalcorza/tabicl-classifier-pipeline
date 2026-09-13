"""Companion template for tools/build_notebook.py — ARTIFACT-INFERENCE (NOTEBOOK_SPEC 1.1 §3.6, §18).

Generate with ``python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py``. The
notebook carries the same package module and the same pinned snapshot as the E2E notebook; it consumes a serving
bundle ZIP produced by a *separate* execution (upload, or an explicit path for non-interactive executors) and never
creates one.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_e2e_notebook_template", Path(__file__).with_name("notebook_template.py"))
assert _spec and _spec.loader
_e2e_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e2e_module)
BADGES, REPO, _E2E = _e2e_module.BADGES, _e2e_module.REPO, _e2e_module.TEMPLATE

TEMPLATE = {
    **{k: _E2E[k] for k in ("package", "repo_name", "pipeline_class", "weights_key", "modules", "entry_module", "model_load", "runtime_imports")},
    "stem": "tabiclv2_classifier_artifact_inference",
    "notebook_name": "tabiclv2_classifier_artifact_inference_colab.ipynb",
    "profile": "ARTIFACT-INFERENCE",
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
    "capability": "serving-state reconstruction from an externally produced DIMER-style TabICLv2 classifier bundle (`artifact.json` + checkpoint + training context in a ZIP) and class-probability inference on genuinely new rows under an explicit `argmax` rule",
    "intro": (
        "This notebook consumes a serving bundle ZIP produced **outside this execution** (for example by the E2E "
        "tutorial in a separate session): it extracts it only after every member passed the path, symlink and "
        "expanded-size checks, verifies the manifest's payload allowlist, sizes and SHA-256 digests, validates the "
        "artifact/runtime provenance, rebuilds the in-context classifier from the bundle alone (bundled checkpoint + "
        "training context + recorded inference settings), accepts genuinely new unlabelled rows, predicts class labels with "
        "class probabilities, and exports results. **No artifact is created here** and no gradient step runs.\n\n"
        "**Trust boundary.** Digest and manifest checks establish internal consistency, not sender authenticity, and "
        "the bundled `checkpoints/best.ckpt` is deserialised by `tabicl` (a Lightning/PyTorch checkpoint) — you are "
        "trusting the producer of the ZIP. The pinned base checkpoint of Section 3 is acquired and digest-verified "
        "independently so the manifest's `baseModelSha256` can be checked against a known-good value. Use only bundles "
        "from a trusted producer."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried package guarantees, resolve and digest-verify the immutable "
        "upstream checkpoint, supply an externally produced bundle and validate it before any model state is "
        "reconstructed, inspect its provenance and runtime compatibility, reconstruct the serving state from the bundle "
        "alone, validate new unlabelled rows into an input manifest, predict class labels with class probabilities under "
        "an explicit `argmax` rule, produce an evaluation report that is `not-measurable` because no labels exist, and export machine-readable predictions "
        "plus provenance."
    ),
    "exclusions": (
        "artifact creation, in-notebook support fitting, fine-tuning, regression, calibrated probabilities, or any "
        "quality claim: without labelled rows nothing is measured, and the exported probabilities are uncalibrated "
        "ensemble outputs with no shipped threshold."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.11+). The default path runs on CPU and uses CUDA automatically when available.",
        "- **Artifact:** an externally produced bundle ZIP (`artifact.json`, `checkpoints/best.ckpt`, `training_context.parquet`; the E2E tutorial writes `outputs/tabiclv2_classifier_artifact.zip`). Supply it through the upload dialog, or set `ARTIFACT_ZIP_PATH` to a file already present in the runtime for non-interactive execution. Nothing in this notebook manufactures it.",
        "- **Data:** one separate, unlabelled CSV with the bundle's feature columns. It is supplied by upload or by `NEW_DATA_PATH`; no sample is bundled, because scoring self-generated rows would not be external-artifact evidence. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Supply the external bundle and validate it before any model state is reconstructed\n\n"
                "Leave `ARTIFACT_ZIP_PATH` empty to upload the ZIP; set it to a file already in the runtime to skip the "
                "dialog (an executor places the file there). Optionally paste the producer's ZIP SHA-256 into "
                "`EXPECTED_ZIP_SHA256` to pin the whole archive. `safe_extract_zip` extracts member by member only after "
                "every member passed the path, symlink and expanded-size checks (AINF3; it never calls `extractall`); "
                "`verify_artifact_bundle` then checks the manifest's payload allowlist, sizes and SHA-256 digests, and "
                "`validate_artifact_runtime` checks the artifact format, the TabICL version and the producer/consumer torch "
                "families (AINF4). The manifest's base-model identity is also compared with the carried package's pinned "
                "identity and digest, so a bundle built on another base model is refused. A mismatch anywhere stops the "
                "notebook."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "import shutil\n\n"
                "ARTIFACT_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "EXPECTED_ZIP_SHA256 = ''  # @param {{type:\"string\"}}\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if ARTIFACT_ZIP_PATH:\n"
                "    zip_name, zip_payload = os.path.basename(ARTIFACT_ZIP_PATH), Path(ARTIFACT_ZIP_PATH).read_bytes()\n"
                "    artifact_source = f'path: {{ARTIFACT_ZIP_PATH}}'\n"
                "else:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError('Upload exactly one bundle ZIP')\n"
                "    zip_name, zip_payload = next(iter(uploaded.items()))\n"
                "    artifact_source = 'upload dialog'\n"
                "zip_path = Path('external-artifact') / Path(zip_name).name\n"
                "zip_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "zip_path.write_bytes(zip_payload)\n"
                "zip_sha256 = sha256_file(zip_path)\n"
                "if EXPECTED_ZIP_SHA256:\n"
                "    expected = EXPECTED_ZIP_SHA256.strip().lower()\n"
                "    if len(expected) != 64 or any(character not in '0123456789abcdef' for character in expected):\n"
                "        raise ValueError('EXPECTED_ZIP_SHA256 must be 64 hex chars')\n"
                "    if zip_sha256 != expected:\n"
                "        raise RuntimeError('Artifact ZIP SHA-256 mismatch')\n"
                "extract_dir = Path('external-artifact') / 'bundle'\n"
                "if extract_dir.exists():\n"
                "    shutil.rmtree(extract_dir)\n"
                "safe_extract_zip(zip_path, extract_dir)\n"
                "matches = list(extract_dir.rglob('artifact.json'))\n"
                "if len(matches) != 1:\n"
                "    raise ValueError('Expected exactly one artifact.json')\n"
                "bundle_root = matches[0].parent\n"
                "manifest = json.loads(matches[0].read_text(encoding='utf-8'))\n"
                "compatibility = validate_artifact_runtime(manifest, expected_tabicl_version=importlib.metadata.version('tabicl'), expected_torch_version=torch.__version__.split('+')[0])\n"
                "if (manifest.get('baseCheckpoint'), manifest.get('baseModelRevision'), manifest.get('baseModelSha256')) != (BASE_CHECKPOINT_NAME, MODEL_REVISION, BASE_MODEL_SHA256):\n"
                "    raise RuntimeError('Bundle was not produced on the pinned base checkpoint carried by this notebook')\n"
                "members = verify_artifact_bundle(bundle_root, manifest)\n"
                "FEATURE_COLUMNS = list(manifest['featureColumns'])\n"
                "TARGET_COLUMN = manifest['targetColumn']\n"
                "inference = manifest['inference']\n"
                "print({{'artifact_source': artifact_source, 'zip': zip_name, 'zip_sha256': zip_sha256, 'mode': manifest.get('mode'), 'selection': manifest.get('selectionBasis'), 'base': manifest.get('baseCheckpoint'), 'base_revision': str(manifest.get('baseModelRevision'))[:12]}})\n"
                "print(json.dumps(compatibility, indent=2, sort_keys=True))\n"
                "print({{'featureColumns': FEATURE_COLUMNS, 'targetColumn': TARGET_COLUMN, 'nEstimators': inference['nEstimators'], 'randomState': inference['randomState'], 'categoricalEncoders': sorted(inference.get('categoricalEncoders', {{}}))}})"
            ),
        },
        {
            "md": (
                "## 5. Reconstruct the in-context classifier from the bundle alone\n\n"
                "The bundle's `inference` block records how the producer ran the model: ensemble size, random seed, many-class "
                "support and the categorical encoders fitted on the training split; `classLabels` records the fitted class "
                "order, which the cell checks against the reconstructed `classes_` so probability columns cannot be permuted. The classifier is built through `create_classifier` on "
                "the **bundled** checkpoint (no auto-download, no network fallback — AINF5/AINF6) and conditioned on the "
                "bundled training context (`fit` registers the rows; nothing is trained). The pinned base checkpoint "
                "verified in Section 3 (`pipe`) is not used for inference; it exists so the manifest's base-model digest "
                "could be checked against a known-good value. This is serving-state reconstruction, not training."
            ),
            "code": (
                "context = pd.read_parquet(members['training_context'])\n"
                "serving = TabICLClassificationPipeline(create_classifier(model_path=members['checkpoint'], allow_auto_download=False, n_estimators=inference['nEstimators'], random_state=inference['randomState'], device=pipe.device), model_path=members['checkpoint'], n_estimators=inference['nEstimators'], random_state=inference['randomState'], device=pipe.device, source='artifact')\n"
                "serving.fit(context[FEATURE_COLUMNS], context[TARGET_COLUMN])\n"
                "CLASSES = list(serving.classes_)\n"
                "if manifest.get('classLabels') is not None and [str(label) for label in CLASSES] != list(manifest['classLabels']):\n"
                "    raise RuntimeError('Reconstructed class labels disagree with the validated manifest.')\n"
                "print({{'context_rows': len(context), 'features': len(FEATURE_COLUMNS), 'classes_': CLASSES, 'device': serving.device, 'source': serving.source, 'checkpoint': str(members['checkpoint'])}})"
            ),
        },
        {
            "md": (
                "## 6. Supply new unlabelled rows → validate → input manifest\n\n"
                "Leave `NEW_DATA_PATH` empty to upload one CSV, or set it to a file already in the runtime. The CSV must "
                "contain the bundle's feature columns (order does not matter; extra columns are preserved in the output "
                "and not passed to the model); duplicate header names, missing features, or pre-existing `prediction` / "
                "`probability_*` columns stop the run. `validate_inputs(..., target_column=None, feature_columns=...)` is the package's "
                "public validation stage for inference tables: it applies exactly the checks `read_inference_csv` applies "
                "and returns an **input manifest** naming the schema, the row count, the extra columns and the "
                "missing-value columns; it is written to `outputs/{stem}_input_manifest.json`. To show what rejection "
                "looks like, the cell also validates a probe with one feature column removed and records the package's "
                "own error message as a finding. The producer's categorical encoders are then applied; unseen values map "
                "to the fitted 'unknown' code and are counted."
            ),
            "code": (
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n"
                "if NEW_DATA_PATH:\n"
                "    input_name, payload = os.path.basename(NEW_DATA_PATH), Path(NEW_DATA_PATH).read_bytes()\n"
                "else:\n"
                "    new_upload = files.upload()\n"
                "    if len(new_upload) != 1:\n"
                "        raise ValueError('Upload exactly one CSV')\n"
                "    input_name, payload = next(iter(new_upload.items()))\n"
                "print({{'ceilings': {{'MIN_TRAIN_ROWS': MIN_TRAIN_ROWS, 'MAX_TRAIN_ROWS': MAX_TRAIN_ROWS, 'MAX_FEATURES': MAX_FEATURES, 'MIN_CLASSES': MIN_CLASSES}}, 'decision_rule': DECISION_RULE, 'required_features': FEATURE_COLUMNS, 'class_labels': CLASSES}})\n"
                "rows = read_inference_csv(payload, FEATURE_COLUMNS)\n"
                "input_manifest = validate_inputs(rows, None, feature_columns=FEATURE_COLUMNS, names=[input_name])\n"
                "# Demonstrate rejection on a probe that breaks the fitted schema; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(rows.drop(columns=[FEATURE_COLUMNS[0]]), None, feature_columns=FEATURE_COLUMNS)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'missing-column-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "X_new, unseen_new = apply_categorical_encoder(rows[FEATURE_COLUMNS], inference.get('categoricalEncoders', {{}}))\n"
                "if unseen_new:\n"
                "    print('unseen categorical values mapped to the unknown code:', unseen_new)"
            ),
        },
        {
            "md": (
                "## 7. Predict, report what cannot be measured, and export\n\n"
                "`predict_proba` returns one column per fitted class **in the bundle's stored class order** and `prediction` "
                "is the `argmax` label — the package ships no probability threshold and the probabilities are uncalibrated "
                "ensemble outputs; for cost-sensitive decisions threshold the relevant `probability_<class>` column on your "
                "own labelled data rather than using the argmax. `evaluation_report` is the "
                "package's public evaluation stage and is produced even here: with no labelled rows its verdict is "
                "`not-measurable` and it states what labelled data would make the task measurable; it is written to "
                "`outputs/{stem}_evaluation_report.json`. The prediction CSV keeps every input column plus `prediction` and one "
                "`probability_<class>` column per class in class order, "
                "and the result JSON records the externally supplied bundle identity (ZIP digest, manifest), the "
                "compatibility block, the input manifest, the notebook's source, the pinned model identity, revision and "
                "licence, and the runtime identity; it contains no credentials."
            ),
            "code": (
                "out = rows.copy()\n"
                "proba = serving.predict_proba(X_new)\n"
                "out['prediction'] = serving.predict(X_new)\n"
                "for index, class_label in enumerate(CLASSES):\n"
                "    out[f'probability_{{class_label}}'] = proba[:, index]\n"
                "out.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "report = evaluation_report(None, n_holdout=0, class_labels=CLASSES, target_column=TARGET_COLUMN, sample_kind='BYOD')\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "payload = {{\n"
                "    'predictions': out.to_dict(orient='records'),\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'artifact': {{'source': artifact_source, 'zip': zip_name, 'zip_sha256': zip_sha256, 'manifest': manifest, 'compatibility': compatibility, 'context_rows': len(context)}},\n"
                "    'inference': {{'nEstimators': inference['nEstimators'], 'randomState': inference['randomState'], 'decisionRule': DECISION_RULE + ' over uncalibrated class probabilities', 'threshold': None, 'classLabels': [str(label) for label in CLASSES], 'unseenCategoricalValues': unseen_new}},\n"
                "    'input': {{'filename': input_name, 'rows': len(out), 'features': FEATURE_COLUMNS}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'tabicl': importlib.metadata.version('tabicl'), 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__, 'device': serving.device}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(out.head())\n"
                "print(json.dumps(report, indent=2))\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A successful run proves that the supplied archive passed the package's path, symlink and expanded-size checks, "
        "that its declared payload matches the manifest's allowlist, sizes and digests, that the bundle names the pinned "
        "base checkpoint carried by this notebook, that the runtime is compatible with the producer's, that an in-context "
        "classifier was rebuilt from the bundle alone with its class order intact, and that schema-compatible new rows were "
        "scored with class probabilities under an explicit `argmax` decision rule — without the repository being reachable. It does **not** authenticate the producer or establish "
        "predictive quality, robustness, calibration, fairness, or production fitness; the evaluation report says "
        "`not-measurable` because no labels exist here, and the exported probabilities must not be thresholded without the "
        "caller's own calibration data. "
        "Never bypass a failed archive, manifest, digest, base-model or schema check; obtain a correct trusted bundle.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint, validate and reconstruct an external bundle, validate the "
        "supplied inference table, execute the public prediction path and emit the shown machine-readable outputs in the "
        "tested runtime. It does **not** establish benchmark superiority, deployment calibration, safety for "
        "high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** score rows with a deliberately unseen categorical value and inspect the unknown-code "
        "count; compare class probabilities across bundles exported with `mode` pretrained versus fine-tuned; hand a labelled "
        "copy of the same rows to `classification_metrics` in the E2E tutorial to obtain a `sample-sanity` report.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        f"- E2E companion (produces the bundle): https://github.com/kurtvalcorza/{REPO}/blob/main/tutorials/tabiclv2_classifier_colab.ipynb\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/soda-inria/tabicl\n"
        "- TabICLv2 paper: https://arxiv.org/abs/2602.11139"
    ),
}
