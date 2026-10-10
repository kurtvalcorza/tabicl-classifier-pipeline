"""Stage runner for the standalone TabICLv2 classifier ARTIFACT-INFERENCE tutorial (NOTEBOOK_SPEC 2.2 §25.13 pattern).

The companion notebook carries this file verbatim as ``tutorial_stages.py`` beside the carried package under ``src/``
and the pinned sample bundle under ``sample-bundle/``, and runs every stage with the interpreter of an isolated,
hash-locked environment. Nothing is installed into the notebook kernel and no bundle is created here.

The default path uses the pinned sample bundle: ``examples/sample-bundle/`` holds its ``artifact.json``, its
``training_context.parquet`` and eight unlabelled rows it never saw. Its manifest declares ``mode: pretrained``, so its
checkpoint is, byte for byte, the pinned base checkpoint the ``weights`` stage downloads and digest-verifies; the
110 MB file is therefore not duplicated in the repository but bound by ``digests.checkpointSha256 ==
BASE_MODEL_SHA256``, which this runner enforces for every ``pretrained`` bundle (TICA-m2). A user bundle (ZIP by
``ARTIFACT_ZIP_PATH`` or upload) is checked against ``EXPECTED_ZIP_SHA256`` — the digest the E2E notebook prints when it
exports — before it is extracted.

Stages: weights → artifact → reconstruct → rows → predict, plus the optional ``activity``. ``check_bundle_identity``,
``check_numeric_features`` and the ``rows`` stage import no model library, so CI exercises them directly.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

STEM = "tabiclv2_classifier_artifact_inference"
PACKAGE = "tabicl_classifier_pipeline"
SAMPLE_DIR = "sample-bundle"
SAMPLE_RECORD = "sample-bundle/SAMPLE_BUNDLE.json"
SAMPLE_ROWS = "sample-bundle/new_rows.csv"
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class Run:
    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path


def package(root: Path):
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(PACKAGE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


def check_trusted_digest(path: Path, expected: str, field: str) -> str:
    expected = expected.strip().lower()
    if not SHA256_HEX.match(expected):
        raise ValueError(f"{field} must be 64 hexadecimal characters (the digest the E2E notebook printed when it exported the bundle), got {expected!r}.")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"Trusted digest mismatch for {path.name}: {field} is {expected}, the supplied file's SHA-256 is {observed}. This is not the bundle the digest was issued for; nothing was extracted. Obtain the bundle and its digest from the producer again.")
    return observed


def check_bundle_identity(P, manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest must declare the pinned base checkpoint; a ``pretrained`` bundle's checkpoint must BE that base
    checkpoint (``digests.checkpointSha256 == BASE_MODEL_SHA256``), so a swapped, code-capable checkpoint behind a
    re-hashed manifest is refused (TICA-m2). A ``fine-tuned`` checkpoint is a derivative bound only by the bundle's own
    digests and, when supplied, the trusted ZIP digest."""
    declared = (manifest.get("baseCheckpoint"), manifest.get("baseModelRevision"), manifest.get("baseModelSha256"))
    if declared != (P.BASE_CHECKPOINT_NAME, P.MODEL_REVISION, P.BASE_MODEL_SHA256):
        raise RuntimeError(f"Bundle was not produced on the pinned base checkpoint carried by this notebook: declares {declared}, expected {(P.BASE_CHECKPOINT_NAME, P.MODEL_REVISION, P.BASE_MODEL_SHA256)}.")
    mode = manifest.get("mode")
    checkpoint_sha = (manifest.get("digests") or {}).get("checkpointSha256")
    if mode == "pretrained":
        if checkpoint_sha != P.BASE_MODEL_SHA256:
            raise RuntimeError(f"The bundle declares mode 'pretrained', so its checkpoint must be the pinned base checkpoint with SHA-256 {P.BASE_MODEL_SHA256}; its manifest records {checkpoint_sha}. Refusing to load a different checkpoint.")
        return {"mode": mode, "checkpoint_binding": "identical to the pinned base checkpoint (SHA-256 checked)"}
    if mode == "fine-tuned":
        return {"mode": mode, "checkpoint_binding": "fine-tuned derivative of the pinned base; bound by the manifest digest and, when supplied, EXPECTED_ZIP_SHA256"}
    raise RuntimeError(f"Unsupported bundle mode {mode!r}: expected 'pretrained' or 'fine-tuned'.")


def check_numeric_features(name: str, rows, features: list[str], encoders: dict[str, Any]):
    """Columns the producer treated as numeric (no fitted encoder) must hold numbers; refuse naming the column and the
    first bad values (TICA-m3), instead of handing strings to the model."""
    import pandas as pd

    out = rows.copy()
    for column in features:
        if column in encoders:
            continue
        original = out[column]
        converted = pd.to_numeric(original, errors="coerce")
        bad = original.notna() & converted.isna()
        if bad.any():
            examples = original[bad].astype(str).unique().tolist()[:5]
            raise ValueError(f"{name}: numeric feature {column!r} has {int(bad.sum())} non-numeric value(s), e.g. {examples}. Fix them (leave missing cells empty).")
        out[column] = converted
    return out


def stage_weights(run: Run) -> None:
    P = package(run.root)
    snapshot = run.weights / P.MODEL_KEY
    snapshot.mkdir(parents=True, exist_ok=True)
    carried = run.root / "weights" / P.MODEL_KEY / P.MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (P.MODEL_ID, P.MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity carried by the package; regenerate the notebook")
    shutil.copyfile(carried, snapshot / P.MANIFEST_NAME)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]})
    fetched = P.stage_missing_files(snapshot, allow_download=True)
    print({"weights_dir": str(snapshot), "fetched": fetched})
    verified = P.verify_snapshot(snapshot)
    print({"verified_files": len(verified["files"]), "revision": verified["revision"], "sha256": [f["sha256"] for f in verified["files"]]})
    run.write_state("weights.json", {"snapshot": str(snapshot), "checkpoint": str(snapshot / P.BASE_CHECKPOINT_NAME)})


def clear_outputs(run: Run) -> list[str]:
    removed = []
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def assemble_sample(run: Run, P, bundle: Path) -> dict[str, Any]:
    record = json.loads((run.root / SAMPLE_RECORD).read_text(encoding="utf-8"))
    manifest_path = run.root / SAMPLE_DIR / "artifact.json"
    digest = check_trusted_digest(manifest_path, record["artifact_sha256"], "the pinned sample digest")
    (bundle / "checkpoints").mkdir(parents=True)
    shutil.copyfile(manifest_path, bundle / "artifact.json")
    shutil.copyfile(run.root / SAMPLE_DIR / "training_context.parquet", bundle / "training_context.parquet")
    base = Path(run.read_state("weights.json", "artifact")["checkpoint"])
    target = bundle / "checkpoints" / "best.ckpt"
    try:
        target.hardlink_to(base)
    except OSError:
        shutil.copyfile(base, target)
    return {"trusted_digest": "verified (artifact.json against the pinned sample record)", "artifact_json_sha256": digest, "producer": record["producer"], "data": record["data"]}


def stage_artifact(run: Run) -> None:
    import pandas as pd
    import torch

    P = package(run.root)
    opts = run.options
    source = opts.get("source", "sample")
    removed = clear_outputs(run)
    if removed:
        print({"removed_previous_outputs": removed})
    for name in ("artifact.json", "rows.json", "rows.parquet"):
        (run.state / name).unlink(missing_ok=True)
    bundle = run.state / "bundle"
    shutil.rmtree(bundle, ignore_errors=True)
    if source == "sample":
        trust = assemble_sample(run, P, bundle)
        root, zip_name, zip_sha = bundle, None, None
    elif source in ("path", "upload"):
        zip_path = Path(opts.get("zip_path") or "")
        if not str(zip_path) or not zip_path.is_file():
            raise FileNotFoundError(f"ARTIFACT_ZIP_PATH {str(zip_path)!r} is not a file in this runtime: point it at the bundle ZIP the E2E notebook exported.")
        expected = str(opts.get("expected_zip_sha256") or "")
        if expected:
            zip_sha = check_trusted_digest(zip_path, expected, "EXPECTED_ZIP_SHA256")
            trust = {"trusted_digest": "verified (EXPECTED_ZIP_SHA256)", "zip_sha256": zip_sha}
        else:
            zip_sha = sha256_file(zip_path)
            trust = {"trusted_digest": "not supplied", "zip_sha256": zip_sha}
            print("No EXPECTED_ZIP_SHA256 was supplied: the checks below establish internal consistency only, not that this is the bundle you were sent. Ask the producer for the digest the E2E notebook printed.")
        P.safe_extract_zip(zip_path, bundle)
        matches = list(bundle.rglob("artifact.json"))
        if len(matches) != 1:
            raise ValueError(f"{zip_path.name}: expected exactly one artifact.json in the bundle, found {len(matches)}.")
        root, zip_name = matches[0].parent, zip_path.name
    else:
        raise ValueError(f"unknown artifact source {source!r}")
    manifest = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    compatibility = P.validate_artifact_runtime(manifest, expected_tabicl_version=importlib.metadata.version("tabicl"), expected_torch_version=torch.__version__.split("+")[0])
    binding = check_bundle_identity(P, manifest)
    members = P.verify_artifact_bundle(root, manifest)
    context = pd.read_parquet(members["training_context"])
    target = manifest["targetColumn"]
    record = {"source": source, "zip": zip_name, **trust, **binding, "root": str(root), "manifest": manifest, "contextRows": len(context), "contextClassCounts": {str(k): int(v) for k, v in context[target].value_counts().sort_index().items()}, "compatibility": compatibility}
    run.write_state("artifact.json", record)
    print({k: record[k] for k in ("source", "zip", "trusted_digest", "mode", "checkpoint_binding")})
    print({"selection": manifest.get("selectionBasis"), "base": manifest.get("baseCheckpoint"), "base_revision": str(manifest.get("baseModelRevision"))[:12], "producer_runtime": manifest.get("runtime")})
    print({"featureColumns": len(manifest["featureColumns"]), "targetColumn": target, "classLabels": manifest.get("classLabels"), "contextRows": len(context), "contextClassCounts": record["contextClassCounts"], "nEstimators": manifest["inference"]["nEstimators"], "categoricalEncoders": sorted(manifest["inference"].get("categoricalEncoders", {}))})


def serving_model(run: Run):
    import pandas as pd

    P = package(run.root)
    art = run.read_state("artifact.json", "reconstruct")
    manifest = art["manifest"]
    root = Path(art["root"])
    members = P.verify_artifact_bundle(root, manifest)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = manifest["inference"]
    serving = P.TabICLClassificationPipeline(P.create_classifier(model_path=members["checkpoint"], allow_auto_download=False, n_estimators=int(run.options.get("n_estimators", inference["nEstimators"])), random_state=inference["randomState"], device=device), model_path=members["checkpoint"], n_estimators=int(run.options.get("n_estimators", inference["nEstimators"])), random_state=inference["randomState"], device=device, source="artifact")
    context = pd.read_parquet(members["training_context"])
    serving.fit(context[manifest["featureColumns"]], context[manifest["targetColumn"]])
    classes = list(serving.classes_)
    if manifest.get("classLabels") is not None and [str(c) for c in classes] != list(manifest["classLabels"]):
        raise RuntimeError("Reconstructed class labels disagree with the validated manifest.")
    return P, art, manifest, serving, classes


def stage_reconstruct(run: Run) -> None:
    _P, art, manifest, serving, classes = serving_model(run)
    print({"context_rows": art["contextRows"], "features": len(manifest["featureColumns"]), "classes_": [str(c) for c in classes], "device": serving.device, "source": serving.source, "checkpoint_binding": art["checkpoint_binding"]})


def stage_rows(run: Run) -> None:
    P = package(run.root)
    art = run.read_state("artifact.json", "rows")
    manifest = art["manifest"]
    features = list(manifest["featureColumns"])
    encoders = manifest["inference"].get("categoricalEncoders", {})
    source = run.options.get("source", "sample")
    if source == "sample":
        if art["source"] != "sample":
            raise ValueError("The pinned sample rows match only the pinned sample bundle. With your own bundle, set NEW_DATA_PATH to your rows (or, in Colab, tick UPLOAD_NEW_DATA).")
        path = run.root / SAMPLE_ROWS
    else:
        path = Path(run.options.get("path") or "")
        if not str(path) or not path.is_file():
            raise FileNotFoundError(f"NEW_DATA_PATH {str(path)!r} does not exist in this runtime: upload the file or correct the path.")
    print({"ceilings": {"MIN_TRAIN_ROWS": P.MIN_TRAIN_ROWS, "MAX_TRAIN_ROWS": P.MAX_TRAIN_ROWS, "MAX_FEATURES": P.MAX_FEATURES, "MIN_CLASSES": P.MIN_CLASSES}, "decision_rule": P.DECISION_RULE, "required_features": len(features), "class_labels": manifest.get("classLabels")})
    rows = P.read_inference_csv(path.read_bytes(), features)
    manifest_in = P.validate_inputs(rows, None, feature_columns=features, names=[path.name])
    # Demonstrate rejection on a probe that breaks the fitted schema; the finding is recorded, not swallowed.
    try:
        P.validate_inputs(rows.drop(columns=[features[0]]), None, feature_columns=features)
    except ValueError as exc:
        manifest_in["findings"].append({"input": "missing-column-probe", "verdict": "rejected", "message": str(exc)})
    rows = check_numeric_features(path.name, rows, features, encoders)
    run.write_output(f"{STEM}_input_manifest.json", manifest_in)
    rows.to_parquet(run.state / "rows.parquet", index=False, engine="pyarrow")
    sample_kind = "sample" if source == "sample" and art["source"] == "sample" else "BYOD"
    run.write_state("rows.json", {"source": source, "name": path.name, "rows": len(rows), "extra_columns": manifest_in["inputs"][0]["extra_columns"], "sample_kind": sample_kind})
    print(json.dumps(manifest_in, indent=2))


def score_rows(run: Run):
    import pandas as pd

    P, art, manifest, serving, classes = serving_model(run)
    rows_state = run.read_state("rows.json", "predict")
    rows = pd.read_parquet(run.state / "rows.parquet", engine="pyarrow")
    X, unseen = P.apply_categorical_encoder(rows[manifest["featureColumns"]], manifest["inference"].get("categoricalEncoders", {}))
    out = rows.copy()
    proba = serving.predict_proba(X)
    out["prediction"] = serving.predict(X)
    for index, label in enumerate(classes):
        out[f"probability_{label}"] = proba[:, index]
    return P, art, manifest, serving, classes, rows_state, out, unseen


def stage_predict(run: Run) -> None:
    import importlib.metadata
    import platform

    import numpy
    import pandas
    import sklearn
    import torch

    P, art, manifest, serving, classes, rows_state, out, unseen = score_rows(run)
    out.to_csv(run.out / f"{STEM}_predictions.csv", index=False)
    report = P.evaluation_report(None, n_holdout=0, class_labels=classes, target_column=manifest["targetColumn"], sample_kind=rows_state["sample_kind"])
    run.write_output(f"{STEM}_evaluation_report.json", report)
    input_manifest = json.loads((run.out / f"{STEM}_input_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    payload = {
        "predictions": out.to_dict(orient="records"),
        "evaluation_report": report,
        "input_manifest": input_manifest,
        "artifact": {k: art.get(k) for k in ("source", "zip", "zip_sha256", "artifact_json_sha256", "trusted_digest", "mode", "checkpoint_binding", "contextRows", "compatibility")} | {"manifest": manifest},
        "inference": {"nEstimators": serving.n_estimators, "randomState": serving.random_state, "decisionRule": P.DECISION_RULE + " over uncalibrated class probabilities", "threshold": None, "classLabels": [str(c) for c in classes], "unseenCategoricalValues": unseen},
        "input": {"filename": rows_state["name"], "source": rows_state["source"], "rows": len(out), "extra_columns": rows_state["extra_columns"], "features": manifest["featureColumns"]},
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "tabicl": importlib.metadata.version("tabicl"), "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__, "device": serving.device},
    }
    run.write_output(f"{STEM}_result.json", payload)
    run.write_state("predict.json", {"n_estimators": serving.n_estimators})
    if unseen:
        print("unseen categorical values mapped to the unknown code:", unseen)
    print(out.round(4).to_string(index=False))
    print(json.dumps({k: report[k] for k in ("verdict", "sample_kind", "reason")}, indent=2))


def stage_activity(run: Run) -> None:
    """Optional activity: score the same rows with a different ensemble size; write to outputs/activity/ only."""
    import numpy as np
    import pandas as pd

    n = int(run.options.get("n_estimators", 32))
    if not 1 <= n <= 64:
        raise ValueError(f"ACTIVITY_N_ESTIMATORS must be from 1 to 64, got {n}")
    canonical = run.read_state("predict.json", "activity")["n_estimators"]
    before = {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()}
    base = pd.read_csv(run.out / f"{STEM}_predictions.csv")
    run.options = {"n_estimators": n}
    _P, _art, _m, _s, classes, _r, out, _u = score_rows(run)
    cols = [f"probability_{c}" for c in classes]
    diff = np.abs(base[cols].to_numpy() - out[cols].to_numpy())
    record = {"changed": {"n_estimators": [canonical, n]}, "rows": len(out), "labels_changed": int((base["prediction"].astype(str).to_numpy() != out["prediction"].astype(str).to_numpy()).sum()), "max_abs_probability_change": float(diff.max()), "mean_abs_probability_change": float(diff.mean())}
    run.write_output(f"activity/{STEM}_activity_n_estimators_{n}.json", record)
    if {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()} != before:
        raise RuntimeError("The activity changed a canonical output; it must write only to outputs/activity/.")
    print(rounded(record))
    print({"canonical_outputs_unchanged": True})


STAGES = {
    "weights": stage_weights,
    "artifact": stage_artifact,
    "reconstruct": stage_reconstruct,
    "rows": stage_rows,
    "predict": stage_predict,
    "activity": stage_activity,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        STAGES[args.stage](run)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
