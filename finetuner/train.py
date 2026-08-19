"""DIMER fine-tuner for TabICLv2 classification."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

TEMPLATE_NAME = "tabicl-classifier-finetuner"
BASE_MODEL = "tabicl-classifier-v2-20260212.ckpt"
TABICL_VERSION = "2.1.1"
DATASET_DIR = Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset"))
OUTPUT_DIR = Path(os.getenv("DIMER_OUTPUT_DIR", "/data/output"))
RESULT_PATH = Path(os.getenv("DIMER_RESULT_PATH", "/data/results/result.json"))
DONE_CALLBACK = os.getenv("DIMER_DONE_CALLBACK", "").strip()
CALLBACK_TIMEOUT_SECONDS = float(os.getenv("DIMER_CALLBACK_TIMEOUT_SECONDS", "10"))
MAX_ARCHIVE_UNCOMPRESSED_BYTES = int(os.getenv("DIMER_MAX_ARCHIVE_UNCOMPRESSED_BYTES", str(1 << 30)))
MAX_SINGLE_CSV_BYTES = int(os.getenv("DIMER_MAX_SINGLE_CSV_BYTES", str(512 << 20)))
MAX_FEATURES = 2_000


def log(message: str) -> None:
    print(f"[{TEMPLATE_NAME}] {message}", flush=True)


def _json_env(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def notify_done_callback() -> dict[str, Any]:
    if not DONE_CALLBACK:
        return {"attempted": False}
    parsed = urlparse(DONE_CALLBACK)
    if parsed.scheme not in {"http", "https"}:
        return {"attempted": False, "error": f"unsupported callback scheme {parsed.scheme!r}"}
    try:
        response = requests.post(DONE_CALLBACK, timeout=CALLBACK_TIMEOUT_SECONDS)
        return {"attempted": True, "ok": response.ok, "statusCode": response.status_code}
    except requests.RequestException as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def _normalize_member(name: str) -> str | None:
    if not name or name.endswith("/"):
        return None
    normalized = name.replace("\\", "/").lstrip("./")
    parts = Path(normalized).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe archive member: {name}")
    if len(parts) > 1 and parts[0].lower() in {"dataset", "datasets"}:
        normalized = str(Path(*parts[1:]))
    return normalized


def _zip_and_member(stem: str) -> tuple[zipfile.ZipFile | None, Any | None]:
    zips = sorted(DATASET_DIR.glob("*.zip"))
    if len(zips) > 1:
        raise ValueError(f"multiple dataset zip files found: {[p.name for p in zips]}")
    if not zips:
        matches = sorted(
            p for p in DATASET_DIR.rglob("*.csv") if p.stem.lower() == stem.lower()
        )
        if len(matches) > 1:
            raise ValueError(f"multiple {stem}.csv candidates found: {[str(p) for p in matches]}")
        if not matches:
            return None, None
        if matches[0].stat().st_size > MAX_SINGLE_CSV_BYTES:
            raise ValueError(f"{matches[0].name} exceeds the configured CSV size limit")
        return None, matches[0]

    zf = zipfile.ZipFile(zips[0])
    total = sum(int(info.file_size) for info in zf.infolist())
    if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        zf.close()
        raise ValueError("dataset archive exceeds the configured uncompressed-size limit")
    matches = []
    for info in zf.infolist():
        logical = _normalize_member(info.filename)
        if logical and Path(logical).suffix.lower() == ".csv" and Path(logical).stem.lower() == stem.lower():
            matches.append(info)
    if len(matches) > 1:
        names = [info.filename for info in matches]
        zf.close()
        raise ValueError(f"multiple {stem}.csv candidates found: {names}")
    if not matches:
        zf.close()
        return None, None
    if matches[0].file_size > MAX_SINGLE_CSV_BYTES:
        zf.close()
        raise ValueError(f"{matches[0].filename} exceeds the configured CSV size limit")
    return zf, matches[0]


def _read_csv(stem: str) -> pd.DataFrame | None:
    zf, source = _zip_and_member(stem)
    if source is None:
        return None
    try:
        if zf is not None:
            with zf.open(source) as handle:
                return pd.read_csv(handle)
        return pd.read_csv(source)
    finally:
        if zf is not None:
            zf.close()


def _stratified_holdout(frame: pd.DataFrame, target: str, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < fraction < 1:
        return frame.reset_index(drop=True), pd.DataFrame(columns=frame.columns)
    counts = frame[target].value_counts()
    if counts.size < 2 or int(counts.min()) < 2:
        raise ValueError("every class needs at least 2 usable rows for a stratified validation split")
    train, val = train_test_split(
        frame,
        test_size=fraction,
        random_state=seed,
        stratify=frame[target],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _stratified_cap(frame: pd.DataFrame, target: str, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.reset_index(drop=True)
    selected, _ = train_test_split(
        frame,
        train_size=cap,
        random_state=seed,
        stratify=frame[target],
    )
    return selected.reset_index(drop=True)


def _prepare_frames(pre: dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str, list[str]]:
    target = str(pre.get("target_column") or "target").strip()
    drop_columns = [c.strip() for c in str(pre.get("drop_columns") or "").split(",") if c.strip()]
    validation_split = float(pre.get("validation_split") if pre.get("validation_split") is not None else 0.2)
    max_train_rows = int(pre.get("max_train_rows") or 10_000)
    max_train_rows = max(300, min(max_train_rows, 50_000))

    train = _read_csv("train")
    if train is None:
        raise FileNotFoundError("no train.csv found")
    if target not in train.columns:
        raise KeyError(f"target column {target!r} not found")
    train = train.dropna(subset=[target]).drop(columns=[c for c in drop_columns if c in train.columns])
    feature_columns = [c for c in train.columns if c != target]
    if not feature_columns:
        raise ValueError("no feature columns remain")
    if len(feature_columns) > MAX_FEATURES:
        raise ValueError(f"{len(feature_columns)} features exceeds configured limit {MAX_FEATURES}")
    if train[target].nunique(dropna=True) < 2:
        raise ValueError("classification target must contain at least 2 classes")

    val = _read_csv("val")
    if val is not None:
        val = val.dropna(subset=[target]).drop(columns=[c for c in drop_columns if c in val.columns])
        if set(val.columns) != set(train.columns):
            raise ValueError("val.csv schema does not match train.csv after preprocessing")
        val = val[train.columns]
    else:
        train, val = _stratified_holdout(train, target, min(max(validation_split, 0.05), 0.4), seed)

    train = _stratified_cap(train, target, max_train_rows, seed)

    test = _read_csv("test")
    if test is not None:
        test = test.dropna(subset=[target]).drop(columns=[c for c in drop_columns if c in test.columns])
        if set(test.columns) != set(train.columns):
            raise ValueError("test.csv schema does not match train.csv after preprocessing")
        test = test[train.columns].reset_index(drop=True)

    return train.reset_index(drop=True), val.reset_index(drop=True), test, target, feature_columns


def _classification_metrics(model, frame: pd.DataFrame, target: str) -> dict[str, Any]:
    X = frame.drop(columns=[target])
    y = frame[target].to_numpy()
    pred = np.asarray(model.predict(X))
    proba = np.asarray(model.predict_proba(X))
    metrics: dict[str, Any] = {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(y, pred)),
        "logLoss": float(log_loss(y, proba, labels=model.classes_)),
    }
    try:
        if len(model.classes_) == 2:
            metrics["rocAuc"] = float(roc_auc_score(y, proba[:, 1]))
        else:
            metrics["rocAucOvr"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=model.classes_))
    except ValueError as exc:
        metrics["rocAucError"] = str(exc)
    return metrics


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_digest() -> dict[str, Any] | None:
    zips = sorted(DATASET_DIR.glob("*.zip"))
    if zips:
        return {"file": zips[0].name, "sha256": _sha256(zips[0])}
    csvs = sorted(DATASET_DIR.rglob("*.csv"))
    if not csvs:
        return None
    h = hashlib.sha256()
    for path in csvs:
        h.update(path.relative_to(DATASET_DIR).as_posix().encode("utf-8"))
        h.update(b"\0")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return {"files": [p.relative_to(DATASET_DIR).as_posix() for p in csvs], "sha256": h.hexdigest()}


def run() -> int:
    hp = _json_env("DIMER_HYPERPARAMETERS_JSON")
    pre = _json_env("DIMER_PREPROCESSING_ARGS_JSON")
    seed = int(hp.get("seed") or 0)
    train, val, test, target, feature_columns = _prepare_frames(pre, seed)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("TabICLv2 fine-tuning requires a CUDA GPU in this DIMER pipeline")

    from tabicl import FinetunedTabICLClassifier, TabICLClassifier

    epochs = int(hp.get("epochs") or 30)
    learning_rate = float(hp.get("learning_rate") or 1e-5)
    weight_decay = float(hp.get("weight_decay") if hp.get("weight_decay") is not None else 0.01)
    patience = int(hp.get("patience") or 8)
    time_limit = float(hp.get("time_limit_seconds") or 1800)
    eval_metric = str(hp.get("eval_metric") or "accuracy")
    n_ft = int(hp.get("n_estimators_finetune") or 2)
    n_val = int(hp.get("n_estimators_validation") or 2)
    n_inf = int(hp.get("n_estimators_inference") or 8)

    artifact_dir = OUTPUT_DIR / "tabicl_classifier"
    ckpt_dir = artifact_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = train.drop(columns=[target]), train[target]
    X_val, y_val = val.drop(columns=[target]), val[target]

    model = FinetunedTabICLClassifier(
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_estimators_finetune=n_ft,
        n_estimators_validation=n_val,
        n_estimators_inference=n_inf,
        early_stopping=True,
        patience=patience,
        time_limit=time_limit,
        eval_metric=eval_metric,
        checkpoint_version=BASE_MODEL,
        device="cuda",
        random_state=seed,
        verbose=True,
        support_many_classes=True,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, output_dir=str(ckpt_dir))

    best_ckpt = ckpt_dir / "best.ckpt"
    if not best_ckpt.exists():
        raise RuntimeError("TabICL fine-tuning completed without producing checkpoints/best.ckpt")

    val_metrics = _classification_metrics(model, val, target)
    test_metrics = _classification_metrics(model, test, target) if test is not None and len(test) else None

    # Build a portable inference artifact: fine-tuned checkpoint + the ICL context table.
    context_path = artifact_dir / "training_context.parquet"
    train.to_parquet(context_path, index=False)
    checkpoint_sha256 = _sha256(best_ckpt)
    training_context_sha256 = _sha256(context_path)
    # artifact.json is the complete inference contract: everything a serving
    # process needs to rebuild the exact model that produced the reported metrics.
    manifest = {
        "artifactFormat": "tabicl-dimer-classifier-v1",
        "checkpoint": "checkpoints/best.ckpt",
        "trainingContext": "training_context.parquet",
        "targetColumn": target,
        "featureColumns": feature_columns,
        "baseCheckpoint": BASE_MODEL,
        "tabiclVersion": TABICL_VERSION,
        "inference": {
            "class": "TabICLClassifier",
            "modelPath": "checkpoints/best.ckpt",
            "nEstimators": n_inf,
            "randomState": seed,
            "device": "cuda",
            "supportManyClasses": True,
            "allowAutoDownload": False,
            "procedure": (
                "TabICLClassifier(model_path, **inference); "
                "fit(context[featureColumns], context[targetColumn]); "
                "predict(X[featureColumns])"
            ),
        },
        "digests": {
            "checkpointSha256": checkpoint_sha256,
            "trainingContextSha256": training_context_sha256,
        },
    }
    (artifact_dir / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Prove the artifact is a self-contained inference contract: reconstruct the
    # model using ONLY artifact.json + the context table, then predict.
    served = json.loads((artifact_dir / "artifact.json").read_text(encoding="utf-8"))
    inference = served["inference"]
    context = pd.read_parquet(artifact_dir / served["trainingContext"])
    ctx_features = context[served["featureColumns"]]
    ctx_target = context[served["targetColumn"]]
    reloaded = TabICLClassifier(
        model_path=str(artifact_dir / inference["modelPath"]),
        allow_auto_download=inference["allowAutoDownload"],
        n_estimators=inference["nEstimators"],
        random_state=inference["randomState"],
        device=inference["device"],
        support_many_classes=inference["supportManyClasses"],
    )
    reloaded.fit(ctx_features, ctx_target)
    smoke_rows = min(8, len(val))
    if smoke_rows:
        _ = reloaded.predict(X_val[served["featureColumns"]].iloc[:smoke_rows])

    # Keep only best.ckpt in the served artifact; drop intermediate epoch checkpoints.
    pruned_bytes = 0
    for stale in sorted(ckpt_dir.glob("epoch*.ckpt")):
        pruned_bytes += stale.stat().st_size
        stale.unlink()
    if pruned_bytes:
        log(f"Pruned {pruned_bytes} bytes of intermediate epoch checkpoints")

    payload = {
        "successful": True,
        "message": f"TabICLv2 fine-tuning succeeded on {len(train)} rows; validation accuracy {val_metrics['accuracy']:.4f}.",
        "metrics": {
            "trainRows": int(len(train)),
            "val": val_metrics,
            "test": test_metrics,
            "numClasses": int(train[target].nunique()),
            "featureCount": len(feature_columns),
            "device": "cuda",
            "mode": "fine-tune",
        },
        "artifacts": {
            "modelDir": str(artifact_dir),
            "checkpoint": str(best_ckpt),
            "trainingContext": str(context_path),
        },
        "provenance": {
            "baseModel": BASE_MODEL,
            "tabiclVersion": TABICL_VERSION,
            "fineTunedCheckpointSha256": checkpoint_sha256,
            "trainingContextSha256": training_context_sha256,
            "artifactDigestSha256": hashlib.sha256(
                (checkpoint_sha256 + training_context_sha256).encode("utf-8")
            ).hexdigest(),
            "dataset": _dataset_digest(),
        },
        "metadata": {
            "template": TEMPLATE_NAME,
            "taskType": "tabular_classification",
            "targetColumn": target,
            "seed": seed,
            "epochs": epochs,
            "learningRate": learning_rate,
            "evalMetric": eval_metric,
        },
    }
    write_result(payload)
    log(f"Callback: {json.dumps(notify_done_callback(), sort_keys=True)}")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "successful": False,
            "message": f"TabICLv2 fine-tuning failed: {exc}",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "metadata": {"template": TEMPLATE_NAME, "taskType": "tabular_classification"},
        }
        try:
            write_result(payload)
            notify_done_callback()
        except Exception as write_exc:  # noqa: BLE001
            log(f"Failed to persist crash result: {write_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
