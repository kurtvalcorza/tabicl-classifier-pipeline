# ruff: noqa: E501  -- long docstrings, messages and single-line test fixtures are kept readable
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import shutil
import stat
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARTIFACT_FORMAT = "tabicl-dimer-classifier-v1"

# Fleet snapshot identity (DIMER Notebook Specification 1.1, ST3/MOD13). The pinned upstream checkpoint is
# unchanged; these are the fleet-standard names for the same repository, revision, license and snapshot key.
# The BASE_* spellings below stay as the package's published names and alias these constants.
MODEL_ID = "jingang/TabICL"
MODEL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
MODEL_LICENSE = "bsd-3-clause"
MODEL_KEY = "tabicl-classifier-v2"
MANIFEST_NAME = "dimer-base-manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY

BASE_MODEL_REPO = MODEL_ID
BASE_CHECKPOINT_NAME = "tabicl-classifier-v2-20260212.ckpt"
BASE_MODEL_REVISION = MODEL_REVISION
BASE_MODEL_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"

# Operational ceilings the tutorials enforce before any model execution (DAT22).
MIN_TRAIN_ROWS = 50  # labelled support rows required to condition the classifier
MIN_EVAL_ROWS = 2  # labelled rows required for a holdout / test partition
MAX_TRAIN_ROWS = 50_000  # support rows per conditioning call
MAX_FEATURES = 2_000  # feature columns per table
MAX_ARTIFACT_EXPANDED_BYTES = 1024 * 1024 * 1024  # 1 GiB expanded ZIP size for a serving artifact
DEFAULT_N_ESTIMATORS = 8
DEFAULT_RANDOM_STATE = 42
MIN_CLASSES = 2  # distinct target classes in the support rows
MIN_ROWS_PER_CLASS = 2  # rows per class for a stratified holdout split
DECISION_RULE = "argmax"  # `predict` is the argmax over `predict_proba`; no threshold is shipped
METRIC_IDS = ("accuracy", "balanced_accuracy", "f1_weighted", "log_loss", "roc_auc")  # the ids `classification_metrics` reports


def runtime_identity() -> dict[str, str]:
    import torch

    return {
        "pythonVersion": sys.version.split()[0],
        "tabiclVersion": importlib.metadata.version("tabicl"),
        "torchVersion": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def create_classifier(
    *,
    model_path: str | Path,
    n_estimators: int,
    random_state: int,
    device: str,
    allow_auto_download: bool = False,
    support_many_classes: bool = True,
):
    """Construct the supported TabICLv2 classification serving estimator."""
    from tabicl import TabICLClassifier

    return TabICLClassifier(
        model_path=str(model_path),
        allow_auto_download=allow_auto_download,
        n_estimators=n_estimators,
        random_state=random_state,
        device=device,
        support_many_classes=support_many_classes,
    )


def create_finetuned_classifier(**kwargs: Any):
    """Construct the supported TabICLv2 classification fine-tuning estimator."""
    from tabicl import FinetunedTabICLClassifier

    kwargs.setdefault("allow_auto_download", False)
    kwargs.setdefault("support_many_classes", True)
    return FinetunedTabICLClassifier(**kwargs)


def fine_tune_classifier(model: Any, X: Any, y: Any, **kwargs: Any) -> Any:
    """Run the upstream fine-tuning operation through the repository API."""
    return model.fit(X, y, **kwargs)


def condition_classifier(model: Any, X: Any, y: Any) -> Any:
    """Register the serving support context required by TabICL inference (no gradient step)."""
    model.fit(X, y)
    return model


def predict_labels(model: Any, X: Any) -> np.ndarray:
    """Return one-dimensional class labels (the argmax of the class probabilities)."""
    values = np.asarray(model.predict(X))
    if values.ndim != 1:
        values = values.reshape(-1)
    return values


def predict_probabilities(model: Any, X: Any) -> np.ndarray:
    """Return finite class probabilities, one column per entry of ``model.classes_`` in that order."""
    values = np.asarray(model.predict_proba(X), dtype=float)
    if values.ndim != 2 or values.shape[1] != len(model.classes_):
        raise RuntimeError("TabICL produced probabilities of an unexpected shape")
    if not np.isfinite(values).all():
        raise RuntimeError("TabICL produced non-finite class probabilities")
    return values


def align_probabilities(proba: Any, model_classes: Sequence[Any], classes: Sequence[Any]) -> np.ndarray:
    """Reorder probability columns from ``model_classes`` order into ``classes`` order (strict: every class in
    ``classes`` must be a column, so blending two models can never permute labels silently)."""
    proba = np.asarray(proba, dtype=float)
    lookup = {label: index for index, label in enumerate(model_classes)}
    missing = [label for label in classes if label not in lookup]
    if missing:
        raise RuntimeError(f"model never saw classes: {missing}")
    return proba[:, [lookup[label] for label in classes]]


def read_single_input(*, env_var: str, label: str) -> tuple[str, bytes]:
    """Read one user-supplied file from an explicit path or Colab upload dialog."""
    explicit = os.environ.get(env_var, "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
        return path.name, path.read_bytes()

    try:
        from google.colab import files  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{label} requires either a Colab upload or environment variable {env_var}"
        ) from exc

    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError(f"Upload exactly one {label}")
    name, payload = next(iter(uploaded.items()))
    return str(name), bytes(payload)


def download_output(path: str | Path) -> Path:
    """Download in Colab; otherwise retain the file at its explicit Jupyter path."""
    resolved = Path(path).resolve()
    try:
        from google.colab import files  # type: ignore
    except ModuleNotFoundError:
        print(f"Output retained at {resolved}")
        return resolved
    files.download(str(resolved))
    return resolved


def _base_version(value: str) -> str:
    return value.split("+")[0]


def validate_artifact_runtime(
    manifest: dict[str, Any],
    *,
    expected_artifact_format: str = ARTIFACT_FORMAT,
    expected_tabicl_version: str = "2.1.1",
    expected_torch_version: str = "2.14.0",
) -> dict[str, Any]:
    """Validate and return artifact/runtime provenance before model deserialization."""
    if manifest.get("artifactFormat") != expected_artifact_format:
        raise ValueError(f"Unsupported artifactFormat: {manifest.get('artifactFormat')}")
    if manifest.get("tabiclVersion") != expected_tabicl_version:
        raise ValueError("Artifact TabICL version does not match the supported runtime")

    observed = runtime_identity()
    if _base_version(observed["torchVersion"]) != expected_torch_version:
        raise RuntimeError(
            f"Runtime torch {observed['torchVersion']} is incompatible with the "
            f"release-verified torch {expected_torch_version}"
        )

    producer = manifest.get("runtime")
    compatibility = {
        "artifactFormat": manifest.get("artifactFormat"),
        "baseCheckpoint": manifest.get("baseCheckpoint"),
        "baseModelRevision": manifest.get("baseModelRevision"),
        "baseModelSha256": manifest.get("baseModelSha256"),
        "tabiclVersion": manifest.get("tabiclVersion"),
        "producerRuntime": producer,
        "consumerRuntime": observed,
        "policy": "TabICL exact; consumer torch base version 2.14.0; device may differ",
    }

    if producer:
        producer_torch = str(producer.get("torchVersion", ""))
        if producer_torch and _base_version(producer_torch) != expected_torch_version:
            raise RuntimeError(
                f"Artifact was produced with torch {producer_torch}; expected release family "
                f"{expected_torch_version}"
            )
    else:
        compatibility["warning"] = (
            "Legacy artifact has no runtime block; TabICL/version and digest checks apply, "
            "but producer runtime compatibility cannot be proven."
        )
    return compatibility


# ---------------------------------------------------------------------------
# Fleet snapshot scheme (NOTEBOOK_SPEC 1.1 ST3/ST4, MOD13): manifest-driven verification and staging.
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local pinned snapshot against its manifest; raise naming the first mismatch.

    The manifest is the parity anchor the standalone tutorials carry inline (ST3). The package's own
    ``BASE_MODEL_SHA256`` is not replaced by it: the manifest entry for ``BASE_CHECKPOINT_NAME`` must equal
    that constant, so the two can never diverge silently.
    """
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    entries = manifest.get("files", [])
    declared = {entry["path"]: entry["sha256"] for entry in entries}
    if declared.get(BASE_CHECKPOINT_NAME) != BASE_MODEL_SHA256:
        raise ValueError(
            f"manifest {BASE_CHECKPOINT_NAME} sha256 {declared.get(BASE_CHECKPOINT_NAME)!r} "
            f"!= BASE_MODEL_SHA256 {BASE_MODEL_SHA256!r}"
        )
    for entry in entries:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=MODEL_ID, filename=relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a clone commits the manifest but
    git-ignores the checkpoint). Returns the relative paths fetched; ``verify_snapshot`` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


class TabICLClassificationPipeline:
    """Serving wrapper: the digest-verified pinned checkpoint behind the upstream ``TabICLClassifier``.

    ``from_pretrained`` stages and verifies the snapshot and constructs the estimator through
    ``create_classifier`` (no auto-download); ``fit`` registers the support context (in-context learning,
    no gradient training), ``predict_proba`` returns finite class probabilities in ``classes_`` order and
    ``predict`` the ``argmax`` label (``DECISION_RULE``; no threshold is shipped).
    """

    def __init__(
        self,
        estimator: Any,
        *,
        model_path: Path,
        n_estimators: int,
        random_state: int,
        device: str,
        source: str = "local-snapshot",
    ) -> None:
        self.estimator = estimator
        self.model_path = Path(model_path)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self.source = source
        self.is_fitted = False

    @classmethod
    def from_pretrained(
        cls,
        weights_dir: str | Path | None = None,
        *,
        allow_download: bool = False,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        random_state: int = DEFAULT_RANDOM_STATE,
        device: str | None = None,
    ) -> TabICLClassificationPipeline:
        """Stage what is missing (at ``MODEL_REVISION``), re-hash every manifest entry, then build the
        estimator on the verified checkpoint. ``tabicl`` deserialises the checkpoint on the first ``fit``."""
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = root / BASE_CHECKPOINT_NAME
        estimator = create_classifier(
            model_path=model_path,
            n_estimators=n_estimators,
            random_state=random_state,
            device=device,
            allow_auto_download=False,
        )
        return cls(
            estimator,
            model_path=model_path,
            n_estimators=n_estimators,
            random_state=random_state,
            device=device,
        )

    @property
    def classes_(self) -> list[Any]:
        if not self.is_fitted:
            raise RuntimeError("Pipeline is not conditioned; call fit(X, y) with the support rows first")
        return list(self.estimator.classes_)

    def fit(self, X: Any, y: Any) -> TabICLClassificationPipeline:
        condition_classifier(self.estimator, X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline is not conditioned; call fit(X, y) with the support rows first")
        return predict_probabilities(self.estimator, X)

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline is not conditioned; call fit(X, y) with the support rows first")
        return predict_labels(self.estimator, X)


# ---------------------------------------------------------------------------
# Table preparation, encoding, metrics (extracted from the tutorials so both notebooks share one code path).
# ---------------------------------------------------------------------------


def _check_classification_table(
    frame: pd.DataFrame,
    target_column: str,
    *,
    min_rows: int,
    max_rows: int = MAX_TRAIN_ROWS,
    max_features: int = MAX_FEATURES,
    min_classes: int = MIN_CLASSES,
) -> tuple[pd.DataFrame, int]:
    """The checks `prepare_classification_table` applies; returns the cleaned table and the dropped-row count."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas.DataFrame")
    if frame.columns.duplicated().any():
        dupes = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise ValueError(f"table contains duplicate column names: {dupes}")
    if target_column not in frame.columns:
        raise KeyError(f"missing target {target_column!r}")
    labelled = frame[target_column].notna()
    dropped = int((~labelled).sum())
    out = frame.loc[labelled].copy().reset_index(drop=True)
    if len(out) < min_rows:
        raise ValueError(f"need at least {min_rows} labelled rows, got {len(out)}")
    features = [column for column in out.columns if column != target_column]
    if not features:
        raise ValueError("No feature columns")
    if len(features) > max_features or len(out) > max_rows:
        raise ValueError(
            f"Operational row/feature ceiling exceeded: rows={len(out)} (MAX_TRAIN_ROWS={max_rows}), "
            f"features={len(features)} (MAX_FEATURES={max_features})"
        )
    if out[target_column].nunique() < min_classes:
        raise ValueError(f"Need at least {min_classes} target classes, got {out[target_column].nunique()}")
    return out, dropped


def prepare_classification_table(
    frame: pd.DataFrame, target_column: str, *, min_rows: int = MIN_TRAIN_ROWS, min_classes: int = MIN_CLASSES
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose target is missing (the count is returned so the notebook can report it — DAT23),
    enforce the row/feature ceilings and the class-count floor."""
    return _check_classification_table(frame, target_column, min_rows=min_rows, min_classes=min_classes)


def check_stratifiable(frame: pd.DataFrame, target_column: str, *, min_rows_per_class: int = MIN_ROWS_PER_CLASS) -> dict[Any, int]:
    """Every class needs at least ``min_rows_per_class`` rows for a stratified split; returns the class counts."""
    counts = frame[target_column].value_counts()
    if counts.size < MIN_CLASSES or int(counts.min()) < min_rows_per_class:
        raise ValueError(
            f"Every class needs >= {min_rows_per_class} rows for a stratified holdout: {counts.to_dict()}"
        )
    return {label: int(n) for label, n in counts.items()}


def align_to_schema(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    train_classes: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Require the training schema exactly and (when ``train_classes`` is given) no class unseen in training;
    order columns like train."""
    expected = set(feature_columns) | {target_column}
    if set(frame.columns) != expected:
        raise ValueError(
            f"schema does not match train; missing={sorted(expected - set(frame.columns))}, "
            f"extra={sorted(set(frame.columns) - expected)}"
        )
    if train_classes is not None:
        unseen = sorted(set(frame[target_column]) - set(train_classes), key=str)
        if unseen:
            raise ValueError(f"target classes unseen in training: {unseen}")
    return frame[[*feature_columns, target_column]].reset_index(drop=True)


def fit_categorical_encoder(frame: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, list[str]]:
    """Ordinal maps for non-numeric columns, fitted on the training split only."""
    encoders: dict[str, list[str]] = {}
    for column in feature_columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            continue
        encoders[column] = sorted({str(value) for value in series.dropna().unique()})
    return encoders


def apply_categorical_encoder(
    frame: pd.DataFrame, encoders: Mapping[str, Sequence[str]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply training-fitted ordinal maps; unseen or missing values get the extra 'unknown' code.

    Returns the encoded frame and, per column, how many unseen values were mapped to the unknown code
    (DAT20: the notebook reports them)."""
    out = frame.copy()
    unseen_counts: dict[str, int] = {}
    for column, categories in encoders.items():
        lookup = {category: index for index, category in enumerate(categories)}
        unknown = len(categories)
        encoded, unseen = [], 0
        for value in out[column]:
            if pd.isna(value):
                encoded.append(unknown)
                continue
            key = str(value)
            if key not in lookup:
                unseen += 1
            encoded.append(lookup.get(key, unknown))
        out[column] = encoded
        if unseen:
            unseen_counts[column] = unseen
    return out, unseen_counts


def _missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(column): int(n) for column, n in frame.isna().sum().items() if n > 0}


def _check_inference_frame(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """The checks `read_inference_csv` applies to an inference table (after the raw-header check)."""
    if frame.columns.duplicated().any():
        dupes = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise ValueError(f"Inference CSV contains duplicate column names: {dupes}")
    if "prediction" in frame.columns or any(str(column).startswith("probability_") for column in frame.columns):
        raise ValueError("Inference CSV already contains prediction/probability output columns")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Inference CSV missing features: {missing}")
    return frame


def raw_csv_header(payload: bytes) -> list[str]:
    """First non-empty CSV row, read before pandas can rename duplicate names."""
    reader = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
    for row in reader:
        if row and any(cell.strip() for cell in row):
            return row
    raise ValueError("CSV has no header")


def read_csv_payload(payload: bytes, label: str) -> pd.DataFrame:
    """Read a CSV payload, refusing duplicate header names before pandas renames them (DAT16)."""
    header = raw_csv_header(payload)
    dupes = sorted({name for name in header if header.count(name) > 1})
    if dupes:
        raise ValueError(f"{label} contains duplicate column names: {dupes}")
    return pd.read_csv(io.BytesIO(payload))


def read_inference_csv(payload: bytes, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Read an unlabelled inference CSV: unique header, no `prediction`/`probability_*` columns, every feature present."""
    return _check_inference_frame(read_csv_payload(payload, "Inference CSV"), feature_columns)


def classification_metrics(y_true: Any, y_pred: Any, y_proba: Any, classes: Sequence[Any]) -> dict[str, float]:
    """The repository's metric set: accuracy, balanced_accuracy, f1_weighted (discrete correctness under the
    argmax rule), log_loss (probability quality; lower is better) and roc_auc (ranking quality; binary or
    one-vs-rest; NaN when undefined, e.g. a single class present)."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    y = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    proba = np.asarray(y_proba, dtype=float)
    labels = list(classes)
    if y.shape != pred.shape or y.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    if proba.ndim != 2 or proba.shape != (y.size, len(labels)):
        raise ValueError("y_proba must have one row per example and one column per class")
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_weighted": float(f1_score(y, pred, average="weighted")),
        "log_loss": float(log_loss(y, proba, labels=labels)),
    }
    try:
        if len(labels) == 2:
            out["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
        else:
            out["roc_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=labels))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def majority_class_baseline(train_targets: Any, holdout_targets: Any) -> dict[str, float]:
    """The trivial baseline: always predict the most frequent support class, with the support class shares as
    the probabilities (so log_loss is defined; roc_auc is NaN because constant scores rank nothing)."""
    train = pd.Series(np.asarray(train_targets).reshape(-1))
    holdout = np.asarray(holdout_targets).reshape(-1)
    if train.size == 0 or holdout.size == 0:
        raise ValueError("targets must be non-empty")
    shares = train.value_counts(normalize=True).sort_index()
    classes = list(shares.index)
    unseen = sorted(set(holdout) - set(classes), key=str)
    if unseen:
        raise ValueError(f"holdout has target classes unseen in training: {unseen}")
    majority = shares.idxmax()
    proba = np.tile(shares.to_numpy(dtype=float), (holdout.size, 1))
    metrics = classification_metrics(holdout, np.full(holdout.shape, majority, dtype=object), proba, classes)
    metrics["roc_auc"] = float("nan")
    return metrics


def compare_metric(candidate: Mapping[str, float], reference: Mapping[str, float], name: str) -> bool:
    """True when `candidate` beats `reference` on metric `name` (lower log_loss, higher everything else)."""
    candidate_value, reference_value = float(candidate[name]), float(reference[name])
    if not (math.isfinite(candidate_value) and math.isfinite(reference_value)):
        raise ValueError(f"{name} unavailable for selection")
    if name == "log_loss":
        return candidate_value < reference_value
    return candidate_value > reference_value


# ---------------------------------------------------------------------------
# Role stages (DAT24 / EVAL21).
# ---------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "input": "pandas.DataFrame, one row per example; numeric/categorical features plus a categorical target",
    "columns": "unique names; non-numeric feature columns are ordinal-encoded with maps fitted on training",
    "target": (
        "any label type; rows whose target is missing are dropped and counted; at least MIN_CLASSES distinct "
        "classes, and every holdout/test class must be present in the support rows"
    ),
    "train_rows": [MIN_TRAIN_ROWS, MAX_TRAIN_ROWS],
    "eval_rows": [MIN_EVAL_ROWS, None],
    "features": [1, MAX_FEATURES],
    "classes": [MIN_CLASSES, None],
    "rows_per_class_for_stratified_split": [MIN_ROWS_PER_CLASS, None],
    "inference_input": (
        "every fitted feature column present; no `prediction` or `probability_*` column; extras pass through"
    ),
    "decision_rule": DECISION_RULE,
    "preprocessing": (
        "no scaling by the package; unseen or missing categorical values map to the fitted 'unknown' code; "
        "the model is conditioned on the (encoded) training rows at prediction time"
    ),
}


def validate_inputs(
    frame: pd.DataFrame,
    target_column: str | None = "target",
    *,
    feature_columns: Sequence[str] | None = None,
    min_rows: int = MIN_TRAIN_ROWS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observed table properties, verdict).

    With a ``target_column`` the table is checked exactly as ``prepare_classification_table`` checks it (the
    number of dropped missing-target rows is recorded, not hidden; class counts are observed); with
    ``target_column=None`` it is an inference table checked against ``feature_columns`` exactly as
    ``read_inference_csv`` checks it. Rejection is reported by raising the same error the core function raises.
    """
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (the table's id)")
    table_id = names[0] if names else "table-0"
    if target_column is None:
        if feature_columns is None:
            raise ValueError("feature_columns is required to validate an inference table")
        checked = _check_inference_frame(frame, list(feature_columns))
        entry: dict[str, Any] = {
            "id": table_id,
            "mode": "inference",
            "rows": len(checked),
            "feature_columns": list(feature_columns),
            "extra_columns": [column for column in checked.columns if column not in feature_columns],
            "missing_value_columns": _missing_counts(checked[list(feature_columns)]),
        }
    else:
        cleaned, dropped = _check_classification_table(frame, target_column, min_rows=min_rows)
        features = [column for column in cleaned.columns if column != target_column]
        encoders = fit_categorical_encoder(cleaned, features)
        counts = cleaned[target_column].value_counts()
        entry = {
            "id": table_id,
            "mode": "fit",
            "rows": len(cleaned),
            "dropped_missing_target_rows": dropped,
            "feature_columns": features,
            "categorical_columns": sorted(encoders),
            "missing_value_columns": _missing_counts(cleaned[features]),
            "classes": [str(label) for label in counts.sort_index().index],
            "class_counts": {str(label): int(n) for label, n in counts.sort_index().items()},
            "majority_class_share": float(counts.max() / counts.sum()),
        }
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [entry],
        "target_column": target_column,
        "min_rows": min_rows if target_column is not None else None,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    metrics: Mapping[str, float] | None,
    *,
    baseline: Mapping[str, float] | None = None,
    independent_test: Mapping[str, float] | None = None,
    n_holdout: int | None = None,
    n_test: int | None = None,
    class_labels: Sequence[Any] | None = None,
    target_column: str | None = None,
    selection: str | None = None,
    sample_kind: str = "sample",
    estimation: str = "single seeded stratified holdout; no dispersion estimate",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    ``metrics`` / ``independent_test`` are dicts from ``classification_metrics`` and ``baseline`` from
    ``majority_class_baseline``; the verdict is ``sample-sanity``. Without metrics (no labelled rows) the
    verdict is ``not-measurable`` and the report says what labelled data would make the task measurable.
    """

    def _entries(source: Mapping[str, float]) -> list[dict[str, Any]]:
        unknown = sorted(set(source) - set(METRIC_IDS))
        if unknown:
            raise ValueError(f"unknown metric ids {unknown}; classification_metrics reports {list(METRIC_IDS)}")
        return [
            {
                "id": metric_id,
                "value": None if not math.isfinite(float(source[metric_id])) else float(source[metric_id]),
                "units": "nats" if metric_id == "log_loss" else "unitless",
                "higher_is_better": metric_id != "log_loss",
            }
            for metric_id in METRIC_IDS
            if metric_id in source
        ]

    base: dict[str, Any] = {
        "task": "tabular classification by in-context conditioning on labelled support rows",
        "decision_rule": DECISION_RULE,
        "score_semantics": "class probabilities in classes_ order, uncalibrated; no threshold shipped",
        "sample_kind": sample_kind,
        "n_holdout": n_holdout,
        "n_test": n_test,
        "class_labels": None if class_labels is None else [str(label) for label in class_labels],
        "target_column": target_column,
        "selection": selection,
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if metrics is None:
        return {
            **base,
            "metrics": [],
            "independent_test": [],
            "verdict": "not-measurable",
            "reason": "no labelled holdout rows were supplied for the scored table",
            "needs": (
                "a labelled holdout table whose classes all appear in the support rows, scored with "
                "`classification_metrics` (accuracy, balanced_accuracy, f1_weighted, log_loss, roc_auc) against "
                "`majority_class_baseline`; an independent test partition from the deployment domain for any "
                "generalisable claim"
            ),
        }
    reported = [{**entry, "estimation": estimation} for entry in _entries(metrics)]
    test_entries: list[dict[str, Any]] = []
    if independent_test is not None:
        test_estimation = "independent test partition, single run"
        test_entries = [{**e, "estimation": test_estimation} for e in _entries(independent_test)]
    baselines = [] if baseline is None else [{"id": "majority_class", "metrics": _entries(baseline)}]
    rows = "an unstated number of" if n_holdout is None else str(n_holdout)
    return {
        **base,
        "metrics": reported,
        "independent_test": test_entries,
        "baselines": baselines,
        "verdict": "sample-sanity",
        "reason": f"{rows} labelled holdout row(s) from one seeded stratified split; tutorial evidence, not a benchmark",
        "needs": (
            "an independent, domain-representative labelled test set for any generalisable quality claim; the "
            "class probabilities are uncalibrated and any decision threshold must be chosen on the caller's data"
        ),
    }


# ---------------------------------------------------------------------------
# Serving-artifact ZIP handling shared by the producer's fresh-reload check and the companion notebook.
# ---------------------------------------------------------------------------


def safe_extract_zip(
    zip_path: str | Path, dest: str | Path, *, max_expanded_bytes: int = MAX_ARTIFACT_EXPANDED_BYTES
) -> Path:
    """Extract a ZIP member by member after every member passed the path, symlink and size checks (AINF3).

    Members are copied individually (never ``extractall``) so a member that fails a check is never written.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        expanded_bytes = 0
        for info in archive.infolist():
            if "\\" in info.filename:
                raise ValueError(f"Ambiguous backslash ZIP member: {info.filename}")
            expanded_bytes += info.file_size
            if expanded_bytes > max_expanded_bytes:
                raise ValueError(f"Artifact exceeds {max_expanded_bytes} expanded bytes")
            name = info.filename
            parts = Path(name).parts
            mode = info.external_attr >> 16
            if name.startswith("/") or ".." in parts or stat.S_ISLNK(mode):
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            target = (dest / Path(name)).resolve()
            if root != target and root not in target.parents:
                raise ValueError("ZIP member escapes destination")
        for info in archive.infolist():
            target = (dest / Path(info.filename)).resolve()
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    return root


def manifest_member_path(root: str | Path, value: Any, field: str) -> Path:
    """Resolve a manifest path inside the bundle root, refusing absolute paths and traversal."""
    rel = Path(str(value))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe {field} path in artifact.json: {value!r}")
    root_resolved = Path(root).resolve()
    target = (root_resolved / rel).resolve()
    if root_resolved != target and root_resolved not in target.parents:
        raise ValueError(f"{field} path escapes artifact root: {value!r}")
    return target


def verify_artifact_bundle(root: str | Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Check an extracted bundle's allowlist, sizes and digests against its manifest; return member paths."""
    root = Path(root)
    ckpt = manifest_member_path(root, manifest["checkpoint"], "checkpoint")
    context_path = manifest_member_path(root, manifest["trainingContext"], "trainingContext")
    payload_files = manifest.get("payloadFiles")
    if payload_files is not None:
        expected_files = {"artifact.json", *payload_files}
        actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if actual_files != expected_files:
            unexpected = sorted(actual_files ^ expected_files)
            raise RuntimeError(f"Unexpected or missing artifact files: {unexpected}")
    if manifest.get("sizes"):
        if ckpt.stat().st_size != manifest["sizes"]["checkpoint"]:
            raise RuntimeError("Checkpoint size mismatch")
        if context_path.stat().st_size != manifest["sizes"]["trainingContext"]:
            raise RuntimeError("Training-context size mismatch")
    if sha256_file(ckpt) != manifest["digests"]["checkpointSha256"]:
        raise RuntimeError("Checkpoint digest mismatch")
    if sha256_file(context_path) != manifest["digests"]["trainingContextSha256"]:
        raise RuntimeError("Training-context digest mismatch")
    return {"checkpoint": ckpt, "training_context": context_path}
