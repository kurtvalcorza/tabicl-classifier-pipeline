"""Stage runner for the standalone TabICLv2 classifier E2E tutorial (NOTEBOOK_SPEC 2.5 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --outputs OUTPUTS --weights WEIGHTS --stage data --options '{...}'

Nothing is installed into the notebook kernel. Each stage is a separate process and starts from files only: the
verified checkpoint under ``--weights``, the partitions, encoders and records of earlier stages under
``RUN_DIR/state``, and the learner-facing exports under ``--outputs``. In-context conditioning (``fit``) registers the
support rows and is repeated by every stage that needs the model; a fine-tuned candidate is kept as a checkpoint file.
On failure a stage writes ``RUN_DIR/state/<stage>.error.json``, which the notebook re-raises in the kernel.

Stages: weights → data → validate → condition → report → fresh_test → predict → export → reload, then the closing
external-artifact section (NOTEBOOK_SPEC 3.0 §19.1) external_artifact → external_reconstruct → external_predict, plus the
optional ``activity``. The external-artifact stages load the pinned sample artifact (a release asset produced by a recorded
earlier run of this notebook, pinned by URL and SHA-256), never this run's export, and read none of the main path's state.
The fresh test of the default MAGIC sample is read only by ``fresh_test``, only under the pre-registered settings (docs/magic-finetune-preregistration.md), and never when ``DIMER_MASK_FRESH_TEST=1``.
The ``data`` and ``validate`` stages import no model library, so CI exercises them directly.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

STEM = "tabiclv2_classifier"
PACKAGE = "tabicl_classifier_pipeline"
SEED = 42
VALIDATION_SPLIT = 0.20
NEW_ROWS = 8
RELOAD_RTOL, RELOAD_ATOL = 1e-5, 1e-7
MAGIC_SOURCE = "Sample: MAGIC Gamma Telescope"
REFERENCE_SOURCE = "Sample: Breast Cancer"  # the earlier default; its 114-row test is an exposed development test
SAMPLE_SOURCES = (MAGIC_SOURCE, REFERENCE_SOURCE, "Sample: Wine", "Sample: Palmer Penguins")
DEFAULT_SOURCE = MAGIC_SOURCE
UPLOAD_SOURCES = ("Upload CSV", "Upload pre-split train/val/test")
PENGUINS_URL = "https://raw.githubusercontent.com/kurtvalcorza/tabicl-classifier-pipeline/169e60fa8d956aa389c144ac9c7988b92db84a79/examples/sample-data/palmer-penguins.zip"
PENGUINS_SHA256 = "fe894295ccc0a447dc020f5d57798968b95eae3da1a4cbe09a638e6b1bd0ba44"
NUMERIC_LIKE_SHARE = 0.9  # an object column whose values are >= 90 % numbers is treated as a numeric column with typos
MIN_SELECTION_HOLDOUT_ROWS = 50

# --- MAGIC Gamma Telescope (UCI 159, CC BY 4.0): pinned split archive, committed at an immutable revision (R96-D1) ---
MAGIC_COMMIT = "fc2a5adb68981e01bb00cae3ea5ea2f377412383"
MAGIC_URL = f"https://raw.githubusercontent.com/kurtvalcorza/tabicl-classifier-pipeline/{MAGIC_COMMIT}/examples/sample-data/magic-gamma.zip"
MAGIC_SHA256 = "2c6108e6d3e53cd9fcc8417df6cc8018ea2faa70a3e9b8e58fd3ef4e1463497d"
MAGIC_SOURCE_SHA256 = "a52d3a378e43df7b678c4c0b2f03a66a058bbfeb27c95ccd037a767e7d172984"
MAGIC_MEMBERS = {
    "train.csv": "0a5f1692662ad23807339a91ebfeef26cc22871ad5df535bd284aea70d16a246",
    "val.csv": "36bc23405d7a5bee814c21297140ffd70436c147c236cbaf2881613a1ebdb0b4",
    "test.csv": "62c3eff6bb323ac150ec60eb4f26d5ed92ce20fe79ed3baa83f32b76e4482b51",
}
MAGIC_ROW_IDS_SHA256 = {
    "train": "009082b81a485027433be0fcb0de99855a480b425ebbb9e70bd37242307210cd",
    "val": "22dbeb1022a76e2cab51dd7a2df9869efa71430867d975f21ac59369e84d17b3",
    "test": "8847cab4e87c8ff30dd6a27045464f7891b2e14dd39149c5e5648348f190f3e4",
    "reserve": "c1b30483d5024fa6f94893a824dcbacf60ca942b941faa790edbb541c97208b3",
}
MAGIC_FRESH_TEST_ROWS = 1500

# --- External-artifact section (NOTEBOOK_SPEC 3.0 §19.1, XAS1–XAS7, SART1–SART11) ---------------------------------------
# The default artifact of the closing section is a release asset of this repository: the deterministic export
# (`outputs/tabiclv2_classifier_artifact.zip`) of a recorded earlier hosted run of this notebook's default path (SART8,
# SART9). It is pinned by URL and SHA-256 and never replaced; a changed artifact gets a new tag and a new pin (SART7,
# SART11). The export of the current run is never used as the external artifact (XAS2).
SAMPLE_ARTIFACT_TAG = "sample-artifact-magic-v1"
SAMPLE_ARTIFACT_URL = "https://github.com/kurtvalcorza/tabicl-classifier-pipeline/releases/download/sample-artifact-magic-v1/tabiclv2_classifier_artifact.zip"  # literal: validators allow only this exact URL
SAMPLE_ARTIFACT_SHA256 = "2dc82d55161382602fa7a5d1e217e3a33b6c6188ba6ddf4f7ff6b087bf67114c"
SAMPLE_ARTIFACT_PRODUCER = {
    "notebook": "tutorials/tabiclv2_classifier_colab.ipynb (default path: MAGIC Gamma Telescope sample)",
    "commit": "383ca046184b2a3aead0c9dd94503338bfbd1253",
    "run": "recorded hosted Colab T4 run; the release notes of the tag name the run record (SART8)",
    "release_tag": SAMPLE_ARTIFACT_TAG,
}
SAMPLE_ARTIFACT_MAX_BYTES = 512 * 1024 * 1024  # a download larger than this is refused before it is hashed or opened
EXTERNAL_DIR = "external_artifact"  # under outputs/: the section's predictions and provenance (XAS6)
EXTERNAL_SAMPLE_ROWS = 8
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

# --- pre-registered fine-tuning grid, selection and reading rule (docs/magic-finetune-preregistration.md) ---
GRID_COMMON = {"weight_decay": 0.01, "grad_clip": 1.0, "max_data_size": 500, "use_lr_scheduler": False, "finetune_ctx_query_ratio": 0.2, "early_stopping": True, "eval_metric": "log_loss", "min_delta": 1e-4, "n_estimators_inference": 8}
GRID = (
    {"id": "C1", "role": "production-like", "learning_rate": 1e-5, "epochs": 30, "patience": 8, "n_estimators_finetune": 2, "n_estimators_validation": 2, "time_limit": 1800},
    {"id": "C2", "role": "larger steps", "learning_rate": 5e-5, "epochs": 10, "patience": 3, "n_estimators_finetune": 2, "n_estimators_validation": 2, "time_limit": 600},
    {"id": "C3", "role": "smaller steps", "learning_rate": 1e-6, "epochs": 10, "patience": 3, "n_estimators_finetune": 2, "n_estimators_validation": 2, "time_limit": 600},
)
SELECTION_TOLERANCE = 0.005  # validation log loss a config must beat the pretrained model by; otherwise keep pretrained
SELECTION_TIE = 1e-4
INFERENCE_ESTIMATORS = 8
BOOTSTRAP = {"seed": 20261019, "n_boot": 10000}
PRACTICAL_DELTA = 0.005  # "improved" needs the CI upper bound < 0 and delta <= -PRACTICAL_DELTA
DETERMINISM = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "cudnn_deterministic": True, "cudnn_benchmark": False, "deterministic_algorithms": True, "warn_only": False, "seed": SEED, "sdpa_backend": "MATH", "flash_attn3": False}
# Amendment 2: every stage that runs the model does so inside deterministic_kernels() (MATH attention, FlashAttention-3 off).
MODEL_STAGES = frozenset(("condition", "report", "fresh_test", "predict", "export", "reload", "activity", "external_reconstruct", "external_predict"))
DETERMINISM_WARNING = re.compile(r"determinis|non-?deterministic|memory.efficient|mem_efficient|efficient attention", re.IGNORECASE)
_CAUGHT_WARNINGS: list = []  # the warnings recorded by the active deterministic_kernels() context
REQUIRED_DEVICE = "Tesla T4"
MASK_ENV = "DIMER_MASK_FRESH_TEST"
# The evaluation of record: the first hosted T4 run under the pre-registered settings (amendment 2), recorded in
# docs/release-verification.md with its evidence. Because a record exists, every pre-registered scoring of the fresh test, the
# first in a new runtime included, is a "reproduction" and prints matches_record / differs against the printed values below.
# The record kept the pretrained model, which is also what a CPU run (no grid) exports.
# Issue #28: on a Tesla T4 the fp16 fine-tuning grid hash varies run to run even at an identical revision. Two hosted runs of the
# same notebook blob (f21920ab) with the same carried stage runner (ee759496...), at commits d4172f5 and edbe94d, printed
# 2aa9e0cd... and 4b1bf536...; their selection and every fresh-test value matched the record exactly. So grid_hash is reported
# (record, this run, equal or not) as information only and never decides matches_record. The record still names the SHA-256 of
# the stage runner that produced it, which is reported the same way.
RECORDED_SELECTION: str = "pretrained"
RECORDED_EVALUATION: dict[str, Any] | None = {
    "label": "pre-registered evaluation (this runtime)",
    "date": "2026-10-20",
    "runner": "Colab CLI sequential execution, fresh hosted Tesla T4 session",
    "commit": "9ea0888d9df24a9568613ec9c51dd1fe65cea443",
    "notebook_blob": "947feb162abbf8126a8579188268024da3ca3ba8",
    "evidence": "docs/execution-evidence/2026-10-20-9ea0888/",
    "device": "Tesla T4",
    "selected": RECORDED_SELECTION,
    "grid_hash": "4b1bf5366b121dd2365c6c9b02408e1b825c247bb6636fee6d8fe35697b20692",
    # SHA-256 of the carried tutorial_stages.py (UTF-8, LF) that produced the record: NOTEBOOK_SOURCE['files']['tutorial_stages.py']
    # in blob 947feb16, equal to tools/tutorial_stages.py at 15c1ae4 (unchanged at 9ea0888).
    "stage_runner_sha256": "cc604098aa235653c2ef5b36cffbca37332539a9ac74e1b52912e1ff71184823",
    "reading": {
        "primary.delta": 0.0,
        "primary.ci95": [0.0, 0.0],
        "primary.verdict": "same model (Δ = 0)",
        "accuracy.pretrained": 0.8993,
        "accuracy.pretrained_wilson95": [0.8831, 0.9136],
        "accuracy.selected": 0.8993,
        "accuracy.difference": 0.0,
        "roc_auc.pretrained": 0.95,
        "roc_auc.selected": 0.95,
        "sensitivity.excluded_rows": 4,
        "sensitivity.n": 1496,
        "sensitivity.primary.delta": 0.0,
        "sensitivity.primary.verdict": "same model (Δ = 0)",
    },
}


def printed_reading(reading: dict[str, Any], sensitivity: dict[str, Any]) -> dict[str, Any]:
    """The fresh-test values in the form the record stores them (rounded as printed)."""
    r = rounded({"reading": reading, "sensitivity": sensitivity})
    rd, sv = r["reading"], r["sensitivity"]
    return {
        "primary.delta": rd["primary"]["delta"],
        "primary.ci95": list(rd["primary"]["ci95"]),
        "primary.verdict": rd["primary"]["verdict"],
        "accuracy.pretrained": rd["accuracy"]["pretrained"],
        "accuracy.pretrained_wilson95": list(rd["accuracy"]["pretrained_wilson95"]),
        "accuracy.selected": rd["accuracy"]["selected"],
        "accuracy.difference": rd["accuracy"]["difference"],
        "roc_auc.pretrained": rd["roc_auc"]["pretrained"],
        "roc_auc.selected": rd["roc_auc"]["selected"],
        "sensitivity.excluded_rows": sv["excluded_rows"],
        "sensitivity.n": sv["n"],
        "sensitivity.primary.delta": sv["primary"]["delta"],
        "sensitivity.primary.verdict": sv["primary"]["verdict"],
    }


def stage_runner_sha256() -> str:
    """SHA-256 of this stage runner as the notebook stamps it (UTF-8, LF): NOTEBOOK_SOURCE['files']['tutorial_stages.py']."""
    return hashlib.sha256(Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()


GRID_HASH_NOTE = "informational only, not compared: on a T4 the fp16 grid hash varies run to run at an identical revision (issue #28)"


def compare_with_record(selected: str, grid_hash_value: str, device: str | None, values: dict[str, Any], runner_sha256: str | None = None) -> dict[str, Any] | None:
    """A reproduction against the evaluation of record: the fields that differ, by name. None when no record exists.

    The selection, the device and every fresh-test value must match the record exactly. The grid hash is reported, with the
    record's value, this run's value and whether they are equal, but is informational and never a difference, whatever the stage
    runner (issue #28: on a T4 the fp16 grid hash has varied run to run at an identical revision). The stage-runner digests are
    reported the same way."""
    if RECORDED_EVALUATION is None:
        return None
    current = runner_sha256 if runner_sha256 is not None else stage_runner_sha256()
    recorded_runner = RECORDED_EVALUATION.get("stage_runner_sha256")
    differs = [k for k, v in (("selected", selected), ("device", device)) if RECORDED_EVALUATION.get(k) != v]
    recorded = RECORDED_EVALUATION.get("reading") or {}
    differs += [f"reading.{k}" for k in recorded if values.get(k) != recorded[k]]
    recorded_grid = RECORDED_EVALUATION.get("grid_hash")
    return {
        "reproduction_of": f"{RECORDED_EVALUATION.get('date')} hosted T4, commit {str(RECORDED_EVALUATION.get('commit'))[:7]}",
        "stage_runner_sha256": {"record": recorded_runner, "this_run": current},
        "same_stage_runner": recorded_runner == current,
        "grid_hash": {"informational": GRID_HASH_NOTE, "record": recorded_grid, "this_run": grid_hash_value, "equal": recorded_grid == grid_hash_value},
        "matches_record": not differs,
        "differs": differs,
    }
TEST_LOOKS = ("pre-registered evaluation (this runtime)", "reproduction", "not scored (exploratory)")
FRESH_DIR = "fresh-test"  # under outputs/; not removed by clear_outputs, so a re-run in the same runtime is a reproduction


def fresh_test_masked() -> bool:
    """Re-read on every call (a mask copied once at install fails open)."""
    return os.environ.get(MASK_ENV, "").strip().lower() in {"1", "true", "yes"}


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write ``path`` via a temporary file and ``os.replace`` so an interrupted write never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def atomic_write_json(path: Path, value: Any) -> Path:
    return atomic_write_bytes(path, json.dumps(value, indent=2, ensure_ascii=False, default=str).encode("utf-8"))


def apply_determinism() -> dict[str, Any]:
    """Pre-registered determinism flags. ``CUBLAS_WORKSPACE_CONFIG`` is set by ``main`` before torch is imported. An op
    with no deterministic CUDA kernel raises (no ``warn_only``) and stops the stage, naming the op."""
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()), "seed": SEED, **attention_backend()}


def attention_backend() -> dict[str, Any]:
    """The attention path as PyTorch and tabicl report it now (observed, not assumed): ``sdpa_backend`` is "MATH" only when
    the math kernel is the sole enabled SDPA backend; ``flash_attn3`` is tabicl's own FlashAttention-3 switch."""
    import torch

    cuda = torch.backends.cuda
    enabled = {"FLASH": cuda.flash_sdp_enabled(), "MEM_EFFICIENT": cuda.mem_efficient_sdp_enabled(), "MATH": cuda.math_sdp_enabled(), "CUDNN": bool(getattr(cuda, "cudnn_sdp_enabled", lambda: False)())}
    names = sorted(name for name, on in enabled.items() if on)
    try:
        tabicl_attention = importlib.import_module("tabicl._model.attention")
        flash3 = bool(getattr(tabicl_attention, "HAS_FLASH_ATTN3", False) and getattr(tabicl_attention, "_use_flash_attn3", False))
    except ImportError:
        flash3 = None
    return {"sdpa_backend": "+".join(names) or "NONE", "flash_attn3": flash3}


def determinism_warnings() -> list[str]:
    """Determinism-related warnings captured so far in this stage (e.g. a kernel that only warns)."""
    return sorted({str(w.message) for w in _CAUGHT_WARNINGS if DETERMINISM_WARNING.search(str(w.message))})


@contextlib.contextmanager
def deterministic_kernels():
    """Amendment 2: the MATH SDPA backend and tabicl's FlashAttention-3 path off around a whole model stage (fine-tuning
    and every inference), the pre-registered determinism flags, and every Python warning captured; determinism-related
    warnings are printed, so a non-deterministic kernel that only warns shows up in the log."""
    import warnings

    from torch.nn.attention import SDPBackend, sdpa_kernel

    tabicl_attention = importlib.import_module("tabicl._model.attention")  # the upstream attention switch, not the model API

    with contextlib.ExitStack() as stack:
        caught = stack.enter_context(warnings.catch_warnings(record=True))
        warnings.simplefilter("always")
        stack.enter_context(sdpa_kernel(SDPBackend.MATH))
        stack.enter_context(tabicl_attention.flash_attn3_toggle(False))
        # tabicl's inference re-toggles FA3 from its own `use_fa3` per forward pass; switching the capability flag off
        # too makes the path unreachable for the whole stage (flash_attn_interface is not in the lock anyway).
        stack.callback(setattr, tabicl_attention, "HAS_FLASH_ATTN3", getattr(tabicl_attention, "HAS_FLASH_ATTN3", False))
        tabicl_attention.HAS_FLASH_ATTN3 = False
        globals()["_CAUGHT_WARNINGS"] = caught  # determinism_warnings() reads the live record during the stage
        apply_determinism()
        try:
            yield
        finally:
            seen = set()
            for w in caught:
                text = f"{w.category.__name__}: {w.message}"
                if text in seen:
                    continue
                seen.add(text)
                prefix = "DETERMINISM WARNING" if DETERMINISM_WARNING.search(str(w.message)) else "warning"
                print(f"{prefix}: {text}", flush=True)
            globals()["_CAUGHT_WARNINGS"] = list(caught)


def device_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


class Run:
    """Paths of one run: carried sources and state under ``root``; learner-facing files under ``outputs``."""

    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        return atomic_write_json(self.state / name, value)

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{name} is incomplete or corrupt (an interrupted stage?): re-run the stage that writes it before '{needed_by}' (run the notebook from Section 4)") from exc

    def write_frame(self, name: str, frame) -> None:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=True, engine="pyarrow")
        atomic_write_bytes(self.state / name, buffer.getvalue())

    def read_frame(self, name: str, needed_by: str):
        import pandas as pd

        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return pd.read_parquet(path, engine="pyarrow")

    def write_output(self, name: str, value: Any) -> Path:
        return atomic_write_json(self.out / name, value)


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


# --------------------------------------------------------------------------------------------------
# data (TIC-m1, TIC-m2)
# --------------------------------------------------------------------------------------------------


def stratified_60_20_20(frame, target: str):
    from sklearn.model_selection import train_test_split

    train, remainder = train_test_split(frame, test_size=0.4, random_state=SEED, stratify=frame[target])
    holdout, test = train_test_split(remainder, test_size=0.5, random_state=SEED, stratify=remainder[target])
    return train, holdout, test


def load_sample(P, source: str):
    """(train, holdout, test, name, target) for one of the three public samples."""
    if source == "Sample: Breast Cancer":
        from sklearn.datasets import load_breast_cancer

        dataset = load_breast_cancer(as_frame=True)
        frame = dataset.frame.rename(columns={dataset.target.name: "target"})
        frame["target"] = frame["target"].map({0: "malignant", 1: "benign"})
        return (*stratified_60_20_20(frame, "target"), "sklearn-breast-cancer", "target")
    if source == "Sample: Wine":
        from sklearn.datasets import load_wine

        dataset = load_wine(as_frame=True)
        frame = dataset.frame.rename(columns={dataset.target.name: "target"})
        frame["target"] = frame["target"].map(dict(enumerate(dataset.target_names)))
        return (*stratified_60_20_20(frame, "target"), "sklearn-wine", "target")
    if source == "Sample: Palmer Penguins":
        import urllib.request
        import zipfile

        with urllib.request.urlopen(PENGUINS_URL, timeout=30) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != PENGUINS_SHA256:
            raise ValueError(f"Palmer Penguins sample archive digest mismatch: {digest} != {PENGUINS_SHA256}")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = {Path(n).name: n for n in archive.namelist() if n.endswith(".csv")}
            if not {"train.csv", "val.csv", "test.csv"} <= set(names):
                raise ValueError("Palmer Penguins sample archive is missing train.csv, val.csv, or test.csv")
            parts = [P.read_csv_payload(archive.read(names[n]), n) for n in ("train.csv", "val.csv", "test.csv")]
        return (*parts, "palmer-penguins-cc0", "species")
    if source == MAGIC_SOURCE:
        raise ValueError("the MAGIC sample is loaded by the data stage (it needs the run's cache); choose another activity table")
    raise ValueError(f"DATA_SOURCE must be one of {list(SAMPLE_SOURCES + UPLOAD_SOURCES)}, got {source!r}")


def fetch_magic(cache: Path) -> bytes:
    """The pinned MAGIC archive: reused from ``cache`` when its digest matches, else fetched at the immutable commit."""
    if cache.is_file() and hashlib.sha256(cache.read_bytes()).hexdigest() == MAGIC_SHA256:
        return cache.read_bytes()
    import urllib.request

    with urllib.request.urlopen(MAGIC_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != MAGIC_SHA256:
        raise ValueError(f"MAGIC sample archive digest mismatch: {digest} != {MAGIC_SHA256}")
    atomic_write_bytes(cache, payload)
    return payload


def magic_member(payload: bytes, name: str) -> bytes:
    """One CSV of the pinned archive, digest-checked. ``test.csv`` is read only by the fresh_test stage."""
    import zipfile

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        data = archive.read(name)
    digest = hashlib.sha256(data).hexdigest()
    if digest != MAGIC_MEMBERS[name]:
        raise ValueError(f"MAGIC sample member {name} digest mismatch: {digest} != {MAGIC_MEMBERS[name]}")
    return data


def check_byod_table(frame, name: str, target: str, text_columns: list[str]) -> None:
    """Refuse, naming the file and the rule, a missing target column and numeric columns spoiled by a few strings
    (they would otherwise be ordinal-encoded silently by sorted string value)."""
    import pandas as pd

    if target not in frame.columns:
        raise ValueError(f"{name}: the target column {target!r} is not in the header {list(map(str, frame.columns))}. Set TARGET_COLUMN to the name of your label column.")
    absent = [c for c in text_columns if c not in frame.columns]
    if absent:
        raise ValueError(f"{name}: TEXT_COLUMNS {absent} are not in the header {list(map(str, frame.columns))}.")
    for column in frame.columns:
        if column == target or column in text_columns or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        numeric = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
        share = float(numeric.notna().mean())
        if share >= NUMERIC_LIKE_SHARE:
            bad = values[numeric.isna()].astype(str).unique().tolist()[:5]
            raise ValueError(
                f"{name}: column {column!r} is numeric except for {int(numeric.isna().sum())} value(s) {bad}; it would be "
                "encoded as categories and lose its numeric meaning. Fix those values (leave missing cells empty), or list "
                "the column in TEXT_COLUMNS if it really is categorical."
            )


def clear_outputs(run: Run) -> list[str]:
    removed = []
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    for name in ("artifact", "artifact-reload"):
        if (run.out / name).is_dir():
            shutil.rmtree(run.out / name)
            removed.append(name + "/")
    return removed


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
    served = P.ensure_converted(snapshot)  # one torch.load(weights_only=True) of the verified source; never again after this
    print({"served_weights": served.name, "served_sha256": P.sha256_file(served), "pinned": P.CONVERTED_MODEL_SHA256})
    run.write_state("weights.json", {"snapshot": str(snapshot), "checkpoint": str(snapshot / P.BASE_CHECKPOINT_NAME), "served": str(served)})


def stage_data(run: Run) -> None:
    import pandas as pd

    P = package(run.root)
    opts = run.options
    source = opts.get("data_source", DEFAULT_SOURCE)
    use_byod = bool(opts.get("use_byod", False))
    masked = fresh_test_masked()
    print({"fresh_test_mask": f"{MASK_ENV}=1: the fresh test is not read in this run (the pinned archive holding it is still downloaded and digest-checked; test.csv is never extracted)" if masked else f"{MASK_ENV} unset: the fresh test is read only by the fresh_test stage, only under the pre-registered settings"})
    if use_byod and source in SAMPLE_SOURCES:
        raise ValueError('USE_BYOD=True requires DATA_SOURCE = "Upload CSV" or "Upload pre-split train/val/test".')
    if source in UPLOAD_SOURCES and not use_byod:
        raise ValueError("Set USE_BYOD=True to use an upload DATA_SOURCE.")
    removed = clear_outputs(run)
    for name in ("train.parquet", "holdout.parquet", "test.parquet", "data.json", "encoded.json", "condition.json", "export_reference.json"):
        (run.state / name).unlink(missing_ok=True)
    if removed:
        print({"removed_previous_outputs": removed})
    text_columns = list(opts.get("text_columns") or [])
    dropped: dict[str, int] = {}
    extra: dict[str, Any] = {}
    if source == MAGIC_SOURCE:
        payload = fetch_magic(run.root / "inputs" / "magic-gamma.zip")
        train = P.read_csv_payload(magic_member(payload, "train.csv"), "train.csv")
        holdout = P.read_csv_payload(magic_member(payload, "val.csv"), "val.csv")
        test, name, target, kind = None, "uci-159-magic-gamma-telescope", "target", "sample"
        extra = {"archive_sha256": MAGIC_SHA256, "source_sha256": MAGIC_SOURCE_SHA256, "row_ids_sha256": MAGIC_ROW_IDS_SHA256, "holdout_role": "validation (1,000 rows): grid selection and early stopping", "test_role": "fresh test (1,500 rows): not loaded here; read only by the fresh_test stage under the pre-registered settings", "fresh_test_rows": MAGIC_FRESH_TEST_ROWS, "fresh_test_member_sha256": MAGIC_MEMBERS["test.csv"], "reserve_rows": 13520}
    elif source in SAMPLE_SOURCES:
        train, holdout, test, name, target = load_sample(P, source)
        kind = "sample"
        extra = {"test_role": "exposed development test (scored in the 2026-10-10 and 2026-10-11 runs; an earlier-run reference)" if source == REFERENCE_SOURCE else "independent test (exploratory sample)"}
    else:
        target = opts.get("target_column") or "target"
        paths = opts.get("paths") or {}
        def read(key: str):
            path = Path(paths.get(key) or "")
            if not str(path) or not path.is_file():
                raise FileNotFoundError(f"BYOD {key} file {str(path)!r} does not exist in this runtime: upload it or correct the path field.")
            if path.suffix.lower() != ".csv":
                raise ValueError(f"{path.name}: BYOD files must be CSV.")
            frame = P.read_csv_payload(path.read_bytes(), path.name)
            check_byod_table(frame, path.name, target, text_columns)
            return frame, path.name
        if source == "Upload CSV":
            frame, name = read("csv")
            raw_manifest = P.validate_inputs(frame, target_column=target, names=[name])
            dropped["csv"] = raw_manifest["inputs"][0]["dropped_missing_target_rows"]
            frame, _ = P.prepare_classification_table(frame, target)
            P.check_stratifiable(frame, target)
            from sklearn.model_selection import train_test_split

            train, holdout = train_test_split(frame, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=frame[target])
            test = None
        else:
            train, train_name = read("train")
            holdout, _ = read("val")
            test = read("test")[0] if paths.get("test") else None
            name = "pre-split upload (" + train_name + ")"
        kind = "BYOD"
        extra = {"test_role": "your test partition (exploratory: BYOD never scores the fresh test)"}
    parts = {"train": train, "holdout": holdout, "test": test}
    for key, frame in parts.items():
        if frame is not None:
            run.write_frame(f"{key}.parquet", frame)
    digest = hashlib.sha256(pd.concat([f for f in parts.values() if f is not None]).to_csv(index=False).encode("utf-8")).hexdigest()
    record = {"sample_kind": kind, "name": name, "source": source, "target": target, "train_rows": len(train), "holdout_rows": len(holdout), "test_rows": 0 if test is None else len(test), "csv_sha256": digest, "dropped_missing_target_rows_before_split": dropped, "text_columns": text_columns, "use_byod": use_byod, **extra}
    run.write_state("data.json", record)
    print(record)
    if dropped.get("csv"):
        print(f"{dropped['csv']} row(s) with a missing target were dropped before the split; Section 5's input manifest records them.")


def stage_validate(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "validate")
    target, name = data["target"], data["name"]
    train = run.read_frame("train.parquet", "validate")
    holdout = run.read_frame("holdout.parquet", "validate")
    test = run.read_frame("test.parquet", "validate") if (run.state / "test.parquet").is_file() else None
    print({"ceilings": {"MIN_TRAIN_ROWS": P.MIN_TRAIN_ROWS, "MIN_EVAL_ROWS": P.MIN_EVAL_ROWS, "MAX_TRAIN_ROWS": P.MAX_TRAIN_ROWS, "MAX_FEATURES": P.MAX_FEATURES, "MIN_CLASSES": P.MIN_CLASSES, "MIN_ROWS_PER_CLASS": P.MIN_ROWS_PER_CLASS}, "decision_rule": P.DECISION_RULE})
    manifest = P.validate_inputs(train, target_column=target, names=[name + ":train"])
    if data["dropped_missing_target_rows_before_split"].get("csv"):
        manifest["inputs"][0]["dropped_missing_target_rows_before_split"] = data["dropped_missing_target_rows_before_split"]["csv"]
    manifest["inputs"].extend(P.validate_inputs(holdout, target_column=target, min_rows=P.MIN_EVAL_ROWS, names=[name + ":holdout"])["inputs"])
    if test is not None:
        manifest["inputs"].extend(P.validate_inputs(test, target_column=target, min_rows=P.MIN_EVAL_ROWS, names=[name + ":test"])["inputs"])
    # Demonstrate rejection on a probe that breaks a ceiling; the finding is recorded, not swallowed.
    try:
        P.validate_inputs(train.head(P.MIN_TRAIN_ROWS - 1), target_column=target)
    except ValueError as exc:
        manifest["findings"].append({"input": "too-few-rows-probe", "verdict": "rejected", "message": str(exc)})
    train_row_ids = list(map(int, train.index)) if train.index.dtype.kind in "iu" else list(range(len(train)))
    holdout_row_ids = list(map(int, holdout.index)) if holdout.index.dtype.kind in "iu" else list(range(len(holdout)))
    train, dropped_train = P.prepare_classification_table(train, target)
    holdout, dropped_holdout = P.prepare_classification_table(holdout, target, min_rows=P.MIN_EVAL_ROWS, min_classes=1)
    dropped_test = 0
    if test is not None:
        test, dropped_test = P.prepare_classification_table(test, target, min_rows=P.MIN_EVAL_ROWS, min_classes=1)
    features = [c for c in train.columns if c != target]
    classes = sorted(train[target].unique(), key=str)
    train = train[features + [target]].reset_index(drop=True)
    holdout = P.align_to_schema(holdout, features, target, classes)
    test = P.align_to_schema(test, features, target, classes) if test is not None else None
    encoders = P.fit_categorical_encoder(train, features)
    train_enc, _ = P.apply_categorical_encoder(train, encoders)
    holdout_enc, unseen_holdout = P.apply_categorical_encoder(holdout, encoders)
    test_enc, unseen_test = P.apply_categorical_encoder(test, encoders) if test is not None else (None, {})
    baseline = P.majority_class_baseline(train_enc[target], holdout_enc[target])
    run.write_output(f"{STEM}_input_manifest.json", manifest)
    print(json.dumps(manifest["inputs"][0], indent=2))
    print("findings:", manifest["findings"])
    for key, frame in (("train_raw", train), ("holdout_raw", holdout), ("train_enc", train_enc), ("holdout_enc", holdout_enc)):
        run.write_frame(f"{key}.parquet", frame)
    if test is not None:
        run.write_frame("test_enc.parquet", test_enc)
    shares = train[target].value_counts(normalize=True).round(3).to_dict()
    dropped = {"train": dropped_train, "holdout": dropped_holdout, "test": dropped_test, **({"before_split": data["dropped_missing_target_rows_before_split"]["csv"]} if data["dropped_missing_target_rows_before_split"].get("csv") else {})}
    run.write_state("encoded.json", {"features": features, "target": target, "classes": [str(c) for c in classes], "encoders": encoders, "baseline": baseline, "dropped": dropped, "holdout_row_ids": holdout_row_ids if len(holdout_row_ids) == len(holdout) else list(range(len(holdout))), "train_row_ids": train_row_ids, "class_shares": {str(k): v for k, v in shares.items()}})
    print({"dropped_missing_target_rows": dropped, "unseen_categorical_values": {"holdout": unseen_holdout, "test": unseen_test}, "encoded_categoricals": sorted(encoders)})
    print({"train": len(train_enc), "holdout": len(holdout_enc), "test": 0 if test_enc is None else len(test_enc), "features": len(features), "classes": [str(c) for c in classes], "training_class_shares": shares})
    print("majority-class baseline on the holdout", rounded(baseline))


# --------------------------------------------------------------------------------------------------
# model stages
# --------------------------------------------------------------------------------------------------


def encoded(run: Run, needed_by: str):
    enc = run.read_state("encoded.json", needed_by)
    train = run.read_frame("train_enc.parquet", needed_by)
    holdout = run.read_frame("holdout_enc.parquet", needed_by)
    test = run.read_frame("test_enc.parquet", needed_by) if (run.state / "test_enc.parquet").is_file() else None
    return enc, train, holdout, test


def pretrained(run: Run, n_estimators: int = INFERENCE_ESTIMATORS, random_state: int = SEED):
    P = package(run.root)
    return P.TabICLClassificationPipeline.from_pretrained(weights_dir=run.weights / P.MODEL_KEY, n_estimators=n_estimators, random_state=random_state)


def from_checkpoint(run: Run, checkpoint: Path, like, source: str):
    P = package(run.root)
    return P.TabICLClassificationPipeline(P.create_classifier(model_path=checkpoint, allow_auto_download=False, n_estimators=like.n_estimators, random_state=SEED, device=like.device), model_path=checkpoint, n_estimators=like.n_estimators, random_state=SEED, device=like.device, source=source)


def active_model(run: Run, enc: dict, train):
    """The model Section 6 selected, conditioned on the support rows."""
    cond = run.read_state("condition.json", "active model")
    model = pretrained(run)
    if cond["active_mode"] == "fine-tuned":
        model = from_checkpoint(run, Path(cond["active_checkpoint"]), model, "fine-tuned")
    model.fit(train[enc["features"]], train[enc["target"]])
    return model, cond


def score(P, model, frame, enc):
    X = frame[enc["features"]]
    return P.classification_metrics(frame[enc["target"]].to_numpy(), model.predict(X), model.predict_proba(X), model.classes_)


PREREGISTERED: dict[str, Any] = {
    "data_source": MAGIC_SOURCE,
    "use_byod": False,
    "archive_sha256": MAGIC_SHA256,
    "row_ids_sha256": MAGIC_ROW_IDS_SHA256,
    "run_fine_tuning": True,
    "grid_epochs_cap": 0,
    "grid_learning_rate": 0.0,
    "random_state": SEED,
    "cpu_grid": False,
    "grid": [dict(c) for c in GRID],
    "grid_common": dict(GRID_COMMON),
    "selection_tolerance": SELECTION_TOLERANCE,
    "inference_estimators": INFERENCE_ESTIMATORS,
    "bootstrap": dict(BOOTSTRAP),
    "determinism": dict(DETERMINISM),
}


def current_settings(opts: dict[str, Any], data: dict[str, Any], determinism: dict[str, Any] | None, device: str) -> dict[str, Any]:
    """Every setting the pre-registration fixes, as this run used it (SCORE_ONCE_RULE item 1)."""
    return {
        "data_source": data.get("source"),
        "use_byod": bool(data.get("use_byod", False)),
        "archive_sha256": data.get("archive_sha256"),
        "row_ids_sha256": data.get("row_ids_sha256"),
        "run_fine_tuning": bool(opts.get("run_fine_tuning", True)),
        "grid_epochs_cap": int(opts.get("grid_epochs_cap", 0) or 0),
        "grid_learning_rate": float(opts.get("grid_learning_rate", 0.0) or 0.0),
        "random_state": int(opts.get("random_state", SEED)),
        "cpu_grid": bool(opts.get("cpu_grid", False)),
        "grid": [dict(c) for c in GRID],
        "grid_common": dict(GRID_COMMON),
        "selection_tolerance": SELECTION_TOLERANCE,
        "inference_estimators": INFERENCE_ESTIMATORS,
        "bootstrap": dict(BOOTSTRAP),
        "determinism": determinism,
        "device": device,
    }


def settings_differing(settings: dict[str, Any]) -> list[str]:
    """Names of the settings that differ from the pre-registration; empty means ``preregistered``. Computed from ALL
    fixed settings, including the device (a Tesla T4 is pre-registered)."""
    differing = [key for key, value in PREREGISTERED.items() if settings.get(key) != value]
    if REQUIRED_DEVICE not in str(settings.get("device", "")):
        differing.append("device")
    return differing


def grid_hash(grid: list[dict[str, Any]], selection: str, sdpa_backend: str | None) -> str:
    """Fingerprint of the grid's outcome, printed and reported for information. On a T4 the fp16 grid has given different hashes
    run to run at an identical revision (issue #28), so it never decides a reproduction; within one run it still checks that the
    stored grid is the one the condition stage wrote."""
    def r(value):
        return None if value is None else round(float(value), 6)

    rows = [{"id": g.get("id"), "epochs_run": g.get("epochs_run"), "curve": [r(v) for v in g.get("val_log_loss_per_epoch") or []], "baseline": r(g.get("baseline_val_log_loss")), "candidate_val_log_loss": r((g.get("candidate_val") or {}).get("log_loss")), "delta_best": r(g.get("param_delta_l2_best")), "delta_last": r(g.get("param_delta_l2_last_epoch"))} for g in grid]
    return hashlib.sha256(json.dumps({"grid": rows, "selection": selection, "sdpa_backend": sdpa_backend}, sort_keys=True).encode("utf-8")).hexdigest()


def model_state(pipeline) -> dict[str, Any]:
    """The loaded network's weights of a conditioned pipeline (tabicl deserialises the verified checkpoint on ``fit``)."""
    return {k: v.detach().cpu() for k, v in pipeline.estimator.model_.state_dict().items()}


def param_delta_l2(a: dict[str, Any], b: dict[str, Any]) -> float:
    """L2 norm of the difference over the floating-point tensors both state dicts hold: 0.0 means the weights are identical."""
    total = 0.0
    for key, value in a.items():
        other = b.get(key)
        if other is None or not hasattr(value, "is_floating_point") or not value.is_floating_point():
            continue
        total += float(((other.float() - value.float()) ** 2).sum())
    return math.sqrt(total)


class StepCounter:
    """Counts optimizer steps instead of computing them (R96 round 2, F2; amendment 1).

    ``attempted`` counts the AMP grad scaler's ``step`` calls (one per training batch); ``taken`` counts the optimizer's own
    steps through a step-post hook, which the scaler does not reach when it skips an update whose fp16 gradients overflowed.
    With AMP off (CPU) the scaler is disabled and every attempt is taken."""

    def __init__(self) -> None:
        self.attempted = 0
        self.taken = 0

    def on_optimizer_step(self, *args, **kwargs) -> None:
        self.taken += 1

    def instrument(self, finetuner) -> None:
        make_amp = finetuner._make_amp

        def counted_make_amp(*args, **kwargs):
            use_amp, scaler, ctx = make_amp(*args, **kwargs)
            step = scaler.step

            def counted_step(optimizer, *a, **k):
                self.attempted += 1
                return step(optimizer, *a, **k)

            scaler.step = counted_step
            return use_amp, scaler, ctx

        finetuner._make_amp = counted_make_amp

    def record(self, planned: int) -> dict[str, int]:
        return {"optimizer_steps_planned": planned, "optimizer_steps_attempted": self.attempted, "optimizer_steps_taken": self.taken, "amp_skipped_steps": self.attempted - self.taken}


def run_config(run: Run, P, cfg: dict[str, Any], pipe, enc: dict, train, holdout, *, device: str, seed: int, epochs_cap: int, lr_override: float, base_state: dict) -> dict[str, Any]:
    """One grid config: fine-tune with early stopping on validation log loss, convert tabicl's ``best.ckpt`` (and the last
    epoch's checkpoint) to safetensors right after training, then re-score ``best.safetensors`` at the inference ensemble
    size. Records the validation curve, optimizer steps and whether the weights tabicl kept changed. No pickle survives."""
    import time

    from tabicl._finetune.data import count_chunks

    ft_dir = run.state / "finetune" / cfg["id"]
    shutil.rmtree(ft_dir, ignore_errors=True)
    ft_dir.mkdir(parents=True)
    epochs = min(cfg["epochs"], epochs_cap) if epochs_cap else cfg["epochs"]
    lr = lr_override or cfg["learning_rate"]
    c = GRID_COMMON
    finetuner = P.create_finetuned_classifier(epochs=epochs, learning_rate=lr, weight_decay=c["weight_decay"], grad_clip=c["grad_clip"], amp=device == "cuda", use_lr_scheduler=c["use_lr_scheduler"], n_estimators_finetune=cfg["n_estimators_finetune"], n_estimators_validation=cfg["n_estimators_validation"], n_estimators_inference=c["n_estimators_inference"], max_data_size=c["max_data_size"], finetune_ctx_query_ratio=c["finetune_ctx_query_ratio"], early_stopping=c["early_stopping"], patience=cfg["patience"], min_delta=c["min_delta"], time_limit=cfg["time_limit"], eval_metric=c["eval_metric"], model_path=str(pipe.model_path), allow_auto_download=False, device=device, random_state=seed, verbose=True, support_many_classes=True)
    curve: list[float] = []
    validate = finetuner._validate_current_model

    def recorded(*args, **kwargs):  # baseline, then one validation per epoch (log loss is negated by tabicl)
        result = validate(*args, **kwargs)
        curve.append(-float(result.primary))
        return result

    finetuner._validate_current_model = recorded
    import torch.optim.optimizer as optim_hooks

    counter = StepCounter()
    counter.instrument(finetuner)
    hook = optim_hooks.register_optimizer_step_post_hook(counter.on_optimizer_step)
    t0 = time.perf_counter()
    try:
        P.fine_tune_classifier(finetuner, train[enc["features"]], train[enc["target"]], X_val=holdout[enc["features"]], y_val=holdout[enc["target"]], output_dir=str(ft_dir))
    finally:
        hook.remove()
    seconds = time.perf_counter() - t0
    best_pickle = ft_dir / "best.ckpt"
    if not best_pickle.exists():
        raise RuntimeError(f"Fine-tuning config {cfg['id']} did not produce best.ckpt")
    steps_per_epoch = count_chunks(len(train), c["max_data_size"])
    epochs_run = max(0, len(curve) - 1)
    epoch_ckpts = sorted(ft_dir.glob("epoch*.ckpt"), key=lambda path: int("".join(ch for ch in path.stem if ch.isdigit()) or 0))
    # tabicl writes these with torch.save; each is read once with torch.load(weights_only=True) and replaced by safetensors.
    best_facts = P.convert_finetuned_checkpoint(best_pickle)
    best = Path(best_facts["path"])
    last = Path(P.convert_finetuned_checkpoint(epoch_ckpts[-1])["path"]) if epoch_ckpts else None
    for checkpoint in epoch_ckpts[:-1]:
        checkpoint.unlink()  # TabICL writes one 331 MB checkpoint per epoch; only the last is read out below
    candidate = from_checkpoint(run, best, pipe, "fine-tuned")
    candidate.fit(train[enc["features"]], train[enc["target"]])
    candidate_val = score(P, candidate, holdout, enc)
    delta_best = param_delta_l2(base_state, model_state(candidate))
    last_val = delta_last = None
    if last is not None:
        final = from_checkpoint(run, last, pipe, "fine-tuned (last epoch, exploratory)")
        final.fit(train[enc["features"]], train[enc["target"]])
        last_val = score(P, final, holdout, enc)
        delta_last = param_delta_l2(base_state, model_state(final))
    if last is not None:
        last.unlink()  # keep best.safetensors only
    record = {"id": cfg["id"], "role": cfg["role"], "learning_rate": lr, "lr_schedule": "constant (no warmup, so no zero-LR step)", "epochs_requested": epochs, "epochs_run": epochs_run, "patience": cfg["patience"], "steps_per_epoch": steps_per_epoch, **counter.record(epochs_run * steps_per_epoch), "optimizer_steps": counter.taken, "nonzero_lr_steps": counter.taken, "baseline_val_log_loss": curve[0] if curve else None, "val_log_loss_per_epoch": curve[1:], "weights_changed": delta_best > 0.0, "param_delta_l2_best": delta_best, "param_delta_l2_last_epoch": delta_last, "candidate_val": candidate_val, "last_epoch_val_exploratory": last_val, "best_checkpoint": str(best), "best_checkpoint_sha256": best_facts["converted_sha256"], "best_ckpt_source_sha256": best_facts["source_sha256"], "seconds": round(seconds, 1), "complete": True}
    print({"config": cfg["id"], "role": cfg["role"], "lr": lr, "epochs_run": epochs_run, "optimizer_steps_attempted": record["optimizer_steps_attempted"], "optimizer_steps_taken": record["optimizer_steps_taken"], "amp_skipped_steps": record["amp_skipped_steps"], "baseline_val_log_loss": rounded(record["baseline_val_log_loss"]), "val_log_loss_per_epoch": rounded(curve[1:]), "weights_changed": record["weights_changed"], "param_delta_l2_best": rounded(delta_best, 6), "param_delta_l2_last_epoch": rounded(delta_last, 6), "candidate_val_log_loss": rounded(candidate_val["log_loss"]), "seconds": record["seconds"]})
    if not record["weights_changed"]:
        print(f"  {cfg['id']}: no epoch beat tabicl's baseline by min_delta {c['min_delta']}, so tabicl restored the pretrained weights and wrote them as best.ckpt, converted to best.safetensors (parameter delta 0). The last epoch's weights did move (delta {rounded(delta_last, 6)}); they are only read out on validation below.")
    return record


def select_config(pretrained_val: dict[str, float], grid: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Pre-registered selection on validation log loss: keep the pretrained model unless a config beats it by more than
    SELECTION_TOLERANCE; ties within SELECTION_TIE go to the config with fewer optimizer steps."""
    reference = pretrained_val["log_loss"]
    # A restored config (kept weights == pretrained weights) is the pretrained model, never "fine-tuned" (amendment 1).
    eligible = [g for g in grid if g.get("complete") and g.get("weights_changed") is True and g["candidate_val"]["log_loss"] < reference - SELECTION_TOLERANCE]
    if not eligible:
        return None, f"validation:log_loss; pretrained kept (no config beat {reference:.4f} by more than {SELECTION_TOLERANCE})"
    lowest = min(g["candidate_val"]["log_loss"] for g in eligible)
    tied = [g for g in eligible if g["candidate_val"]["log_loss"] <= lowest + SELECTION_TIE]
    chosen = min(tied, key=lambda g: (g["optimizer_steps"], g["candidate_val"]["log_loss"]))
    return chosen, f"validation:log_loss; {chosen['id']} beat the pretrained model by {reference - chosen['candidate_val']['log_loss']:.4f} > {SELECTION_TOLERANCE}"


def stage_condition(run: Run) -> None:
    import torch

    P = package(run.root)
    data = run.read_state("data.json", "condition")
    enc, train, holdout, test = encoded(run, "condition")
    opts = run.options
    for retired in ("fine_tune_epochs", "eval_metric"):
        if retired in opts:
            raise ValueError(f"{retired} was replaced by the pre-registered grid (docs/magic-finetune-preregistration.md); use GRID_EPOCHS_CAP / GRID_LEARNING_RATE for an exploratory run")
    run_ft = bool(opts.get("run_fine_tuning", True))
    seed = int(opts.get("random_state", SEED))
    epochs_cap = int(opts.get("grid_epochs_cap", 0) or 0)
    lr_override = float(opts.get("grid_learning_rate", 0.0) or 0.0)
    if epochs_cap < 0 or not 0.0 <= lr_override <= 1e-3:
        raise ValueError("GRID_EPOCHS_CAP must be >= 0 and GRID_LEARNING_RATE from 0 (pre-registered) to 1e-3")
    determinism = apply_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = device_name()
    settings = current_settings(opts, data, determinism, name)
    differing = settings_differing(settings)
    print({"device": name, "determinism": determinism, "preregistered": not differing, "settings_differing": differing})
    X_train, y_train = train[enc["features"]], train[enc["target"]]
    pipe = pretrained(run, random_state=seed)
    pipe.fit(X_train, y_train)
    pretrained_holdout = score(P, pipe, holdout, enc)
    test_role = data.get("test_role", "independent test")
    pretrained_test = score(P, pipe, test, enc) if test is not None else None
    print("pretrained validation", rounded(pretrained_holdout))
    if pretrained_test:
        print(f"pretrained {test_role}", rounded(pretrained_test))
    grid: list[dict[str, Any]] = []
    fine_tune: dict[str, Any] = {"requested": run_ft, "ran": False}
    cpu_grid = bool(opts.get("cpu_grid", False))
    if run_ft and device != "cuda" and not cpu_grid:
        fine_tune["skipped"] = "no CUDA device: the pre-registered grid runs on a GPU runtime (a CPU run would take tens of minutes); recorded here, and the run is exploratory"
        print("Fine-tuning skipped:", fine_tune["skipped"], "- choose Runtime > Change runtime type > T4 GPU to run the grid.")
    elif run_ft and len(holdout) < MIN_SELECTION_HOLDOUT_ROWS:
        fine_tune["skipped"] = f"validation partition too small for selection: {len(holdout)} < {MIN_SELECTION_HOLDOUT_ROWS} rows"
        print("Fine-tuning skipped:", fine_tune["skipped"])
    elif run_ft:
        base_state = model_state(pipe)
        for cfg in GRID:
            grid.append(run_config(run, P, cfg, pipe, enc, train, holdout, device=device, seed=seed, epochs_cap=epochs_cap, lr_override=lr_override, base_state=base_state))
        for g in grid:  # the package's strict comparison, recorded beside the pre-registered tolerance rule
            g["strictly_better_than_pretrained"] = bool(P.compare_metric(g["candidate_val"], pretrained_holdout, "log_loss"))
        fine_tune.update(ran=True, configs=[g["id"] for g in grid])
    chosen, selection = select_config(pretrained_holdout, grid) if grid else (None, "default:pretrained (no grid run)")
    active_mode, active_checkpoint = ("fine-tuned", chosen["best_checkpoint"]) if chosen else ("pretrained", str(pipe.bundle_checkpoint))
    for g in grid:  # keep the selected checkpoint only; every config's digest stays in the record
        if not chosen or g["id"] != chosen["id"]:
            Path(g["best_checkpoint"]).unlink(missing_ok=True)
    readout = None
    finals = [g for g in grid if g.get("last_epoch_val_exploratory")]
    if finals:
        best_final = min(finals, key=lambda g: g["last_epoch_val_exploratory"]["log_loss"])
        readout = {"label": "exploratory readout, validation only (never the fresh test)", "config": best_final["id"], "last_epoch_val": best_final["last_epoch_val_exploratory"], "param_delta_l2_last_epoch": best_final["param_delta_l2_last_epoch"], "pretrained_val": pretrained_holdout, "val_log_loss_difference_last_minus_pretrained": best_final["last_epoch_val_exploratory"]["log_loss"] - pretrained_holdout["log_loss"]}
    candidate_holdout = chosen["candidate_val"] if chosen else (min((g["candidate_val"] for g in grid), key=lambda m: m["log_loss"]) if grid else None)
    candidate_test = None
    if chosen and test is not None:
        model = from_checkpoint(run, Path(active_checkpoint), pipe, "fine-tuned")
        model.fit(X_train, y_train)
        candidate_test = score(P, model, test, enc)
    ghash = grid_hash(grid, selection, determinism.get("sdpa_backend"))
    record = {"active_mode": active_mode, "active_checkpoint": active_checkpoint, "active_checkpoint_sha256": sha256_file(Path(active_checkpoint)), "selection": selection, "selection_tolerance": SELECTION_TOLERANCE, "eval_metric": "log_loss", "fine_tune": fine_tune, "grid": grid, "grid_hash": ghash, "exploratory_readout": readout, "pretrained_holdout": pretrained_holdout, "pretrained_test": pretrained_test, "candidate_holdout": candidate_holdout, "candidate_test": candidate_test, "test_role": test_role, "classes": [str(c) for c in pipe.classes_], "device": pipe.device, "device_name": name, "settings": settings, "settings_differing": differing, "preregistered": not differing, "determinism_warnings": determinism_warnings()}
    run.write_state("condition.json", record)
    print({"recommended_for_export": active_mode, "selection_basis": selection, "grid_hash": ghash, "device": name, "classes_": record["classes"]})
    if readout:
        print({"exploratory_readout": rounded(readout)})
    print("grid hash (informational: on a T4 it can differ run to run at an identical revision, issue #28):", ghash)


# --------------------------------------------------------------------------------------------------
# fresh test (score-once, fail-closed; SCORE_ONCE_RULE.md items 1-8)
# --------------------------------------------------------------------------------------------------


def wilson(successes: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def fresh_test_reading(y, p_pretrained, p_selected, classes, *, same_model: bool, exclude=None) -> dict[str, Any]:
    """The pre-registered reading rule on row-level arrays: paired bootstrap over rows of the log-loss difference
    (selected - pretrained), the verdict, and the secondary accuracy readouts."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(str)
    labels = [str(c) for c in classes]
    index = np.array([labels.index(v) for v in y])
    pa = np.asarray(p_pretrained, dtype=float)
    pb = np.asarray(p_selected, dtype=float)
    keep = np.ones(len(y), dtype=bool) if exclude is None else ~np.asarray(exclude, dtype=bool)
    y, index, pa, pb = y[keep], index[keep], pa[keep], pb[keep]
    n = len(y)
    rows = np.arange(n)
    ll_a = -np.log(np.clip(pa[rows, index], 1e-15, 1.0))
    ll_b = -np.log(np.clip(pb[rows, index], 1e-15, 1.0))
    ok_a = (pa.argmax(axis=1) == index).astype(float)
    ok_b = (pb.argmax(axis=1) == index).astype(float)
    d_ll, d_acc = ll_b - ll_a, ok_b - ok_a
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    boot_ll, boot_acc = [], []
    for start in range(0, BOOTSTRAP["n_boot"], 1000):
        draws = rng.integers(0, n, size=(min(1000, BOOTSTRAP["n_boot"] - start), n))
        boot_ll.append(d_ll[draws].mean(axis=1))
        boot_acc.append(d_acc[draws].mean(axis=1))
    boot_ll, boot_acc = np.concatenate(boot_ll), np.concatenate(boot_acc)
    delta = float(d_ll.mean())
    ci = [float(np.percentile(boot_ll, 2.5)), float(np.percentile(boot_ll, 97.5))]
    if same_model:
        verdict = "same model (Δ = 0)"
    elif ci[1] < 0 and delta <= -PRACTICAL_DELTA:
        verdict = "improved"
    elif ci[0] > 0:
        verdict = "worse"
    else:
        verdict = "no detectable difference"

    def auc(p):
        try:
            return float(roc_auc_score(index, p[:, 1])) if len(labels) == 2 else float(roc_auc_score(index, p, multi_class="ovr"))
        except ValueError:
            return float("nan")

    return {"n": n, "primary": {"endpoint": "fresh-test log loss difference, selected - pretrained", "delta": delta, "ci95": ci, "verdict": verdict, "practical_delta": PRACTICAL_DELTA, "bootstrap": dict(BOOTSTRAP)}, "log_loss": {"pretrained": float(ll_a.mean()), "selected": float(ll_b.mean())}, "accuracy": {"pretrained": float(ok_a.mean()), "pretrained_wilson95": wilson(int(ok_a.sum()), n), "selected": float(ok_b.mean()), "selected_wilson95": wilson(int(ok_b.sum()), n), "difference": float(d_acc.mean()), "difference_ci95": [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))]}, "roc_auc": {"pretrained": auc(pa), "selected": auc(pb)}}


def fresh_test_probabilities(run: Run, P, cond: dict, test):
    """(y, p_pretrained, p_selected, classes, max_row_sum_deviation) on the fresh test, both models conditioned on the
    same support rows at the inference ensemble size."""
    import numpy as np

    enc, train, _holdout, _ = encoded(run, "fresh_test")
    features, target = enc["features"], enc["target"]
    test = P.align_to_schema(test, features, target, enc["classes"])
    test, _ = P.apply_categorical_encoder(test, enc["encoders"])
    pre = pretrained(run)
    pre.fit(train[features], train[target])
    raw = np.asarray(pre.estimator.predict_proba(test[features]), dtype=float)
    p_pre, deviation = P.renormalise_probabilities(raw)
    if cond["active_mode"] == "pretrained":
        p_sel = p_pre
    else:
        sel = from_checkpoint(run, Path(cond["active_checkpoint"]), pre, "fine-tuned")
        sel.fit(train[features], train[target])
        p_sel = sel.predict_proba(test[features])
    return test[target].to_numpy(), p_pre, p_sel, [str(c) for c in pre.classes_], deviation


def read_ledger(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """(entries, corrupt). A corrupt or malformed ledger is set aside, never trusted and never a crash."""
    if not path.is_file():
        return [], False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("entries"), list):
            return value["entries"], False
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    n = 1
    while path.with_name(f"ledger.corrupt-{n}.json").exists():
        n += 1
    os.replace(path, path.with_name(f"ledger.corrupt-{n}.json"))
    print(f"The fresh-test ledger was unreadable and was set aside as ledger.corrupt-{n}.json; this scoring is labelled a reproduction, never the first evaluation.")
    return [], True


def look_marker(run: Run) -> Path:
    """Marker of a pre-registered look, beside the per-run directories (outside outputs/ and outside any one run)."""
    return run.root.parent / f"{STEM}_fresh_test_look_{MAGIC_SHA256[:16]}.json"


def next_index(folder: Path, kind: str) -> int:
    return len(list(folder.glob(f"{STEM}_fresh_test_{kind}_*.json"))) + 1


def stage_fresh_test(run: Run) -> None:
    import pandas as pd

    masked = fresh_test_masked()
    print({"fresh_test_mask": f"{MASK_ENV}=1" if masked else f"{MASK_ENV} unset"})
    if masked:
        print("Fresh test masked: it is not read, scored or recorded in this run (not a look).")
        return
    P = package(run.root)
    data = run.read_state("data.json", "fresh_test")
    if data["source"] != MAGIC_SOURCE:
        print(f"No fresh test for {data['source']!r}: the fresh test belongs to the pre-registered MAGIC sample. Section 7 showed this source's {data.get('test_role', 'test')}.")
        return
    cond = run.read_state("condition.json", "fresh_test")
    fresh = run.out / FRESH_DIR
    # Reset and recompute: nothing cached by an earlier stage is trusted.
    differing = settings_differing(cond.get("settings") or {})
    grid = cond.get("grid") or []
    checks = {"grid_complete": [g.get("id") for g in grid] == [c["id"] for c in GRID] and all(g.get("complete") for g in grid)}
    sdpa = ((cond.get("settings") or {}).get("determinism") or {}).get("sdpa_backend")
    checks["grid_hash_matches"] = checks["grid_complete"] and grid_hash(grid, cond.get("selection", ""), sdpa) == cond.get("grid_hash")
    active = Path(cond.get("active_checkpoint", ""))
    checks["selected_model_matches"] = active.is_file() and sha256_file(active) == cond.get("active_checkpoint_sha256")
    cache = run.root / "inputs" / "magic-gamma.zip"
    payload = cache.read_bytes() if cache.is_file() else b""
    checks["fresh_pool_digest"] = hashlib.sha256(payload).hexdigest() == MAGIC_SHA256
    validation = {"pretrained": cond.get("pretrained_holdout"), "selected_config_candidate": cond.get("candidate_holdout"), "selection": cond.get("selection")}
    if differing:
        record = {"test_look": TEST_LOOKS[2], "settings_differing": differing, "preregistered": False, "validation_only": validation, "exploratory_readout": cond.get("exploratory_readout"), "note": "This run differs from the pre-registration, so the fresh test was not read. Only the pre-registered settings on a Tesla T4 score it; see docs/release-verification.md for the recorded evaluation and its reproductions."}
        out = atomic_write_json(fresh / f"{STEM}_fresh_test_exploratory_{next_index(fresh, 'exploratory')}.json", record)
        print({"test_look": TEST_LOOKS[2], "settings_differing": differing, "written": f"outputs/{FRESH_DIR}/{out.name}"})
        return
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"fresh test not scored: pre-scoring check(s) failed {failed}; this is not a look. Re-run the notebook from Section 4.")
    test_bytes = magic_member(payload, "test.csv")  # digest-checked; parsed only below
    ledger_path = fresh / "ledger.json"
    entries, corrupt = read_ledger(ledger_path)
    first_report = fresh / f"{STEM}_fresh_test_report.json"
    marker = look_marker(run)
    first = RECORDED_EVALUATION is None and not corrupt and not entries and not first_report.exists() and not marker.exists()
    label = TEST_LOOKS[0] if first else TEST_LOOKS[1]
    test = P.read_csv_payload(test_bytes, "test.csv")
    keys = {}
    for member in ("train.csv", "val.csv", "test.csv"):
        frame = pd.read_csv(io.BytesIO(test_bytes if member == "test.csv" else magic_member(payload, member)), dtype=str, keep_default_na=False)
        keys[member] = frame.drop(columns=["target"]).agg(",".join, axis=1)
    duplicated = keys["test.csv"].isin(set(keys["train.csv"]) | set(keys["val.csv"])).to_numpy()
    y, p_pre, p_sel, classes, deviation = fresh_test_probabilities(run, P, cond, test)
    same = cond["active_mode"] == "pretrained"
    reading = fresh_test_reading(y, p_pre, p_sel, classes, same_model=same)
    sensitivity = fresh_test_reading(y, p_pre, p_sel, classes, same_model=same, exclude=duplicated)
    sens = {"excluded_rows": int(duplicated.sum()), **{k: sensitivity[k] for k in ("n", "primary")}}
    comparison = compare_with_record(cond["active_mode"], cond["grid_hash"], cond.get("device_name"), printed_reading(reading, sens)) if label == TEST_LOOKS[1] else None
    report = {"test_look": label, "settings_differing": [], "preregistered": True, "recorded_evaluation": RECORDED_EVALUATION, "checks": checks, "selected": {"mode": cond["active_mode"], "selection": cond["selection"]}, "grid_hash": cond["grid_hash"], "device": cond.get("device_name"), "determinism": (cond.get("settings") or {}).get("determinism"), "fresh_test": {"rows": len(y), "member_sha256": MAGIC_MEMBERS["test.csv"], "row_ids_sha256": MAGIC_ROW_IDS_SHA256["test"], "max_probability_row_sum_deviation": deviation}, **reading, "sensitivity_without_duplicated_rows": sens, "comparison_with_record": comparison, "power": "planning estimates: a single accuracy on 1,500 rows about ±0.017 (Wilson); a paired accuracy difference about ±0.011 at 5 % discordance; a paired log-loss difference about ±0.017 (pessimistic proxy)", "validation": validation, "exploratory_readout": cond.get("exploratory_readout"), "notebook_source": json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}}
    out = first_report if first else fresh / f"{STEM}_fresh_test_reproduction_{next_index(fresh, 'reproduction')}.json"
    atomic_write_json(out, report)  # the look counts from here: the report is written
    if not marker.exists():  # outside outputs/: deleting outputs/ cannot make a later look claim to be the first
        atomic_write_json(marker, {"first_look": out.name, "archive_sha256": MAGIC_SHA256, "grid_hash": cond["grid_hash"]})
    entries.append({"file": out.name, "test_look": label, "device": cond.get("device_name"), "grid_hash": cond["grid_hash"], "delta": reading["primary"]["delta"]})
    atomic_write_json(ledger_path, {"entries": entries})
    print({"test_look": label, "written": f"outputs/{FRESH_DIR}/{out.name}", "selected": cond["active_mode"]})
    print(rounded({"primary": reading["primary"], "accuracy": reading["accuracy"], "roc_auc": reading["roc_auc"], "sensitivity_without_duplicated_rows": report["sensitivity_without_duplicated_rows"]}))
    if comparison is not None:
        print(comparison)
    print("Scored only under the pre-registered settings; see docs/release-verification.md for the recorded evaluation and its reproductions.")


def stage_report(run: Run) -> None:
    import time

    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier

    P = package(run.root)
    data = run.read_state("data.json", "report")
    enc, train, holdout, test = encoded(run, "report")
    model, cond = active_model(run, enc, train)
    classes = list(model.classes_)
    features, target = enc["features"], enc["target"]
    X_train, y_train = train[features], train[target]
    lgbm = LGBMClassifier(random_state=SEED, n_estimators=100, verbose=-1).fit(X_train, y_train)
    rf = RandomForestClassifier(random_state=SEED, n_estimators=100).fit(X_train, y_train)

    def timed(m, frame, repeats=5):
        X = frame[features]
        if repeats > 1:
            m.predict_proba(X)  # discarded warm-up call
        latencies = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            raw = np.asarray(m.predict_proba(X), dtype=float)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        proba = P.align_probabilities(raw, list(m.classes_), classes)
        return proba, np.asarray(classes, dtype=object)[np.argmax(proba, axis=1)], float(np.median(latencies))

    y = holdout[target].to_numpy()
    n = len(holdout)
    p_tab, pred_tab, lat_tab = timed(model, holdout)
    p_lgbm, pred_lgbm, lat_lgbm = timed(lgbm, holdout)
    p_rf, pred_rf, lat_rf = timed(rf, holdout)
    classical = {"tabicl_" + cond["active_mode"]: P.classification_metrics(y, pred_tab, p_tab, classes), "lightgbm": P.classification_metrics(y, pred_lgbm, p_lgbm, classes), "random_forest": P.classification_metrics(y, pred_rf, p_rf, classes)}
    errors = {name: int((pred != y).sum()) for name, pred in (("tabicl_" + cond["active_mode"], pred_tab), ("lightgbm", pred_lgbm), ("random_forest", pred_rf))}
    print({"holdout_rows": n, "one_row_resolution_pct": round(100.0 / n, 2)})
    for name, values in classical.items():
        print(f"{name:<20} accuracy={values['accuracy']:.4f} errors={errors[name]} roc_auc={values['roc_auc']:.4f} log_loss={values['log_loss']:.4f}")
    print({"latency_ms_median_of_5": {"tabicl": round(lat_tab, 2), "lightgbm": round(lat_lgbm, 2), "random_forest": round(lat_rf, 2)}, "rows": n, "device": model.device})
    best_acc, tied = -1.0, []
    for w in np.linspace(0.0, 1.0, 101):
        acc = float(np.mean(np.asarray(classes, dtype=object)[np.argmax(w * p_tab + (1.0 - w) * p_lgbm, axis=1)] == y))
        if acc > best_acc + 1e-9:
            best_acc, tied = acc, [float(w)]
        elif abs(acc - best_acc) <= 1e-9:
            tied.append(float(w))
    best_w = tied[len(tied) // 2]  # upper-median tied weight: even ties break toward the foundation model
    p_blend = best_w * p_tab + (1.0 - best_w) * p_lgbm
    blend = P.classification_metrics(y, np.asarray(classes, dtype=object)[np.argmax(p_blend, axis=1)], p_blend, classes)
    blend_record = {"blend_objective": "maximise holdout accuracy", "weight_tabicl": best_w, "tied_weights": len(tied), "tied_weight_range": [min(tied), max(tied)], "blend_holdout": blend, "accuracy_delta_vs_tabicl": round(blend["accuracy"] - classical["tabicl_" + cond["active_mode"]]["accuracy"], 4)}
    print(rounded(blend_record))
    if len(tied) > 1:
        print(f"Holdout accuracy is the same for every blend weight from {min(tied):.2f} to {max(tied):.2f} ({len(tied)} grid points); the selected weight {best_w:.2f} is not uniquely informative.")
    if test is not None:
        y_test = test[target].to_numpy()
        pt, predt, _ = timed(model, test, repeats=1)
        pl, predl, _ = timed(lgbm, test, repeats=1)
        pb = best_w * pt + (1.0 - best_w) * pl
        tab_test = P.classification_metrics(y_test, predt, pt, classes)
        blend_test = P.classification_metrics(y_test, np.asarray(classes, dtype=object)[np.argmax(pb, axis=1)], pb, classes)
        print({data.get("test_role", "independent test"): rounded({"tabicl": tab_test, "lightgbm": P.classification_metrics(y_test, predl, pl, classes), "blend": blend_test})})
        if blend["accuracy"] > classical["tabicl_" + cond["active_mode"]]["accuracy"] and blend_test["accuracy"] <= tab_test["accuracy"]:
            print("WARNING mixed evidence: the blend improved holdout accuracy but not independent-test accuracy; treat the holdout gain as selection-biased.")
    else:
        if data["source"] == MAGIC_SOURCE:
            print("The MAGIC fresh test is not read here: the fresh_test stage (Section 7b) scores it, only under the pre-registered settings.")
        print("The validation blend score is selection-biased demonstration evidence (the weight was chosen on validation).")
    active_holdout = cond["candidate_holdout"] if cond["active_mode"] == "fine-tuned" else cond["pretrained_holdout"]
    active_test = cond["candidate_test"] if cond["active_mode"] == "fine-tuned" else cond["pretrained_test"]
    report = P.evaluation_report(active_holdout, baseline=enc["baseline"], independent_test=active_test, n_holdout=n, n_test=None if test is None else len(test), class_labels=classes, target_column=target, selection=cond["selection"], sample_kind=data["sample_kind"], estimation="single seeded stratified split (support/holdout/independent test); no dispersion estimate")
    report["classical_baselines_holdout"] = {k: v for k, v in classical.items() if not k.startswith("tabicl_")}
    report["holdout_errors"] = errors
    report["blend_holdout"] = {"weight_tabicl": best_w, "tied_weight_range": [min(tied), max(tied)], **blend}
    report["fine_tune"] = cond["fine_tune"]
    report["fine_tune_grid"] = cond.get("grid")
    report["grid_hash"] = cond.get("grid_hash")
    report["selection_tolerance"] = cond.get("selection_tolerance")
    report["exploratory_readout"] = cond.get("exploratory_readout")
    report["test_role"] = data.get("test_role")
    report["fresh_test"] = {"scored_by": "fresh_test stage", "preregistered": cond.get("preregistered"), "settings_differing": cond.get("settings_differing"), "files": f"outputs/{FRESH_DIR}/"}
    report["interpretation"] = f"One validation row is {1.0 / n:.4f} of accuracy ({n} rows). Log loss is the pre-registered primary metric because it registers changes accuracy cannot; no dispersion over splits was measured."
    run.write_output(f"{STEM}_evaluation_report.json", report)
    print(json.dumps({k: report[k] for k in ("verdict", "reason", "decision_rule", "selection", "n_holdout", "n_test", "interpretation")}, indent=2))


def runtime_versions(device: str) -> dict[str, Any]:
    import importlib.metadata
    import platform

    import numpy
    import pandas
    import sklearn
    import torch

    return {"python": platform.python_version(), "torch": torch.__version__, "tabicl": importlib.metadata.version("tabicl"), "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__, "device": device}


def new_rows_frame(run: Run, enc: dict):
    """The first eight holdout rows (raw features, unlabelled) with their original row id: the default new-data input
    (TIC-S2)."""
    raw = run.read_frame("holdout_raw.parquet", "new rows").reset_index(drop=True)
    rows = raw[enc["features"]].head(NEW_ROWS).copy()
    rows.insert(0, "row_id", enc["holdout_row_ids"][: len(rows)])
    return rows


def with_predictions(P, model, rows, enc, classes):
    X, unseen = P.apply_categorical_encoder(rows[enc["features"]], enc["encoders"])
    out = rows.copy()
    proba = model.predict_proba(X)
    out["prediction"] = model.predict(X)
    for index, label in enumerate(classes):
        out[f"probability_{label}"] = proba[:, index]
    return out, unseen


def stage_predict(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "predict")
    enc, train, holdout, test = encoded(run, "predict")
    model, cond = active_model(run, enc, train)
    classes = list(model.classes_)
    new_data = None
    path = run.options.get("new_data_path") or ""
    if path:
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"NEW_DATA_PATH {path!r} does not exist in this runtime.")
        rows = P.read_inference_csv(file.read_bytes(), enc["features"])
        manifest = P.validate_inputs(rows, None, feature_columns=enc["features"], names=[file.name])
        out, unseen = with_predictions(P, model, rows, enc, classes)
        new_data = {"input": file.name, "rows": len(out), "unseen_categorical_values": unseen, "input_manifest": manifest}
    else:
        rows = new_rows_frame(run, enc)
        out, unseen = with_predictions(P, model, rows, enc, classes)
        rows.to_csv(run.out / f"{STEM}_new_rows.csv", index=False)
        print("No new-data file: the first eight holdout rows (already scored in Section 7) are scored instead; their unlabelled copy is written to", f"outputs/{STEM}_new_rows.csv")
    out.to_csv(run.out / f"{STEM}_predictions.csv", index=False)
    report = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    input_manifest = json.loads((run.out / f"{STEM}_input_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    active_holdout = cond["candidate_holdout"] if cond["active_mode"] == "fine-tuned" else cond["pretrained_holdout"]
    active_test = cond["candidate_test"] if cond["active_mode"] == "fine-tuned" else cond["pretrained_test"]
    payload = {
        "predictions": out.to_dict(orient="records"),
        "new_data": new_data,
        "metrics": {"active_mode": cond["active_mode"], "holdout": active_holdout, "independent_test": active_test, "pretrained_holdout": cond["pretrained_holdout"], "candidate_holdout": cond["candidate_holdout"]},
        "fine_tune": cond["fine_tune"],
        "majority_class_baseline": enc["baseline"],
        "evaluation_report": report,
        "input_manifest": input_manifest,
        "inference": {"decisionRule": P.DECISION_RULE + " over uncalibrated class probabilities", "threshold": None, "classLabels": [str(c) for c in classes]},
        "sample": {"kind": data["sample_kind"], "name": data["name"], "source": data["source"], "csv_sha256": data["csv_sha256"], "train_rows": len(train), "holdout_rows": len(holdout), "test_rows": 0 if test is None else len(test), "training_class_shares": enc["class_shares"]},
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "runtime": runtime_versions(model.device),
    }
    run.write_output(f"{STEM}_result.json", payload)
    if unseen:
        print("unseen categorical values mapped to the unknown code:", unseen)
    print(out.round(4).to_string(index=False))


def build_manifest(P, *, enc: dict, classes: list, mode: str, selection: str, metrics: dict | None, n_estimators: int, checkpoint: Path | None, context: Path, runtime: dict, source: dict, checkpoint_facts: tuple[str, int] | None = None) -> dict[str, Any]:
    """The bundle manifest written by the export stage."""
    return {
        "artifactFormat": P.ARTIFACT_FORMAT,
        "checkpoint": P.BUNDLE_CHECKPOINT,
        "trainingContext": "training_context.parquet",
        "targetColumn": enc["target"],
        "featureColumns": enc["features"],
        "classLabels": [str(c) for c in classes],
        "baseCheckpoint": P.BASE_CHECKPOINT_NAME,
        "baseModelRevision": P.MODEL_REVISION,
        "baseModelSha256": P.BASE_MODEL_SHA256,
        "tabiclVersion": "2.1.1",
        "mode": mode,
        "selectionBasis": selection,
        "metrics": metrics,
        "inference": {"class": "TabICLClassifier", "weightsFormat": "safetensors (config in the string metadata; load with create_classifier)", "modelPath": P.BUNDLE_CHECKPOINT, "nEstimators": n_estimators, "randomState": SEED, "supportManyClasses": True, "allowAutoDownload": False, "decisionRule": P.DECISION_RULE, "categoricalEncoders": enc["encoders"]},
        "digests": {"checkpointSha256": checkpoint_facts[0] if checkpoint_facts else sha256_file(checkpoint), "trainingContextSha256": sha256_file(context)},
        "sizes": {"checkpoint": checkpoint_facts[1] if checkpoint_facts else checkpoint.stat().st_size, "trainingContext": context.stat().st_size},
        "payloadFiles": [P.BUNDLE_CHECKPOINT, "training_context.parquet"],
        "runtime": runtime,
        "notebookSource": source,
    }


ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)  # the earliest ZIP timestamp; fixed so equal members give an equal ZIP digest


def deterministic_zip(root: Path, archive: Path) -> Path:
    """Write every file under ``root`` to ``archive`` in sorted order with a fixed timestamp and mode, atomically.

    ``shutil.make_archive`` stamps members with their file-system mtimes, so two runs that write identical members still
    print different ZIP digests. Here the digest depends only on the member names and bytes.
    """
    import zipfile

    files = sorted(p for p in root.rglob("*") if p.is_file())
    tmp = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with path.open("rb") as source, zf.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
    os.replace(tmp, archive)
    return archive


def stage_export(run: Run) -> None:
    import importlib.metadata
    import platform

    import numpy
    import pandas
    import sklearn
    import torch

    P = package(run.root)
    enc, train, holdout, _test = encoded(run, "export")
    model, cond = active_model(run, enc, train)
    classes = list(model.classes_)
    art = run.out / "artifact"
    shutil.rmtree(art, ignore_errors=True)
    (art / "checkpoints").mkdir(parents=True)
    ckpt = art / P.BUNDLE_CHECKPOINT  # a converted fine-tune or, for "pretrained", the converted base (CONVERTED_MODEL_SHA256)
    if Path(cond["active_checkpoint"]).suffix != ".safetensors":
        raise RuntimeError(f"the selected checkpoint {cond['active_checkpoint']} is not a safetensors file; re-run Section 6")
    shutil.copy2(cond["active_checkpoint"], ckpt)
    context = art / "training_context.parquet"
    train[enc["features"] + [enc["target"]]].to_parquet(context, index=False)
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    runtime = {"pythonVersion": platform.python_version(), "torchVersion": torch.__version__, "pandasVersion": pandas.__version__, "pyarrowVersion": importlib.metadata.version("pyarrow"), "scikitLearnVersion": sklearn.__version__, "device": model.device}
    metrics = {"selectionMetric": cond["eval_metric"], "pretrainedHoldout": cond["pretrained_holdout"], "fineTunedHoldout": cond["candidate_holdout"], "pretrainedIndependentTest": cond["pretrained_test"], "fineTunedIndependentTest": cond["candidate_test"]}
    manifest = build_manifest(P, enc=enc, classes=classes, mode=cond["active_mode"], selection=cond["selection"], metrics=metrics, n_estimators=model.n_estimators, checkpoint=ckpt, context=context, runtime=runtime, source=source)
    (art / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive = deterministic_zip(art, run.out / f"{STEM}_artifact.zip")
    smoke = holdout[enc["features"]].iloc[: min(NEW_ROWS, len(holdout))]
    run.write_state("export_reference.json", {"labels": [str(x) for x in model.predict(smoke)], "proba": numpy.asarray(model.predict_proba(smoke)).tolist(), "classes": [str(c) for c in classes], "device": model.device})
    digest = sha256_file(archive)
    result_path = run.out / f"{STEM}_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["artifact"] = {"zip": archive.name, "zip_sha256": digest, "manifest": manifest}
        run.write_output(result_path.name, result)
    record = {"artifact_zip": str(archive), "zip_sha256": digest, "mode": cond["active_mode"], "checkpoint_bytes": ckpt.stat().st_size, "training_context_sha256": manifest["digests"]["trainingContextSha256"]}
    print(record)
    print(f"Trusted digest of this export: whoever loads {archive.name} elsewhere checks it against '{digest}' (EXPECTED_ZIP_SHA256 in Section 10). Section 11 itself never loads this run's export; it loads the pinned sample artifact.")


def stage_reload(run: Run) -> None:
    import numpy as np
    import pandas as pd

    P = package(run.root)
    enc, _train, holdout, _test = encoded(run, "reload")
    reference = run.read_state("export_reference.json", "reload")
    archive = run.out / f"{STEM}_artifact.zip"
    reload_dir = run.out / "artifact-reload"
    shutil.rmtree(reload_dir, ignore_errors=True)
    root = P.safe_extract_zip(archive, reload_dir)
    served = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    members = P.verify_artifact_bundle(root, served)
    context = pd.read_parquet(members["training_context"])
    model = P.TabICLClassificationPipeline(P.create_classifier(model_path=members["checkpoint"], allow_auto_download=False, n_estimators=served["inference"]["nEstimators"], random_state=served["inference"]["randomState"], device=reference["device"]), model_path=members["checkpoint"], n_estimators=served["inference"]["nEstimators"], random_state=served["inference"]["randomState"], device=reference["device"], source="artifact")
    model.fit(context[served["featureColumns"]], context[served["targetColumn"]])
    if [str(c) for c in model.classes_] != reference["classes"]:
        raise RuntimeError("Reloaded class labels differ from the exported class order")
    smoke = holdout[enc["features"]].iloc[: min(NEW_ROWS, len(holdout))]
    if [str(x) for x in model.predict(smoke)] != reference["labels"]:
        raise RuntimeError("Prediction mismatch: the classifier rebuilt from the bundle predicts different labels")
    proba = np.asarray(model.predict_proba(smoke))
    np.testing.assert_allclose(np.asarray(reference["proba"]), proba, rtol=RELOAD_RTOL, atol=RELOAD_ATOL)
    print({"reload_dir": str(reload_dir), "process": "fresh (separate from the exporting stage)", "labels_identical": True, "max_abs_probability_difference": float(np.max(np.abs(np.asarray(reference["proba"]) - proba)))})
    print(f"PASS: bundle extracted safely, digests verified, classifier rebuilt from the bundle alone; labels identical and probabilities equivalent (rtol={RELOAD_RTOL}, atol={RELOAD_ATOL}).")


def stage_activity(run: Run) -> None:
    """Optional activity: run the pretrained model on the three-class Wine sample and compare the one-row resolution,
    accuracy and one-vs-rest ROC-AUC with the canonical run; writes only to outputs/activity/."""
    P = package(run.root)
    source = run.options.get("data_source", "Sample: Wine")
    if source not in SAMPLE_SOURCES:
        raise ValueError(f"ACTIVITY_DATA_SOURCE must be one of {list(SAMPLE_SOURCES)}")
    before = {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()}
    train, holdout, _test, name, target = load_sample(P, source)
    train, _ = P.prepare_classification_table(train, target)
    features = [c for c in train.columns if c != target]
    classes = sorted(train[target].unique(), key=str)
    holdout = P.align_to_schema(holdout, features, target, classes)
    encoders = P.fit_categorical_encoder(train, features)
    train_enc, _ = P.apply_categorical_encoder(train, encoders)
    holdout_enc, _ = P.apply_categorical_encoder(holdout, encoders)
    enc = {"features": features, "target": target}
    model = pretrained(run)
    model.fit(train_enc[features], train_enc[target])
    metrics = score(P, model, holdout_enc, enc)
    baseline = P.majority_class_baseline(train_enc[target], holdout_enc[target])
    canonical = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    record = {"activity_sample": name, "classes": [str(c) for c in classes], "holdout_rows": len(holdout_enc), "one_row_resolution": 1.0 / len(holdout_enc), "tabicl_pretrained_holdout": metrics, "majority_class_baseline": baseline, "canonical": {"n_holdout": canonical["n_holdout"], "one_row_resolution": 1.0 / canonical["n_holdout"], "metrics": {m["id"]: m["value"] for m in canonical["metrics"]}}}
    run.write_output(f"activity/{STEM}_activity_{name}.json", record)
    after = {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()}
    if before != after:
        raise RuntimeError("The activity changed a canonical output; it must write only to outputs/activity/.")
    print(rounded(record))
    print({"canonical_outputs_unchanged": True})


# --- External-artifact section (§19.1): stages external_artifact → external_reconstruct → external_predict ---------------
# These stages read only the carried files, the run directory's own `inputs/` cache and their own state; they never read
# a file that the data, validate, condition, report, fresh_test, predict, export or reload stages wrote (XAS5).


def check_trusted_digest(path: Path, expected: str, field: str) -> str:
    """Refuse a file whose SHA-256 is not ``expected``; nothing is extracted or deserialised before this passes."""
    expected = str(expected).strip().lower()
    if not SHA256_HEX.match(expected):
        raise ValueError(f"{field} must be 64 hexadecimal characters (the digest the producer published with the artifact), got {expected!r}.")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"Trusted digest mismatch for {path.name}: {field} is {expected}, the file's SHA-256 is {observed}. This is not the artifact the digest was issued for; nothing was extracted. Obtain the artifact and its digest from the producer again.")
    return observed


def fetch_pinned_artifact(cache: Path, url: str, expected: str) -> Path:
    """The pinned sample artifact: reused from ``cache`` when its digest matches, else downloaded (no authentication),
    size-capped, and accepted only at the pinned SHA-256. The digest is checked before anything opens the file."""
    if cache.is_file() and sha256_file(cache) == expected:
        return cache
    import urllib.request

    cache.parent.mkdir(parents=True, exist_ok=True)
    part = cache.with_name(cache.name + ".part")
    size = 0
    with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > SAMPLE_ARTIFACT_MAX_BYTES:
                out.close()
                part.unlink(missing_ok=True)
                raise ValueError(f"The pinned sample artifact at {url} exceeds {SAMPLE_ARTIFACT_MAX_BYTES} bytes: refused before hashing or extraction.")
            out.write(chunk)
    try:
        check_trusted_digest(part, expected, "the pinned SAMPLE_ARTIFACT_SHA256")
    except ValueError:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, cache)
    return cache


def check_bundle_identity(P, manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest must declare the pinned base checkpoint; a ``pretrained`` bundle's checkpoint must BE the pinned base:
    in a v2 bundle the converted base (``digests.checkpointSha256 == CONVERTED_MODEL_SHA256``), in a deprecated v1 bundle
    the source checkpoint (``== BASE_MODEL_SHA256``), so a swapped checkpoint behind a re-hashed manifest is refused
    (TICA-m2). A ``fine-tuned`` checkpoint is a derivative bound only by the bundle's own digests and the trusted ZIP digest."""
    declared = (manifest.get("baseCheckpoint"), manifest.get("baseModelRevision"), manifest.get("baseModelSha256"))
    if declared != (P.BASE_CHECKPOINT_NAME, P.MODEL_REVISION, P.BASE_MODEL_SHA256):
        raise RuntimeError(f"Artifact was not produced on the pinned base checkpoint carried by this notebook: declares {declared}, expected {(P.BASE_CHECKPOINT_NAME, P.MODEL_REVISION, P.BASE_MODEL_SHA256)}.")
    mode = manifest.get("mode")
    checkpoint_sha = (manifest.get("digests") or {}).get("checkpointSha256")
    if mode == "pretrained":
        legacy = manifest.get("artifactFormat") == P.LEGACY_ARTIFACT_FORMAT
        expected = P.BASE_MODEL_SHA256 if legacy else P.CONVERTED_MODEL_SHA256
        if checkpoint_sha != expected:
            raise RuntimeError(f"The artifact declares mode 'pretrained', so its checkpoint must be the pinned base with SHA-256 {expected}; its manifest records {checkpoint_sha}. Refusing to load a different checkpoint.")
        what = "the pinned base checkpoint (deprecated v1 pickle layout)" if legacy else "the pinned safetensors conversion of the base"
        return {"mode": mode, "checkpoint_binding": f"identical to {what} (SHA-256 checked)"}
    if mode == "fine-tuned":
        return {"mode": mode, "checkpoint_binding": "fine-tuned derivative of the pinned base; bound by the manifest digest and the trusted ZIP digest"}
    raise RuntimeError(f"Unsupported artifact mode {mode!r}: expected 'pretrained' or 'fine-tuned'.")


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


def magic_contract(run: Run, P) -> dict[str, Any]:
    """The data contract of the current default path, read from the pinned MAGIC archive (never ``test.csv``)."""
    payload = fetch_magic(run.root / "inputs" / "magic-gamma.zip")
    train = P.read_csv_payload(magic_member(payload, "train.csv"), "train.csv")
    features = [c for c in train.columns if c != "target"]
    return {"task": "binary classification", "featureColumns": features, "targetColumn": "target", "classLabels": sorted(str(c) for c in train["target"].unique()), "archive_sha256": MAGIC_SHA256}


def check_data_contract(manifest: dict[str, Any], contract: dict[str, Any]) -> None:
    """SART10: the artifact must follow the current default path's features, target and labels, or it is refused."""
    for key in ("featureColumns", "targetColumn", "classLabels"):
        if manifest.get(key) != contract[key]:
            raise RuntimeError(f"The artifact does not follow this notebook's default data contract (SART10): its {key} is {manifest.get(key)!r}, the MAGIC default path's is {contract[key]!r}. A sample artifact built from another dataset or an earlier default is refused even when its digest verifies.")


def external_out(run: Run) -> Path:
    return run.out / EXTERNAL_DIR


def stage_external_artifact(run: Run) -> None:
    import importlib.metadata

    import pandas as pd
    import torch

    P = package(run.root)
    opts = run.options
    source = opts.get("source", "pinned")
    shutil.rmtree(external_out(run), ignore_errors=True)  # a failed check leaves no earlier success record behind (VER6)
    for name in ("external_artifact.json", "external_rows.json", "external_rows.parquet"):
        (run.state / name).unlink(missing_ok=True)
    bundle = run.state / "external_bundle"
    shutil.rmtree(bundle, ignore_errors=True)
    if source == "pinned":
        # SART3: provenance, format and integrity mechanism, stated before anything is downloaded or deserialised.
        print({"artifact": "pinned sample artifact (created outside this run)", "url": SAMPLE_ARTIFACT_URL, "pinned_sha256": SAMPLE_ARTIFACT_SHA256, "producer": SAMPLE_ARTIFACT_PRODUCER, "format": P.ARTIFACT_FORMAT, "integrity": "SHA-256 of the whole ZIP, checked before extraction; then per-member sizes and digests from artifact.json"})
        zip_path = fetch_pinned_artifact(run.root / "inputs" / "external_artifact.zip", SAMPLE_ARTIFACT_URL, SAMPLE_ARTIFACT_SHA256)
        trust = {"trusted_digest": "verified (pinned SAMPLE_ARTIFACT_SHA256)", "zip_sha256": SAMPLE_ARTIFACT_SHA256, "url": SAMPLE_ARTIFACT_URL, "producer": SAMPLE_ARTIFACT_PRODUCER}
    elif source == "own":
        zip_path = Path(opts.get("zip_path") or "")
        if not str(zip_path) or not zip_path.is_file():
            raise FileNotFoundError(f"ARTIFACT_ZIP_PATH {str(zip_path)!r} is not a file in this runtime: point it at the artifact ZIP you were given.")
        expected = str(opts.get("expected_zip_sha256") or "")
        if expected:
            trust = {"trusted_digest": "verified (EXPECTED_ZIP_SHA256)", "zip_sha256": check_trusted_digest(zip_path, expected, "EXPECTED_ZIP_SHA256")}
        else:
            trust = {"trusted_digest": "not supplied", "zip_sha256": sha256_file(zip_path)}
            print("No EXPECTED_ZIP_SHA256 was supplied: the checks below establish internal consistency only, not that this is the artifact you were sent. Ask the producer for the digest it printed when it exported.")
    else:
        raise ValueError(f"ARTIFACT_SOURCE must be the pinned sample or your own ZIP, got {source!r}")
    P.safe_extract_zip(zip_path, bundle)
    matches = list(bundle.rglob("artifact.json"))
    if len(matches) != 1:
        raise ValueError(f"{zip_path.name}: expected exactly one artifact.json in the artifact, found {len(matches)}.")
    root = matches[0].parent
    manifest = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    compatibility = P.validate_artifact_runtime(manifest, expected_tabicl_version=importlib.metadata.version("tabicl"), expected_torch_version=torch.__version__.split("+")[0])
    binding = check_bundle_identity(P, manifest)
    members = P.verify_artifact_bundle(root, manifest)
    contract = None
    if source == "pinned":
        contract = magic_contract(run, P)
        check_data_contract(manifest, contract)
    context = pd.read_parquet(members["training_context"])
    target = manifest["targetColumn"]
    record = {"source": source, "zip": zip_path.name, **trust, **binding, "root": str(root), "manifest": manifest, "contextRows": len(context), "contextClassCounts": {str(k): int(v) for k, v in context[target].value_counts().sort_index().items()}, "compatibility": compatibility, "data_contract": "MAGIC default path (features, target and class labels match)" if contract else "not checked (your own artifact; your rows must follow its manifest)"}
    run.write_state("external_artifact.json", record)
    print({k: record[k] for k in ("source", "zip", "zip_sha256", "trusted_digest", "mode", "checkpoint_binding", "data_contract")})
    print({"artifactFormat": manifest.get("artifactFormat"), "tabiclVersion": manifest.get("tabiclVersion"), "selection": manifest.get("selectionBasis"), "base": manifest.get("baseCheckpoint"), "base_revision": str(manifest.get("baseModelRevision"))[:12], "producer_runtime": manifest.get("runtime"), "producer_notebook_source": {k: (manifest.get("notebookSource") or {}).get(k) for k in ("repository", "revision", "generator")}})
    print({"featureColumns": len(manifest["featureColumns"]), "targetColumn": target, "classLabels": manifest.get("classLabels"), "contextRows": len(context), "contextClassCounts": record["contextClassCounts"], "nEstimators": manifest["inference"]["nEstimators"], "categoricalEncoders": sorted(manifest["inference"].get("categoricalEncoders", {}))})


def external_serving(run: Run):
    """Serving state rebuilt from the validated artifact's files alone: its checkpoint, its training context (the
    fitted support rows) and its fitted categorical encoders. Nothing is refitted from inference data."""
    import pandas as pd
    import torch

    P = package(run.root)
    art = run.read_state("external_artifact.json", "external_reconstruct")
    manifest = art["manifest"]
    members = P.verify_artifact_bundle(Path(art["root"]), manifest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = manifest["inference"]
    n_estimators, random_state = int(inference["nEstimators"]), inference["randomState"]
    serving = P.TabICLClassificationPipeline(P.create_classifier(model_path=members["checkpoint"], allow_auto_download=False, n_estimators=n_estimators, random_state=random_state, device=device), model_path=members["checkpoint"], n_estimators=n_estimators, random_state=random_state, device=device, source="artifact")
    context = pd.read_parquet(members["training_context"])
    serving.fit(context[manifest["featureColumns"]], context[manifest["targetColumn"]])
    classes = list(serving.classes_)
    if manifest.get("classLabels") is not None and [str(c) for c in classes] != list(manifest["classLabels"]):
        raise RuntimeError("Reconstructed class labels disagree with the validated manifest.")
    return P, art, manifest, serving, classes


def stage_external_reconstruct(run: Run) -> None:
    _P, art, manifest, serving, classes = external_serving(run)
    print({"context_rows": art["contextRows"], "features": len(manifest["featureColumns"]), "classes_": [str(c) for c in classes], "device": serving.device, "source": serving.source, "checkpoint_binding": art["checkpoint_binding"], "categorical_encoders_restored": sorted(manifest["inference"].get("categoricalEncoders", {}))})


def external_sample_rows(run: Run, P, features: list[str]):
    """The default input: the first eight rows of the pinned MAGIC validation member, label removed. They follow the
    default data contract (SART10) and are not in the artifact's support rows; ``test.csv`` is never read."""
    payload = fetch_magic(run.root / "inputs" / "magic-gamma.zip")
    holdout = P.read_csv_payload(magic_member(payload, "val.csv"), "val.csv").reset_index(drop=True)
    rows = holdout[features].head(EXTERNAL_SAMPLE_ROWS).copy()
    rows.insert(0, "row_id", list(range(len(rows))))
    return rows


def stage_external_predict(run: Run) -> None:
    import io as _io
    import platform

    import numpy
    import pandas
    import sklearn

    P, art, manifest, serving, classes = external_serving(run)
    features = list(manifest["featureColumns"])
    encoders = manifest["inference"].get("categoricalEncoders", {})
    path = str(run.options.get("rows_path") or "")
    if path:
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"EXTERNAL_ROWS_PATH {path!r} does not exist in this runtime: upload the file or correct the path.")
        name, payload, kind = file.name, file.read_bytes(), "BYOD"
    else:
        if art["source"] != "pinned":
            raise ValueError("The MAGIC sample rows follow the pinned sample artifact's contract only. With your own artifact, set EXTERNAL_ROWS_PATH to rows that follow its manifest.")
        sample = external_sample_rows(run, P, features)
        buffer = _io.StringIO()
        sample.to_csv(buffer, index=False)
        name, payload, kind = f"magic-val-first-{len(sample)}-rows.csv", buffer.getvalue().encode("utf-8"), "sample"
    rows = P.read_inference_csv(payload, features)
    manifest_in = P.validate_inputs(rows, None, feature_columns=features, names=[name])
    try:  # a probe that breaks the artifact's schema is refused; the finding is recorded, not swallowed
        P.validate_inputs(rows.drop(columns=[features[0]]), None, feature_columns=features)
    except ValueError as exc:
        manifest_in["findings"].append({"input": "missing-column-probe", "verdict": "rejected", "message": str(exc)})
    else:
        raise RuntimeError("The missing-column probe was accepted: input validation is not working, so no prediction is written.")
    rows = check_numeric_features(name, rows, features, encoders)
    X, unseen = P.apply_categorical_encoder(rows[features], encoders)
    out = rows.copy()
    proba = numpy.asarray(serving.predict_proba(X))
    if not numpy.all(numpy.isfinite(proba)) or float(numpy.max(numpy.abs(proba.sum(axis=1) - 1.0))) > importlib.import_module(f"{PACKAGE}.api").MAX_PROBABILITY_ROW_SUM_DEVIATION:
        raise RuntimeError("Class probabilities are not finite or do not sum to 1 per row: no prediction is written.")
    out["prediction"] = serving.predict(X)
    for index, label in enumerate(classes):
        out[f"probability_{label}"] = proba[:, index]
    target = external_out(run)
    target.mkdir(parents=True, exist_ok=True)
    out.to_csv(target / f"{STEM}_external_predictions.csv", index=False)
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    payload_out = {
        "section": "external artifact (NOTEBOOK_SPEC 3.0 §19.1): an artifact made outside this run, validated and served here",
        "predictions": out.to_dict(orient="records"),
        "input_manifest": manifest_in,
        "input": {"filename": name, "kind": kind, "rows": len(out), "extra_columns": manifest_in["inputs"][0]["extra_columns"], "features": features},
        "artifact": {k: art.get(k) for k in ("source", "zip", "zip_sha256", "url", "producer", "trusted_digest", "mode", "checkpoint_binding", "data_contract", "contextRows", "compatibility")} | {"manifest": manifest},
        "inference": {"nEstimators": serving.n_estimators, "randomState": serving.random_state, "decisionRule": P.DECISION_RULE + " over uncalibrated class probabilities", "threshold": None, "classLabels": [str(c) for c in classes], "unseenCategoricalValues": unseen},
        "notebook_source": source,
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "runtime": {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__, "device": serving.device},
    }
    atomic_write_json(target / f"{STEM}_external_result.json", payload_out)
    if unseen:
        print("unseen categorical values mapped to the unknown code:", unseen)
    print(out.round(4).to_string(index=False))
    print({"outputs": [f"outputs/{EXTERNAL_DIR}/{STEM}_external_predictions.csv", f"outputs/{EXTERNAL_DIR}/{STEM}_external_result.json"], "input_kind": kind, "artifact_source": art["source"], "zip_sha256": art["zip_sha256"]})


STAGES = {
    "weights": stage_weights,
    "data": stage_data,
    "validate": stage_validate,
    "condition": stage_condition,
    "report": stage_report,
    "predict": stage_predict,
    "export": stage_export,
    "reload": stage_reload,
    "activity": stage_activity,
    "fresh_test": stage_fresh_test,
    "external_artifact": stage_external_artifact,
    "external_reconstruct": stage_external_reconstruct,
    "external_predict": stage_external_predict,
}
REFUSAL_STAGES = {"data", "validate", "predict"}  # input refusals there are shown as one message, not a traceback (DAT20)


def is_refusal(stage: str, options: dict, exc: BaseException) -> bool:
    """A refused user input (DAT20), not a genuine failure: the kernel shows it as one line and never with a traceback.

    Your own artifact ZIP (``external_artifact`` with ``source='own'``) and your own rows (``external_predict`` with a
    ``rows_path``, or your own artifact without rows) are user inputs too; the pinned artifact and sample rows are not.
    """
    import zipfile

    if not isinstance(exc, ValueError | FileNotFoundError | zipfile.BadZipFile):
        return False
    if stage in REFUSAL_STAGES:
        return True
    if stage == "external_artifact":
        return options.get("source") == "own"
    if stage == "external_predict":
        return bool(options.get("rows_path")) or "your own artifact" in str(exc)
    return False


def refusal_name(run: Run, options: dict, message: str) -> str:
    """The file a refusal names (DAT19): the input the message starts with, else every input the stage was given."""
    paths = options.get("paths") or {}
    given = [*paths.values(), options.get("new_data_path"), options.get("zip_path"), options.get("rows_path")]
    names = [Path(str(value)).name for value in given if value]
    for name in names:
        if message.startswith(name + ":") or message.startswith(name + " "):
            return name
    if not names and (run.state / "data.json").is_file():
        names = [str(json.loads((run.state / "data.json").read_text(encoding="utf-8")).get("name") or "")]
    return ", ".join(name for name in names if name) or "(no file)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISM["CUBLAS_WORKSPACE_CONFIG"]  # before CUDA initialises
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        if args.stage in MODEL_STAGES:
            with deterministic_kernels():
                STAGES[args.stage](run)
        else:
            STAGES[args.stage](run)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        refusal = is_refusal(args.stage, run.options, exc)
        record = {"type": type(exc).__name__, "message": str(exc), "refusal": refusal}
        if refusal:  # DAT20: no traceback streams into the cell; the kernel prints the one Refused line
            record["name"] = refusal_name(run, run.options, str(exc))
        else:
            traceback.print_exc()
        atomic_write_json(error_file, record)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
