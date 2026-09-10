from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'tutorials/tabiclv2_classifier_colab.ipynb'
INFERENCE = ROOT / 'tutorials/tabiclv2_classifier_artifact_inference_colab.ipynb'
README = ROOT / "tutorials/README.md"
ROOT_README = ROOT / "README.md"
RELEASE_LOCK = ROOT / "tutorials/requirements-release.lock"

MODEL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
CHECKPOINT_NAME = 'tabicl-classifier-v2-20260212.ckpt'
CHECKPOINT_SHA256 = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
TABICL_VERSION = "2.1.1"
BASE_CLASS = 'TabICLClassifier'
FINETUNED_CLASS = 'FinetunedTabICLClassifier'
ARTIFACT_FORMAT = 'tabicl-dimer-classifier-v1'
NOTEBOOK_SPEC = '1.0'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_nb(path: Path) -> dict:
    require(path.exists(), f"missing notebook: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def text(nb: dict) -> str:
    chunks = []
    for cell in nb["cells"]:
        src = cell.get("source", "")
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def code_text(nb: dict) -> str:
    out = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        out.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(out)


def compile_cells(nb: dict, label: str) -> None:
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        s = "".join(src) if isinstance(src, list) else str(src)
        s = "\n".join(line for line in s.splitlines() if not line.lstrip().startswith("%"))
        if s.strip():
            ast.parse(s, filename=f"{label}:cell{i}")


main_nb = load_nb(MAIN)
inf_nb = load_nb(INFERENCE)
main_all = text(main_nb)
inf_all = text(inf_nb)
main = code_text(main_nb)
inf = code_text(inf_nb)
compile_cells(main_nb, "main")
compile_cells(inf_nb, "inference")

main_dimer = main_nb.get("metadata", {}).get("dimer", {})
inf_dimer = inf_nb.get("metadata", {}).get("dimer", {})
require(main_dimer.get("notebook_profile") == "E2E", "main notebook metadata profile must be E2E")
require(inf_dimer.get("notebook_profile") == "ARTIFACT-INFERENCE", "inference notebook metadata profile must be ARTIFACT-INFERENCE")
require(str(main_dimer.get("notebook_spec")) == NOTEBOOK_SPEC, "main notebook spec metadata must be 1.0")
require(str(inf_dimer.get("notebook_spec")) == NOTEBOOK_SPEC, "inference notebook spec metadata must be 1.0")
require('**Profile:** `E2E`' in main_all, "main notebook missing visible E2E profile declaration")
require('**Profile:** `ARTIFACT-INFERENCE`' in inf_all, "inference notebook missing visible ARTIFACT-INFERENCE profile declaration")

for marker in (
    'requirements-release.lock',
    'RELEASE_LOCK_SHA256',
    'subprocess.run',
    CHECKPOINT_NAME,
    MODEL_REVISION,
    CHECKPOINT_SHA256,
    'CHECKPOINT_SOURCE = "Pinned upstream"',
    '"DIMER ZIP"',
    "allow_auto_download=False",
    BASE_CLASS,
    FINETUNED_CLASS,
    "RUN_FINE_TUNING = False",
    "MIN_SELECTION_HOLDOUT_ROWS = 50",
    "default:pretrained",
    "holdout-too-small",
    "read_csv_payload",
    "contains duplicate column names",
    ARTIFACT_FORMAT,
    "training_context.parquet",
    "checkpoints/best.ckpt",
    "artifact.json",
    "baseline_metrics[EVAL_METRIC]",
    "MAX_ARTIFACT_EXPANDED_BYTES",
    "checkpointBytes",
    "trainingContextBytes",
    "Expected inference CSV feature columns",
):
    require(marker in main, f"main code missing {marker!r}")
for marker in (
    "GPT-5.6 Sol High",
    "OpenAI / ChatGPT",
    "active Google Colab runtime",
    "Reproducibility and remaining variability",
    "This notebook does not demonstrate",
):
    require(marker in main_all, f"main notebook missing {marker!r}")

for marker in (
    'requirements-release.lock',
    'RELEASE_LOCK_SHA256',
    'subprocess.run',
    BASE_CLASS,
    ARTIFACT_FORMAT,
    "EXPECTED_ZIP_SHA256",
    "safe_extract_zip",
    "allow_auto_download=False",
    "trainingContext",
    "checkpointSha256",
    "trainingContextSha256",
    "checkpointBytes",
    "trainingContextBytes",
    "read_inference_csv",
    "Inference CSV contains duplicate column names",
    "manifest_member_path",
    "rel.is_absolute()",
    "Ambiguous backslash",
    "baseModelRevision",
    "baseModelSha256",
    "wholeArchiveSha256",
    "runtimePyTorch",
    "MAX_ARCHIVE_EXPANDED_BYTES",
    "MAX_ARCHIVE_MEMBERS",
    "Unexpected unlisted artifact file",
    "Expected inference CSV feature columns",
):
    require(marker in inf, f"inference code missing {marker!r}")
for marker in (
    "GPT-5.6 Sol High",
    "active Google Colab runtime",
    "Interpretation and limits",
    "This notebook does not demonstrate",
    "remaining run-to-run variability",
):
    require(marker in inf_all, f"inference notebook missing {marker!r}")

require(RELEASE_LOCK.exists(), "missing tutorials/requirements-release.lock")
lock_bytes = RELEASE_LOCK.read_bytes()
lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
require(lock_sha256 == "64e9a167567495263694f555195ca7df4488016cc76ab1e327e480509a4ac6d5", f"release lock SHA-256 drifted: {lock_sha256}")
lock_lines = [line.strip() for line in lock_bytes.decode("utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
require(lock_lines, "release lock is empty")
for line in lock_lines:
    require("==" in line, f"release lock contains non-exact dependency: {line}")
for notebook_code, label in ((main, "main"), (inf, "inference")):
    require(f'RELEASE_LOCK_SHA256 = "64e9a167567495263694f555195ca7df4488016cc76ab1e327e480509a4ac6d5"' in notebook_code, f"{label} notebook lock digest mismatch")
    require("%pip -q install" not in notebook_code, f"{label} notebook still resolves an independent direct pip graph")

require(FINETUNED_CLASS not in inf, "inference notebook must not import/use fine-tuning class")
require("RUN_FINE_TUNING" not in inf, "inference notebook must not expose fine-tuning")
for marker in ['TabICLRegressor', 'FinetunedTabICLRegressor', 'tabicl-dimer-regressor-v1']:
    require(marker not in main, f"task leakage in main notebook: {marker}")
    require(marker not in inf, f"task leakage in inference notebook: {marker}")
for marker in ['predict_proba', 'support_many_classes=True', 'probability_']:
    require(marker in main, f"task-specific main marker missing: {marker}")

root_readme = ROOT_README.read_text(encoding="utf-8")
tutorial_readme = README.read_text(encoding="utf-8")
github_badge = "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white"
colab_url = "https://colab.research.google.com/github/kurtvalcorza/tabicl-classifier-pipeline/blob/main/tutorials/tabiclv2_classifier_colab.ipynb"
for doc, label in ((root_readme, "root README"), (tutorial_readme, "tutorial README")):
    require(github_badge in doc, f"{label} missing GitHub badge")
    require(colab_url in doc, f"{label} missing main Colab badge")
require(colab_url in main_all, "main notebook missing its Open In Colab badge")
require("DIMER Notebook Specification v1.0" in tutorial_readme, "tutorial README missing notebook spec declaration")
require("| `E2E` |" in tutorial_readme, "tutorial README missing E2E profile mapping")
require("| `ARTIFACT-INFERENCE` |" in tutorial_readme, "tutorial README missing artifact-inference profile mapping")
require("Static validation is not execution evidence" in tutorial_readme, "tutorial README must distinguish static and runtime evidence")
require("separate fresh kernel" in tutorial_readme, "tutorial README missing fresh-kernel execution boundary")

print("Standalone TabICLv2 Classifier Colab tutorials: NOTEBOOK_SPEC v1.0 static conformance OK")
