from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'tutorials/tabiclv2_classifier_colab.ipynb'
INFERENCE = ROOT / 'tutorials/tabiclv2_classifier_artifact_inference_colab.ipynb'
README = ROOT / "tutorials/README.md"
ROOT_README = ROOT / "README.md"

MODEL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
CHECKPOINT_NAME = 'tabicl-classifier-v2-20260212.ckpt'
CHECKPOINT_SHA256 = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
TABICL_VERSION = "2.1.1"
BASE_CLASS = 'TabICLClassifier'
FINETUNED_CLASS = 'FinetunedTabICLClassifier'
ARTIFACT_FORMAT = 'tabicl-dimer-classifier-v1'

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

for marker in (
    'tabicl[finetune]==2.1.1',
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
):
    require(marker in main, f"main code missing {marker!r}")
for marker in ("GPT-5.6 Sol High", "OpenAI / ChatGPT"):
    require(marker in main_all, f"main notebook missing {marker!r}")

for marker in (
    'tabicl==2.1.1',
    BASE_CLASS,
    ARTIFACT_FORMAT,
    "EXPECTED_ZIP_SHA256",
    "safe_extract_zip",
    "allow_auto_download=False",
    "trainingContext",
    "checkpointSha256",
    "trainingContextSha256",
    "read_inference_csv",
    "Inference CSV contains duplicate column names",
    "manifest_member_path",
    "rel.is_absolute()",
):
    require(marker in inf, f"inference code missing {marker!r}")
require("GPT-5.6 Sol High" in inf_all, "inference notebook missing AI provenance")

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

print("Standalone TabICLv2 Classifier Colab tutorials: OK")
