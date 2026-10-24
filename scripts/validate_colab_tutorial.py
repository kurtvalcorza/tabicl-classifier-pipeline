#!/usr/bin/env python3
"""Repository-specific static checks for the standalone tutorials (NOTEBOOK_SPEC 2.2).

The carrier, parity, hygiene and profile checks live in ``tools/validate_release_assets.py`` (run first). This
script keeps the invariants that are specific to this repository's release contract: the exact PEP 517 build
backend, the release requirement/lock files and their agreement with the ``pyproject.toml`` runtime pins, the
hash lock the notebooks carry, the fine-tuning gate, the serving-bundle hardening markers, and the executor's
evidence contract. The notebooks carry the package and a stage runner as text in one carrier cell
(``metadata.dimer.embedded_sources``); the "own code" checked here is the notebook's kernel cells plus the carried
stage runner, and the "all text" adds the carried package. It claims no runtime execution evidence.
"""
# ruff: noqa: E501  -- rule messages name the requirement in full; they are kept on one line
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials" / "tabiclv2_classifier_colab.ipynb"
README = ROOT / "tutorials" / "README.md"
REQ = ROOT / "tutorials" / "requirements-release.in"
LOCK = ROOT / "tutorials" / "requirements-release.lock"
COLAB_LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PYPROJECT = ROOT / "pyproject.toml"
API = ROOT / "src" / "tabicl_classifier_pipeline" / "api.py"
EXECUTOR = ROOT / "scripts" / "execute_notebook_release_paths.py"
BUILD_BACKEND_PIN = "setuptools==78.1.0"
CHECKPOINT_NAME = "tabicl-classifier-v2-20260212.ckpt"
MODEL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
ARTIFACT_FORMAT = "tabicl-dimer-classifier-v2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_nb(path: Path) -> dict:
    require(path.exists(), f"missing notebook {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _is_carrier(cell: dict) -> bool:
    return bool(cell.get("metadata", {}).get("dimer", {}).get("embedded_sources"))


def carried(nb: dict) -> dict[str, str]:
    """The CARRIED_FILES literal of the notebook's single carrier cell."""
    cells = [c for c in nb["cells"] if c.get("cell_type") == "code" and _is_carrier(c)]
    require(len(cells) == 1, "exactly one carrier cell (metadata.dimer.embedded_sources) is required")
    for node in ast.parse(_source(cells[0])).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CARRIED_FILES":
            return ast.literal_eval(node.value)
    raise AssertionError("the carrier cell does not assign CARRIED_FILES")


def text(nb: dict) -> str:
    """Every non-carrier cell plus every carried text file."""
    own = [_source(c) for c in nb["cells"] if not _is_carrier(c)]
    return "\n".join(own + list(carried(nb).values()))


def code_text(nb: dict, *, outside_modules: bool = False) -> str:
    """The notebook's code cells; with outside_modules, its own kernel cells plus the carried stage runner."""
    parts = [_source(c) for c in nb["cells"] if c.get("cell_type") == "code" and not (outside_modules and _is_carrier(c))]
    if outside_modules:
        parts.append(carried(nb)["tutorial_stages.py"])
    return "\n".join(parts)


def sys_path_src() -> None:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


main_nb = load_nb(MAIN)
main_all = text(main_nb)
main_own = code_text(main_nb, outside_modules=True)
# NOTEBOOK_SPEC 3.0 (KND11): the artifact-inference companion was folded into the E2E notebook's closing
# external-artifact section (§19.1); the checks that bound the companion now bind that section.
require(not (ROOT / "tutorials" / "tabiclv2_classifier_artifact_inference_colab.ipynb").exists(), "the retired companion notebook must not ship (KND11)")
inf_all, inf_own = main_all, main_own

# Profiles / learning contract (the carrier-level checks are the validator's).
for nb, profile, label, spec_version in ((main_nb, "E2E", "main", "3.0"),):
    dimer = nb.get("metadata", {}).get("dimer", {})
    require(dimer.get("notebook_profile") == profile, f"{label} profile metadata")
    require(dimer.get("notebook_spec") == spec_version and dimer.get("standalone") is True, f"{label} must be standalone spec {spec_version}")
for marker in ("**Profile:** `E2E`", "**This notebook is standalone.**", "## Interpretation and limits"):
    require(marker in main_all, f"main missing {marker!r}")
for marker in ("## 10. Using an artifact made elsewhere", "**This section needs only the setup cells (Sections 1–3).**", "**Trust boundary.**", "outputs/external_artifact/"):
    require(marker in inf_all, f"external-artifact section missing {marker!r}")

# Reproducible environment: the notebooks carry the hash lock compiled from pyproject's pins; the release files stay
# consistent with the same pins.
require(REQ.exists() and LOCK.exists(), "release requirements/lock missing")
req_lines = [line.strip() for line in REQ.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
lock_lines = [line.strip() for line in LOCK.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
require(all("==" in line for line in req_lines), "top-level release requirements contain non-exact pins")
require(all("==" in line for line in lock_lines), "transitive release lock contains non-exact pins")
locked = {line.split("==", 1)[0].lower().replace("_", "-"): line.split("==", 1)[1] for line in lock_lines}
for package in ("tabicl", "numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "huggingface-hub", "transformers", "wandb", "lightgbm", "einops"):
    require(package in locked, f"transitive lock missing {package}")
# The lock was resolved with pip-compile, which reports setuptools under "unsafe" as a comment rather than a
# pinned line; the exact PEP 517 build backend is therefore asserted in pyproject.toml (next checks).
require("setuptools" in LOCK.read_text(encoding="utf-8"), "release lock must mention the build backend (unsafe block)")
pyproject = PYPROJECT.read_text(encoding="utf-8")
require(f'requires = ["{BUILD_BACKEND_PIN}"]' in pyproject, "PEP 517 build backend must be exact-pinned to the release lock")
require('build-backend = "setuptools.build_meta"' in pyproject, "unexpected PEP 517 build backend")
require("setuptools>=" not in pyproject and "setuptools~=" not in pyproject, "floating setuptools build requirement is forbidden")
deps_block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", pyproject, re.M | re.S)
require(deps_block is not None, "pyproject.toml must declare [project].dependencies (the notebooks' PINS)")
pins = {p.split("==", 1)[0].lower().replace("_", "-"): p.split("==", 1)[1] for p in re.findall(r'"([^"]+)"', deps_block.group(1))}
require(all("==" in p for p in re.findall(r'"([^"]+)"', deps_block.group(1))), "pyproject runtime deps must be == pinned")
for name, version in pins.items():
    if name in locked:
        require(locked[name] == version, f"pyproject pin {name}=={version} disagrees with requirements-release.lock ({locked[name]})")
require(pins.get("torch") == "2.14.0", "torch must be pinned to the release-verified 2.14.0")
require(COLAB_LOCK.exists(), "tutorials/requirements-colab.lock.txt (the hash lock the notebooks carry) is missing")
colab_lock = COLAB_LOCK.read_text(encoding="utf-8")
require(re.search(r"^torch==2\.14\.0 ", colab_lock, re.M) is not None, "requirements-colab.lock.txt must pin torch==2.14.0")
colab_locked = {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", colab_lock, re.M)}
for name, version in pins.items():
    require(colab_locked.get(name) == version, f"requirements-colab.lock.txt does not pin {name}=={version} (found {colab_locked.get(name)})")
require("--hash=sha256:" in colab_lock, "requirements-colab.lock.txt must carry hashes")
for nb, label in ((main_nb, "main"),):
    require(carried(nb).get("requirements.txt") == colab_lock.replace("\r\n", "\n"), f"{label} notebook must carry requirements-colab.lock.txt verbatim")

# G2: core model operations flow through the carried public API; no direct tabicl use in the notebooks' own cells.
api_source = API.read_text(encoding="utf-8")
for marker in ("def create_classifier", "def create_finetuned_classifier", "def fine_tune_classifier", "def condition_classifier", "def predict_labels", "def predict_probabilities", "def align_probabilities", "def classification_metrics", "def majority_class_baseline", "def check_stratifiable", "def validate_artifact_runtime", "class TabICLClassificationPipeline", "def safe_extract_zip", "def verify_artifact_bundle", "def manifest_member_path", "def validate_inputs", "def evaluation_report"):
    require(marker in api_source, f"public API missing {marker}")
for own, label in ((main_own, "main"),):
    require("from tabicl import" not in own and "import tabicl" not in own, f"{label} bypasses the carried API with a direct tabicl import")
    require("from tabicl_classifier_pipeline" not in own, f"{label} imports the repository package (ST1)")
    require("P.verify_snapshot(snapshot)" in own, f"{label} does not verify the pinned snapshot through the carried API")
require("TabICLClassificationPipeline.from_pretrained(weights_dir=run.weights / P.MODEL_KEY" in main_own, "main does not resolve the model through the carried API")
for marker in ("create_finetuned_classifier(", "fine_tune_classifier(", "pipe.fit(X_train, y_train)", "model.predict(", "model.predict_proba(", "compare_metric("):
    require(marker in main_own, f"main does not exercise repository API operation {marker}")
for marker in ("create_classifier(", "serving.fit(", "serving.predict(", "serving.predict_proba(", "validate_artifact_runtime(", "safe_extract_zip(", "verify_artifact_bundle(", "fetch_pinned_artifact(", "SAMPLE_ARTIFACT_SHA256"):
    require(marker in inf_own, f"external-artifact section does not exercise repository API operation {marker}")

# Fine-tuning is the pipeline's production adaptation and runs on the default path when CUDA is present (TIC-M2);
# without CUDA the stage records why it was skipped. Statically protect its callable path and its guards.
require("RUN_FINE_TUNING = True  # @param" in main_own, "fine-tune gate must default to True (TIC-M2)")
require("finetuner = P.create_finetuned_classifier(" in main_own, "fine-tune constructor does not use repository API")
require('"epochs": 30, "patience": 8' in main_own and '"time_limit": 1800' in main_own and '"max_data_size": 500' in main_own and '"use_lr_scheduler": False' in main_own, "the default fine-tune must be the pre-registered grid (R96-M3: real steps, no zero-LR step, bounded)")
require(
    main_own.index('if run_ft and device != "cuda" and not cpu_grid:') < main_own.index("grid.append(run_config("),
    "fine-tune workspace must be initialized after the CUDA guard",
)
require("MIN_SELECTION_HOLDOUT_ROWS = 50" in main_own, "selection evidence guard missing")

# Model / task provenance and semantics.
for marker in (CHECKPOINT_NAME, MODEL_REVISION, CHECKPOINT_SHA256, ARTIFACT_FORMAT):
    require(marker in main_all, f"main missing provenance marker {marker!r}")
for marker in ('"artifactFormat": P.ARTIFACT_FORMAT', "training_context.parquet", "P.BUNDLE_CHECKPOINT", "artifact.json", "RUN_NEW_DATA_INFERENCE = False  # @param"):
    require(marker in main_own, f"main code missing {marker!r}")
for marker in ("classification_metrics(", "majority_class_baseline(", "Need at least {min_classes} target classes", "unseen in training", "trivial baseline"):
    require(marker in main_all, f"classification requirement missing {marker!r}")
require("uncalibrated" in main_all and "no threshold shipped" in main_all, "main probability-calibration boundary missing")
require("Reproducibility boundary" in main_all, "main variability statement missing")
require("BYOD privacy boundary" in main_all, "data locality/privacy statement missing")
require("Files stay inside this runtime" in inf_all, "external-artifact data locality statement missing")

# Artifact provenance must establish runtime compatibility before checkpoint reconstruction.
require("validate_artifact_runtime(" in inf_own, "artifact runtime compatibility is not validated")
section = inf_own[inf_own.index("def stage_external_artifact("):]
fetcher = inf_own[inf_own.index("def fetch_pinned_artifact("):inf_own.index("def check_bundle_identity(")]
require(fetcher.index("check_trusted_digest(part") < fetcher.index("os.replace(part, cache)"), "the pinned download must be digest-checked before it is kept")
require(section.index("fetch_pinned_artifact(") < section.index("safe_extract_zip("), "the pinned digest must be checked before extraction")
require(section.index("validate_artifact_runtime(") < section.index("create_classifier("), "runtime compatibility must be established before model reconstruction")
require(section.index("verify_artifact_bundle(") < section.index("create_classifier("), "bundle digests must be verified before model reconstruction")

# Archive / artifact hardening lives in the carried module; the notebooks must use it and never extract raw.
for marker in ("MAX_ARTIFACT_EXPANDED_BYTES", "Ambiguous backslash ZIP member", "Unexpected or missing artifact files", "Checkpoint size mismatch", "Training-context size mismatch", "def manifest_member_path"):
    require(marker in api_source, f"api hardening marker missing {marker!r}")
for own, label in ((main_own, "main"),):
    # zipfile.ZipFile is used read-only (the uv wheel, the pinned Penguins archive); extraction goes through safe_extract_zip.
    require(".extractall(" not in own and ".extract(" not in own, f"{label} must extract only through safe_extract_zip")
for marker in ('"payloadFiles"', '"sizes"', '"runtime"', "safe_extract_zip(", "verify_artifact_bundle("):
    require(marker in main_own, f"main hardening marker missing {marker!r}")
require("EXPECTED_ZIP_SHA256" in inf_own, "external-artifact hardening marker missing EXPECTED_ZIP_SHA256")

# Source hygiene.
for label, content in (("main", main_all),):
    for token in ("TODO", "TBD", "FIXME"):
        require(re.search(rf"\b{token}\b", content) is None, f"{label} contains {token}")
    require("C:\\Users\\" not in content and "/home/" not in content, f"{label} contains developer-local path")

# Registry.
registry = README.read_text(encoding="utf-8")
require("| `E2E` |" in registry and "| `ARTIFACT-INFERENCE` |" not in registry, "tutorial registry must list the E2E notebook and no artifact-inference companion")
require("DIMER Notebook Specification 3.0" in registry, "tutorial registry missing the spec version")
require("requirements-release.lock" in registry, "tutorial registry does not document transitive lock")

# Executor evidence contract (the previous pair's kernel executor; must not fake execution).
executor = EXECUTOR.read_text(encoding="utf-8")
for marker in ("NotebookClient", "client.execute()", "separate fresh Jupyter kernels", "separate fresh kernel"):
    require(marker in executor, f"release executor missing notebook-engine evidence marker {marker!r}")
require("exec(compile(" not in executor, "release verifier still executes notebook cells as plain Python")
require("startswith(\"%\")" not in executor, "release verifier still strips notebook magics")

print("NOTEBOOK_SPEC 3.0 repository-specific static conformance: OK")
print("NOTE: static validation is not clean-runtime execution evidence.")
