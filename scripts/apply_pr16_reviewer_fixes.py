from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
LOCK = TUTORIALS / "requirements-release.lock"
MAIN = TUTORIALS / "tabiclv2_classifier_colab.ipynb"
INFERENCE = TUTORIALS / "tabiclv2_classifier_artifact_inference_colab.ipynb"
README = TUTORIALS / "README.md"
CONFORMANCE = ROOT / "NOTEBOOK_SPEC_CONFORMANCE.md"
RELEASE_EVIDENCE = TUTORIALS / "RELEASE_EVIDENCE.md"
VALIDATOR = ROOT / "scripts" / "validate_colab_tutorial.py"
OLD_REQUIREMENTS = TUTORIALS / "requirements-release.txt"

LOCK_URL = (
    "https://raw.githubusercontent.com/kurtvalcorza/tabicl-classifier-pipeline/"
    "main/tutorials/requirements-release.lock"
)


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"{label}: expected source fragment not found")
    return text.replace(old, new, 1)


def install_cell(lock_sha256: str, include_lightgbm: bool) -> str:
    packages = ["tabicl", "pyarrow", "pandas", "numpy", "scikit-learn", "torch"]
    if include_lightgbm:
        packages.extend(["lightgbm", "huggingface-hub"])
    package_list = repr(packages)
    return f'''import hashlib
import importlib.metadata
import subprocess
import sys
import urllib.request
from pathlib import Path

RELEASE_LOCK_SHA256 = "{lock_sha256}"
RELEASE_LOCK_URL = "{LOCK_URL}"
LOCAL_RELEASE_LOCK = Path("tutorials/requirements-release.lock")
RUNTIME_RELEASE_LOCK = Path("/tmp/tabicl-classifier-requirements-release.lock")

if LOCAL_RELEASE_LOCK.exists():
    lock_payload = LOCAL_RELEASE_LOCK.read_bytes()
    lock_source = str(LOCAL_RELEASE_LOCK)
else:
    with urllib.request.urlopen(RELEASE_LOCK_URL, timeout=30) as response:
        lock_payload = response.read()
    lock_source = RELEASE_LOCK_URL

observed_lock_sha256 = hashlib.sha256(lock_payload).hexdigest()
if observed_lock_sha256 != RELEASE_LOCK_SHA256:
    raise RuntimeError(
        f"Release dependency lock SHA-256 mismatch: {{observed_lock_sha256}}; "
        f"expected {{RELEASE_LOCK_SHA256}}"
    )
RUNTIME_RELEASE_LOCK.write_bytes(lock_payload)
print("Release dependency lock:", lock_source)
print("Release dependency lock SHA-256:", observed_lock_sha256)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(RUNTIME_RELEASE_LOCK)],
    check=True,
)

import torch

TABICL_VERSION = "2.1.1"
if importlib.metadata.version("tabicl") != TABICL_VERSION:
    raise RuntimeError("Unexpected tabicl version")
print("Python:", sys.version.split()[0])
for package in {package_list}:
    print(f"{{package}}:", importlib.metadata.version(package))
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''


def patch_main(lock_sha256: str) -> None:
    notebook = load_notebook(MAIN)
    install_done = False
    for cell in notebook["cells"]:
        text = source_text(cell)
        if cell.get("cell_type") == "code" and "%pip -q install" in text:
            set_source(cell, install_cell(lock_sha256, include_lightgbm=True))
            install_done = True
        if cell.get("cell_type") == "markdown" and "## 1. Install and inspect the runtime" in text:
            text = text.replace(
                "The notebook pins every package it installs directly so the release path does not float between package releases. PyTorch is deliberately treated as part of the Colab/runtime substrate rather than reinstalled after import, which preserves the runtime's tested CPU/CUDA pairing; its exact version is printed and captured in release execution evidence. The cell verifies all direct pins and prints Python, TabICL, PyTorch, the principal data/ML libraries, and the effective accelerator.",
                "The notebook installs from the repository's fully resolved Python 3.12 release lock graph. In a cloned repository it uses `tutorials/requirements-release.lock`; standalone Colab fetches that same lock from GitHub and verifies its embedded SHA-256 before installation. This locks transitive dependencies as well as the direct notebook requirements, including PyTorch. The cell then prints Python, TabICL, PyTorch, the principal data/ML libraries, and the effective accelerator.",
            )
            text = text.replace(
                "**What to look for:** `TabICL: 2.1.1`; every `...: <version>` line for a pinned package must match the declared pin. `CUDA: False` is valid for the default pretrained path; CUDA is required only when optional gradient fine-tuning is enabled.",
                "**What to look for:** the verified release-lock SHA-256 followed by `tabicl: 2.1.1`. `CUDA: False` is valid for the default pretrained path; CUDA is required only when optional gradient fine-tuning is enabled.",
            )
            set_source(cell, text)
        if cell.get("cell_type") == "markdown" and "- **External access:** PyPI" in text:
            text = text.replace(
                "- **External access:** PyPI for the pinned Python packages and Hugging Face for the immutable upstream checkpoint on the default path; no credentials are required for those public resources.",
                "- **External access:** GitHub raw content for the SHA-256-verified release lock when the repository is not locally present, PyPI for the locked Python packages, and Hugging Face for the immutable upstream checkpoint on the default path; no credentials are required for those public resources.",
            )
            set_source(cell, text)
    if not install_done:
        raise RuntimeError("main notebook install cell not found")
    write_notebook(MAIN, notebook)


def patch_inference(lock_sha256: str) -> None:
    notebook = load_notebook(INFERENCE)
    install_done = False
    path_done = False
    provenance_done = False
    for cell in notebook["cells"]:
        text = source_text(cell)
        if cell.get("cell_type") == "code" and "%pip -q install" in text:
            set_source(cell, install_cell(lock_sha256, include_lightgbm=False))
            install_done = True
            continue
        if cell.get("cell_type") == "markdown" and "## 1. Install and verify the matching runtime" in text:
            text = text.replace(
                "The directly installed packages are pinned exactly. PyTorch is treated as part of the runtime substrate so the notebook does not replace an already provisioned CPU/CUDA build; its effective version is printed. The artifact's recorded TabICL version is checked against the pinned estimator before deserialization.",
                "The notebook installs from the same fully resolved Python 3.12 release lock graph as the producer tutorial. In a cloned repository it uses `tutorials/requirements-release.lock`; standalone Colab fetches the lock from GitHub and verifies its embedded SHA-256 before installation. The artifact's recorded TabICL version is checked against that locked runtime before deserialization, and the effective Python, TabICL, PyTorch, and accelerator identities are printed.",
            )
            set_source(cell, text)
        if cell.get("cell_type") == "code" and "def manifest_member_path" in text:
            old = '''    rel = Path(str(value))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe {field} path in artifact.json: {value!r}")
'''
            new = '''    raw_value = str(value)
    if "\\\\" in raw_value:
        raise ValueError(f"Ambiguous backslash {field} path in artifact.json: {value!r}")
    rel = Path(raw_value)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe {field} path in artifact.json: {value!r}")
'''
            text = replace_once(text, old, new, "manifest backslash hardening")
            path_done = True

            old_print = '''print("✓ Artifact structure and digests verified")
print(f"  mode={manifest.get('mode')} selection={manifest.get('selectionBasis')} base={manifest.get('baseCheckpoint')} @ {str(manifest.get('baseModelRevision'))[:12]}")
'''
            new_print = '''print("✓ Artifact structure and digests verified")
print("Artifact provenance and runtime contract:")
print("  artifactFormat:", manifest.get("artifactFormat"))
print("  tabiclVersion:", manifest.get("tabiclVersion"))
print("  baseCheckpoint:", manifest.get("baseCheckpoint"))
print("  baseModelRevision:", manifest.get("baseModelRevision"))
print("  baseModelSha256:", manifest.get("baseModelSha256"))
print("  checkpointSource:", manifest.get("checkpointSource"))
print("  checkpointSha256:", manifest.get("digests", {}).get("checkpointSha256"))
print("  wholeArchiveSha256:", observed)
print("  mode:", manifest.get("mode"))
print("  selectionBasis:", manifest.get("selectionBasis"))
print("  runtimeTabICL:", TABICL_VERSION)
print("  runtimePyTorch:", torch.__version__)
'''
            text = replace_once(text, old_print, new_print, "artifact provenance display")
            provenance_done = True
            set_source(cell, text)
    if not install_done:
        raise RuntimeError("inference notebook install cell not found")
    if not path_done:
        raise RuntimeError("manifest path helper not patched")
    if not provenance_done:
        raise RuntimeError("artifact provenance display not patched")
    write_notebook(INFERENCE, notebook)


def patch_docs(lock_sha256: str) -> None:
    readme = README.read_text(encoding="utf-8")
    old = "The tutorials pin their directly installed packages. PyTorch is treated as part of the supported runtime substrate rather than reinstalled after import, so the notebooks print the effective Python, PyTorch, TabICL, accelerator, and principal package versions used by each run. Explicit seeds control the tutorial's splits and model-level stochastic settings; CUDA kernels, hardware, library internals, and wall-clock latency can still introduce run-to-run variation, so reproducibility claims do not imply bitwise identity across hardware."
    new = f"The tutorials install from the fully resolved `tutorials/requirements-release.lock` graph generated from `tutorials/requirements-release.in` on Python 3.12. Standalone Colab fetches the lock from GitHub only when a local repository copy is unavailable and verifies SHA-256 `{lock_sha256}` before installation. The notebooks then print the effective Python, PyTorch, TabICL, accelerator, and principal package versions used by each run. Explicit seeds control the tutorial's splits and model-level stochastic settings; CUDA kernels, hardware, library internals, and wall-clock latency can still introduce run-to-run variation, so reproducibility claims do not imply bitwise identity across hardware."
    readme = replace_once(readme, old, new, "tutorial README runtime section")
    README.write_text(readme, encoding="utf-8")

    conformance = CONFORMANCE.read_text(encoding="utf-8")
    anchor = "## Automated verification\n"
    env2 = f'''## Dependency lock (ENV2)\n\nThe release-grade notebooks install from `tutorials/requirements-release.lock`, a fully resolved Python 3.12 graph generated from `tutorials/requirements-release.in`. Standalone Colab verifies lock SHA-256 `{lock_sha256}` before installation. The lock includes transitive dependencies rather than relying only on exact top-level requirements.\n\n'''
    if "## Dependency lock (ENV2)" not in conformance:
        conformance = replace_once(conformance, anchor, env2 + anchor, "conformance ENV2 section")
    CONFORMANCE.write_text(conformance, encoding="utf-8")

    evidence = RELEASE_EVIDENCE.read_text(encoding="utf-8")
    if "runner-writable path" not in evidence:
        evidence += "\nGitHub-hosted execution transparently remaps Colab's `/content` path to a runner-writable path and records that mapping in the JSON evidence. This is a filesystem adaptation only; notebook task logic and producer/consumer separation are unchanged.\n"
    RELEASE_EVIDENCE.write_text(evidence, encoding="utf-8")


def patch_validator(lock_sha256: str) -> None:
    validator = VALIDATOR.read_text(encoding="utf-8")
    validator = validator.replace(
        "import ast\nimport json\n",
        "import ast\nimport hashlib\nimport json\n",
        1,
    )
    validator = validator.replace(
        "ROOT_README = ROOT / \"README.md\"\n",
        "ROOT_README = ROOT / \"README.md\"\nRELEASE_LOCK = ROOT / \"tutorials/requirements-release.lock\"\n",
        1,
    )

    direct_main = '''    'tabicl[finetune]==2.1.1',
    'lightgbm==4.7.0',
    'pyarrow==25.0.1',
    'pandas==3.0.5',
    'scikit-learn==1.9.0',
    'huggingface_hub==1.30.0',
'''
    validator = replace_once(
        validator,
        direct_main,
        '''    'requirements-release.lock',
    'RELEASE_LOCK_SHA256',
    'subprocess.run',
''',
        "validator main lock markers",
    )
    direct_inf = '''    'tabicl==2.1.1',
    'pyarrow==25.0.1',
    'pandas==3.0.5',
    'scikit-learn==1.9.0',
'''
    validator = replace_once(
        validator,
        direct_inf,
        '''    'requirements-release.lock',
    'RELEASE_LOCK_SHA256',
    'subprocess.run',
''',
        "validator inference lock markers",
    )
    validator = validator.replace(
        '    "rel.is_absolute()",\n',
        '    "rel.is_absolute()",\n    "Ambiguous backslash",\n    "baseModelRevision",\n    "baseModelSha256",\n    "wholeArchiveSha256",\n    "runtimePyTorch",\n',
        1,
    )

    old_forbidden = '''for forbidden in (
    'lightgbm>=',
    'pyarrow>=',
    'pandas>=',
    'scikit-learn>=',
    'huggingface_hub>=',
):
    require(forbidden not in main, f"main notebook contains floating direct dependency: {forbidden}")
for forbidden in ('pyarrow>=', 'pandas>=', 'scikit-learn>='):
    require(forbidden not in inf, f"inference notebook contains floating direct dependency: {forbidden}")

'''
    new_forbidden = f'''require(RELEASE_LOCK.exists(), "missing tutorials/requirements-release.lock")
lock_bytes = RELEASE_LOCK.read_bytes()
lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
require(lock_sha256 == "{lock_sha256}", f"release lock SHA-256 drifted: {{lock_sha256}}")
lock_lines = [line.strip() for line in lock_bytes.decode("utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
require(lock_lines, "release lock is empty")
for line in lock_lines:
    require("==" in line, f"release lock contains non-exact dependency: {{line}}")
for notebook_code, label in ((main, "main"), (inf, "inference")):
    require(f'RELEASE_LOCK_SHA256 = "{lock_sha256}"' in notebook_code, f"{{label}} notebook lock digest mismatch")
    require("%pip -q install" not in notebook_code, f"{{label}} notebook still resolves an independent direct pip graph")

'''
    validator = replace_once(validator, old_forbidden, new_forbidden, "validator ENV2 checks")
    VALIDATOR.write_text(validator, encoding="utf-8")


def main() -> None:
    if not LOCK.exists():
        raise RuntimeError("tutorials/requirements-release.lock must be generated before applying fixes")
    lock_sha256 = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    patch_main(lock_sha256)
    patch_inference(lock_sha256)
    patch_docs(lock_sha256)
    patch_validator(lock_sha256)
    OLD_REQUIREMENTS.unlink(missing_ok=True)
    print("Applied PR #16 reviewer fixes")
    print("Release lock SHA-256:", lock_sha256)


if __name__ == "__main__":
    main()
