from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials" / "tabiclv2_classifier_colab.ipynb"
INFERENCE = ROOT / "tutorials" / "tabiclv2_classifier_artifact_inference_colab.ipynb"
EVIDENCE_DIR = ROOT / "release-evidence"
CONTENT_DIR = Path(
    os.environ.get("NOTEBOOK_RELEASE_CONTENT_DIR", str(EVIDENCE_DIR / "content"))
).resolve()
COLAB_CONTENT_ROOT = "/content"
ARTIFACT = CONTENT_DIR / "tabiclv2-classifier-artifact.zip"
INFERENCE_CSV = CONTENT_DIR / "tabiclv2_classifier_release_inference.csv"


def colab_stub(content_dir: Path) -> str:
    return f'''
from pathlib import Path
import sys
import types

Path(r"{content_dir}").mkdir(parents=True, exist_ok=True)
try:
    import google
except ImportError:
    google = types.ModuleType("google")
    google.__path__ = []
    sys.modules["google"] = google

_colab = types.ModuleType("google.colab")

class _ReleaseFiles:
    def upload(self):
        raise RuntimeError("Release harness did not configure an upload payload for this notebook stage")

    def download(self, path):
        print(f"[release harness] download requested: {{path}}")

_colab.files = _ReleaseFiles()
setattr(google, "colab", _colab)
sys.modules["google.colab"] = _colab
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute_notebook(
    source: Path,
    output: Path,
    prep_source: str,
    replacements: dict[str, str] | None = None,
) -> None:
    notebook = nbformat.read(source, as_version=4)
    effective_replacements = {COLAB_CONTENT_ROOT: str(CONTENT_DIR)}
    if replacements:
        effective_replacements.update(replacements)

    for old, new in effective_replacements.items():
        replaced = old == COLAB_CONTENT_ROOT
        for cell in notebook.cells:
            if cell.cell_type == "code" and old in cell.source:
                cell.source = cell.source.replace(old, new)
                replaced = True
        if not replaced:
            raise RuntimeError(
                f"Could not apply release-harness replacement in {source.name}: {old!r}"
            )

    prep = nbformat.v4.new_code_cell(prep_source)
    prep.metadata["tags"] = ["release-harness"]
    notebook.cells.insert(0, prep)
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, output)


def package_versions() -> dict[str, str | None]:
    packages = [
        "tabicl",
        "lightgbm",
        "pyarrow",
        "pandas",
        "scikit-learn",
        "huggingface-hub",
        "torch",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    if ARTIFACT.exists():
        ARTIFACT.unlink()

    main_output = EVIDENCE_DIR / "tabiclv2_classifier_colab.executed.ipynb"
    inference_output = EVIDENCE_DIR / "tabiclv2_classifier_artifact_inference_colab.executed.ipynb"

    evidence: dict[str, object] = {
        "notebookSpec": "1.0",
        "revision": os.environ.get("NOTEBOOK_RELEASE_SHA")
        or os.environ.get("GITHUB_SHA")
        or "unknown",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runner": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "executionBoundary": "separate fresh Jupyter kernels",
        "pathRemap": {
            "notebookColabRoot": COLAB_CONTENT_ROOT,
            "runnerWritableRoot": str(CONTENT_DIR),
        },
        "profiles": {
            MAIN.name: "E2E",
            INFERENCE.name: "ARTIFACT-INFERENCE",
        },
        "outcomes": {},
    }

    try:
        execute_notebook(MAIN, main_output, colab_stub(CONTENT_DIR))
        if not ARTIFACT.exists():
            raise RuntimeError(f"Main notebook did not produce expected artifact: {ARTIFACT}")
        artifact_sha256 = sha256_file(ARTIFACT)
        evidence["outcomes"][MAIN.name] = "PASS"
        evidence["artifactSha256"] = artifact_sha256

        # Build a separate user-style CSV outside the companion notebook execution.
        from sklearn.datasets import load_breast_cancer

        sample = load_breast_cancer(as_frame=True)
        sample.data.tail(8).to_csv(INFERENCE_CSV, index=False)

        companion_prep = colab_stub(CONTENT_DIR) + f'''
from pathlib import Path
import google.colab

_release_uploads = [
    ("{ARTIFACT.name}", Path(r"{ARTIFACT}").read_bytes()),
    ("release_inference.csv", Path(r"{INFERENCE_CSV}").read_bytes()),
]

def _release_upload():
    if not _release_uploads:
        raise RuntimeError("Unexpected additional upload request in artifact-inference notebook")
    name, payload = _release_uploads.pop(0)
    print(f"[release harness] supplying external upload: {{name}}")
    return {{name: payload}}

google.colab.files.upload = _release_upload
'''
        execute_notebook(
            INFERENCE,
            inference_output,
            companion_prep,
            replacements={
                'EXPECTED_ZIP_SHA256 = ""': f'EXPECTED_ZIP_SHA256 = "{artifact_sha256}"'
            },
        )
        evidence["outcomes"][INFERENCE.name] = "PASS"
        evidence["packages"] = package_versions()

        try:
            import torch

            evidence["runner"]["cudaAvailable"] = bool(torch.cuda.is_available())
            evidence["runner"]["torch"] = torch.__version__
        except Exception as exc:  # evidence collection must not hide a successful notebook run
            evidence["runner"]["torchProbeError"] = repr(exc)

        evidence["notes"] = [
            "The default E2E path is exercised without optional gradient fine-tuning.",
            "The artifact-inference notebook consumes the producer artifact in a separate fresh kernel and scores a separate CSV upload.",
            "The harness remaps Colab's /content path to a writable runner directory without changing notebook task semantics.",
            "GPU-only fine-tuning remains an optional branch and is not claimed by this CPU release execution record.",
        ]
        (EVIDENCE_DIR / "notebook-release-evidence.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        print("Notebook release execution: PASS")
    except Exception as exc:
        evidence["failure"] = repr(exc)
        (EVIDENCE_DIR / "notebook-release-evidence.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
