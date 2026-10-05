"""ARTIFACT-INFERENCE companion: shared package keys, the pinned sample bundle it carries, and no self-production.

The PAR1–PAR3 and ST1 checks for the companion run in `tests/test_notebook_parity.py`, which is parametrised over
both templates.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE = _load("notebook_template_artifact_inference").TEMPLATE
PRIMARY = _load("notebook_template").TEMPLATE


def test_companion_shares_the_primary_package_keys() -> None:
    assert TEMPLATE["profile"] == "ARTIFACT-INFERENCE"
    assert TEMPLATE["notebook_name"] != PRIMARY["notebook_name"]
    for key in ("package", "repo_name", "weights_key", "modules", "entry_module", "lock", "managed_python", "uv", "install_flags"):
        assert TEMPLATE.get(key) == PRIMARY.get(key), key


def test_companion_carries_the_pinned_sample_bundle() -> None:
    """TICA-M1: the default path's bundle (without its checkpoint, which is the pinned base) and rows are carried."""
    record = json.loads((ROOT / "examples/sample-bundle/SAMPLE_BUNDLE.json").read_text(encoding="utf-8"))
    carried = set(TEMPLATE["carried_extra"]) | set(TEMPLATE["carried_binary"])
    assert carried == {"sample-bundle/artifact.json", "sample-bundle/new_rows.csv", "sample-bundle/SAMPLE_BUNDLE.json", "sample-bundle/training_context.parquet"}
    for name, facts in record["files"].items():
        data = (ROOT / "examples/sample-bundle" / name).read_bytes()
        assert len(data) == facts["bytes"] and hashlib.sha256(data).hexdigest() == facts["sha256"], name
    assert record["artifact_sha256"] == record["files"]["artifact.json"]["sha256"]
    manifest = json.loads((ROOT / "examples/sample-bundle/artifact.json").read_text(encoding="utf-8"))
    base = json.loads((ROOT / "weights" / TEMPLATE["weights_key"] / "dimer-base-manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "pretrained"
    assert manifest["digests"]["checkpointSha256"] == base["files"][0]["sha256"] == record["checkpoint"]["sha256"]


def test_companion_never_self_produces() -> None:
    runner = (TOOLS / "tutorial_stages_artifact_inference.py").read_text(encoding="utf-8")
    nb = json.loads((ROOT / "tutorials" / TEMPLATE["notebook_name"]).read_text(encoding="utf-8"))
    own = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("dimer", {}).get("embedded_sources"))
    for marker in ("load_breast_cancer(", "load_wine(", "make_archive(", "create_finetuned_classifier(", "fine_tune_classifier("):
        assert marker not in runner and marker not in own, marker
    calls = {n.func.attr for n in ast.walk(ast.parse(runner)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert {"safe_extract_zip", "verify_artifact_bundle", "validate_artifact_runtime", "validate_inputs", "evaluation_report"} <= calls
