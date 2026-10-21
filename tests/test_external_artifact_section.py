"""The E2E notebook's closing external-artifact section (NOTEBOOK_SPEC 3.0 §19.1, XAS1–XAS7, SART1–SART11).

Offline and model-free (CI's light pins only): the pin, the digest refusal before extraction, the SART10 contract check,
that the section's cells use only names the setup cells define (XAS5), and the `outputs/external_artifact/` path (XAS6).
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import builtins
import contextlib
import importlib.util
import inspect
import io
import json
import shutil
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
E2E = ROOT / "tutorials" / "tabiclv2_classifier_colab.ipynb"
MAGIC_ZIP = ROOT / "examples" / "sample-data" / "magic-gamma.zip"
PINNED_URL = "https://github.com/kurtvalcorza/tabicl-classifier-pipeline/releases/download/sample-artifact-magic-v1/tabiclv2_classifier_artifact.zip"
PINNED_SHA256 = "2dc82d55161382602fa7a5d1e217e3a33b6c6188ba6ddf4f7ff6b087bf67114c"
SECTION_START = "## 10. Using an artifact made elsewhere"
SECTION_END = "## 11. Optional activity"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("xas_tutorial_stages", TOOLS / "tutorial_stages.py")


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _cells() -> list[dict]:
    return json.loads(E2E.read_text(encoding="utf-8"))["cells"]


def _section_cells() -> list[dict]:
    cells = _cells()
    start = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _src(c).startswith(SECTION_START))
    end = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _src(c).startswith(SECTION_END))
    return cells[start:end]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    try:
        (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    except OSError:
        shutil.copytree(ROOT / "src", root / "src")
    (root / "inputs").mkdir()
    shutil.copyfile(MAGIC_ZIP, root / "inputs" / "magic-gamma.zip")  # the pinned MAGIC archive, reused by digest (no download)
    return root


def _torch_stub(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="2.14.0", cuda=types.SimpleNamespace(is_available=lambda: False)))


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


# ---------------------------------------------------------------- the pin (SART6, XAS2)


def test_section_pins_the_release_asset_by_url_and_digest() -> None:
    assert STAGES.SAMPLE_ARTIFACT_URL == PINNED_URL
    assert STAGES.SAMPLE_ARTIFACT_SHA256 == PINNED_SHA256
    assert STAGES.SAMPLE_ARTIFACT_PRODUCER["commit"].startswith("383ca04")
    assert STAGES.SAMPLE_ARTIFACT_PRODUCER["release_tag"] == "sample-artifact-magic-v1"


def test_default_path_fetches_the_pin_and_never_this_runs_export(tmp_path, monkeypatch) -> None:
    """XAS2: the default source downloads the pinned URL at the pinned digest; the current run's export is never read."""
    _torch_stub(monkeypatch)
    root = _root(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / f"{STAGES.STEM}_artifact.zip").write_bytes(b"this run's export: must never be opened here")
    seen = []

    def fake_fetch(cache, url, expected):
        seen.append((cache, url, expected))
        raise RuntimeError("stop after the fetch")

    monkeypatch.setattr(STAGES, "fetch_pinned_artifact", fake_fetch)
    run = STAGES.Run(root, outputs, tmp_path / "weights", {})
    with pytest.raises(RuntimeError, match="stop after the fetch"):
        _quiet(STAGES.stage_external_artifact, run)
    assert seen == [(root / "inputs" / "external_artifact.zip", PINNED_URL, PINNED_SHA256)]
    for fn in (STAGES.stage_external_artifact, STAGES.stage_external_reconstruct, STAGES.stage_external_predict, STAGES.external_serving):
        body = inspect.getsource(fn)
        assert "{STEM}_artifact.zip" not in body and "artifact-reload" not in body and "export_reference" not in body, fn.__name__


# ---------------------------------------------------------------- integrity (XAS4, VER6)


def _zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("artifact.json", "{}")
    return path


def test_pinned_digest_mismatch_refuses_before_extraction(tmp_path) -> None:
    archive = _zip(tmp_path / "tampered.zip")
    cache = tmp_path / "cache" / "external_artifact.zip"
    with pytest.raises(ValueError, match=r"Trusted digest mismatch .* the pinned SAMPLE_ARTIFACT_SHA256 is .*nothing was extracted"):
        STAGES.fetch_pinned_artifact(cache, archive.resolve().as_uri(), PINNED_SHA256)
    assert not cache.exists() and not cache.with_name(cache.name + ".part").exists()


def test_pinned_digest_match_is_kept_and_reused(tmp_path) -> None:
    import hashlib

    archive = _zip(tmp_path / "standin.zip")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    cache = tmp_path / "cache" / "external_artifact.zip"
    assert STAGES.fetch_pinned_artifact(cache, archive.resolve().as_uri(), digest) == cache
    archive.unlink()  # a second call reuses the verified cache without the network
    assert STAGES.fetch_pinned_artifact(cache, "file:///nonexistent.zip", digest) == cache


def test_own_artifact_with_a_wrong_digest_is_refused_before_extraction(tmp_path, monkeypatch) -> None:
    _torch_stub(monkeypatch)
    root = _root(tmp_path)
    archive = _zip(tmp_path / "mine.zip")
    run = STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {"source": "own", "zip_path": str(archive), "expected_zip_sha256": "0" * 64})
    with pytest.raises(ValueError, match="EXPECTED_ZIP_SHA256 is 0{64}.*nothing was extracted"):
        _quiet(STAGES.stage_external_artifact, run)
    assert not (root / "state" / "external_bundle").exists()
    assert not (tmp_path / "outputs" / STAGES.EXTERNAL_DIR).exists()


def test_artifact_from_another_dataset_breaks_the_data_contract(tmp_path) -> None:
    """SART10: a hash-valid artifact built on another table (the retired Breast Cancer bundle) is refused."""
    P = STAGES.package(_root(tmp_path))
    run = STAGES.Run(tmp_path / "run", tmp_path / "outputs", tmp_path / "weights", {})
    contract = STAGES.magic_contract(run, P)
    assert contract["targetColumn"] == "target" and contract["classLabels"] == ["g", "h"] and len(contract["featureColumns"]) == 10
    STAGES.check_data_contract({k: contract[k] for k in ("featureColumns", "targetColumn", "classLabels")}, contract)
    with pytest.raises(RuntimeError, match=r"default data contract \(SART10\)"):
        STAGES.check_data_contract({"featureColumns": ["mean radius"], "targetColumn": "target", "classLabels": ["benign", "malignant"]}, contract)


# ---------------------------------------------------------------- XAS5: setup cells and its own cells only


def _defined(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def test_section_cells_reference_only_setup_cell_names() -> None:
    setup = [c for c in _cells() if c["cell_type"] == "code" and c.get("metadata", {}).get("cellView") == "form"]
    assert len(setup) == 4, "the four Infrastructure cells (Sections 1-3)"
    setup_names = set().union(*(_defined(ast.parse(_src(c))) for c in setup))
    section_code = [_src(c) for c in _section_cells() if c["cell_type"] == "code"]
    assert len(section_code) == 3
    own = set().union(*(_defined(ast.parse(s)) for s in section_code))
    used = {n.id for s in section_code for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    foreign = used - own - setup_names - set(dir(builtins))
    assert not foreign, f"the section uses names defined outside the setup cells and itself: {sorted(foreign)}"
    assert {"run_stage", "ROOT", "Path"} <= used & setup_names


def test_section_states_its_dependency_and_its_difference_from_the_reload() -> None:
    md = "\n".join(_src(c) for c in _section_cells() if c["cell_type"] == "markdown")
    assert "**This section needs only the setup cells (Sections 1–3).**" in md
    assert "the reload shows that this run's export survives serialization, and this section shows that an artifact made elsewhere can be validated and served here" in md
    assert "**Trust boundary.**" in md and "outputs/external_artifact/" in md


def test_section_defaults_run_the_pinned_artifact_and_sample_rows_without_upload(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "run_stage": lambda stage, **o: calls.append((stage, o))}
    for source in (_src(c) for c in _section_cells() if c["cell_type"] == "code"):
        exec(compile(source, "<section>", "exec"), ns)
    assert calls == [("external_artifact", {"source": "pinned"}), ("external_reconstruct", {}), ("external_predict", {"rows_path": ""})]


def test_section_runs_last_on_the_default_path_after_the_reload() -> None:
    code = [_src(c) for c in _cells() if c["cell_type"] == "code"]
    reload_at = next(i for i, s in enumerate(code) if "run_stage('reload')" in s)
    section_at = next(i for i, s in enumerate(code) if "run_stage('external_artifact'" in s)
    assert reload_at < section_at
    later = "\n".join(code[section_at + 3:])
    assert "run_stage(" not in later.replace("run_stage('activity'", ""), "only the optional activity may follow the section"
    meta = json.loads(E2E.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["notebook_spec"] == "3.0" and meta["notebook_kind"] == "TUTORIAL"
    assert not list((ROOT / "tutorials").glob("*_artifact_inference_colab.ipynb")), "KND11: no separate artifact-inference notebook"


# ---------------------------------------------------------------- XAS6: the outputs path


class _Serving:
    n_estimators, random_state, device = 8, 42, "cpu"

    def predict_proba(self, X):
        return np.tile([0.75, 0.25], (len(X), 1))

    def predict(self, X):
        return np.array(["g"] * len(X))


def test_predictions_and_provenance_go_to_outputs_external_artifact(tmp_path, monkeypatch) -> None:
    root = _root(tmp_path)
    outputs = tmp_path / "outputs"
    run = STAGES.Run(root, outputs, tmp_path / "weights", {})
    P = STAGES.package(root)
    contract = STAGES.magic_contract(run, P)
    manifest = {"featureColumns": contract["featureColumns"], "targetColumn": "target", "classLabels": ["g", "h"], "inference": {"categoricalEncoders": {}}}
    art = {"source": "pinned", "zip": "tabiclv2_classifier_artifact.zip", "zip_sha256": PINNED_SHA256, "url": PINNED_URL, "trusted_digest": "verified (pinned SAMPLE_ARTIFACT_SHA256)", "mode": "pretrained", "manifest": manifest}
    monkeypatch.setattr(STAGES, "external_serving", lambda _run: (P, art, manifest, _Serving(), ["g", "h"]))
    _quiet(STAGES.stage_external_predict, run)
    assert STAGES.EXTERNAL_DIR == "external_artifact"
    written = sorted(p.relative_to(outputs).as_posix() for p in outputs.rglob("*") if p.is_file())
    assert written == ["external_artifact/tabiclv2_classifier_external_predictions.csv", "external_artifact/tabiclv2_classifier_external_result.json"]
    result = json.loads((outputs / "external_artifact" / "tabiclv2_classifier_external_result.json").read_text())
    assert result["artifact"]["zip_sha256"] == PINNED_SHA256 and result["input"]["kind"] == "sample" and result["input"]["rows"] == 8
    assert [f["verdict"] for f in result["input_manifest"]["findings"] if f["input"] == "missing-column-probe"] == ["rejected"]
