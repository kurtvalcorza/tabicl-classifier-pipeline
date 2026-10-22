"""Regression tests for the 2026-10-05 notebook review (TIC-M1..M3, TIC-m1..m3; TICA-M1..M3, TICA-m1..m3).

They need only CI's dependencies (no torch, no tabicl, no lightgbm, no checkpoint): they exec the notebooks' own kernel
cells with stand-ins for `run_stage` and `google.colab`, run the stage runners' model-free parts (`data`, `validate`,
and check the generated notebook statically. The artifact-inference companion they also covered (TICA-*) was folded
into the E2E notebook's external-artifact section under NOTEBOOK_SPEC 3.0; its checks now live in
`tests/test_external_artifact_section.py`. Each test names its finding.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
E2E = ROOT / "tutorials" / "tabiclv2_classifier_colab.ipynb"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("tic_tutorial_stages", TOOLS / "tutorial_stages.py")


def _nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_cells(path: Path) -> list[str]:
    return [_src(c) for c in _nb(path)["cells"] if c["cell_type"] == "code"]


def _cell_with(path: Path, needle: str) -> str:
    found = [s for s in _code_cells(path) if needle in s]
    assert len(found) == 1, needle
    return found[0]


def _set(src: str, name: str, value) -> str:
    out = []
    for line in src.split("\n"):
        if line.startswith(f"{name} = "):
            line = f"{name} = {value!r}" + (line[line.index("  #"):] if "  #" in line else "")
        out.append(line)
    return "\n".join(out)


@contextlib.contextmanager
def _colab(queue):
    files = types.SimpleNamespace(calls=0)

    def upload():
        files.calls += 1
        return queue.pop(0)

    files.upload = upload
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield files
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _no_colab(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)  # `from google.colab import files` -> ImportError


def _kernel(tmp_path: Path) -> tuple[dict, list]:
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "OUTPUTS": tmp_path / "outputs", "Refused": type("Refused", (RuntimeError,), {}), "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o))}
    return ns, calls


def _run(root: Path, outputs: Path, options: dict | None = None, module=STAGES):
    src = root / "src"
    if not src.exists():
        root.mkdir(parents=True, exist_ok=True)
        src.symlink_to(ROOT / "src", target_is_directory=True)
    return module.Run(root, outputs, root / "weights", options or {})


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def _breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.rename(columns={dataset.target.name: "target"})
    frame["target"] = frame["target"].map({0: "malignant", 1: "benign"})
    return frame


# ---------------------------------------------------------------- TIC-M1 / TICA-M2: isolated runtime


@pytest.mark.parametrize("path", [E2E], ids=["TIC-M1"])
def test_m1_no_kernel_install_and_no_restart_instruction(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "Restart the runtime" not in text and "restart the runtime" not in text.replace("no runtime restart", "")
    own = [_src(c) for c in _nb(path)["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]
    for src in own:
        assert not re.search(r"['\"]-m['\"]\s*,\s*['\"]pip['\"]|^\s*[%!]\s*pip\b|['\"]pip install", src, re.M), src[:80]
    install = _cell_with(path, "# @title Infrastructure: build (or reuse) the isolated")
    assert "'--require-hashes'" in install and "--managed-python" in install
    assert "MPLBACKEND='Agg'" in install and "'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'" in install


def _exec_check_cell(ns: dict, path: Path, **overrides) -> None:
    src = _cell_with(path, "# @title Infrastructure: check the runtime")
    src = re.sub(r"'environment': [0-9.]+}", "'environment': 0.0}", src, count=1)  # CI disks are small; the rule is tested, not the size
    src = re.sub(r"\{'weights': max\(0\.0, [0-9.]+", "{'weights': max(0.0, 0.0", src, count=1)
    for name, value in overrides.items():
        src = _set(src, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<check>", "exec"), ns)


@pytest.mark.parametrize("path", [E2E], ids=["TIC-M1"])
def test_m1_section1_is_idempotent(path: Path, tmp_path, monkeypatch) -> None:
    """Re-running the Section 1 cell keeps the run directory, so later cells are not stranded."""
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns, path)
    first = ns["ROOT"]
    (first / "tutorial_stages.py").write_text("# carried")
    _exec_check_cell(ns, path)
    assert ns["ROOT"] == first and (first / "tutorial_stages.py").is_file()
    _exec_check_cell(ns, path, NEW_RUN_DIRECTORY=True)
    assert ns["ROOT"] != first


def test_m1_second_run_all_reuses_the_matching_environment(tmp_path, monkeypatch) -> None:
    """TIC-M1: the venv is keyed on the lock digest; a second exec of the install cell builds nothing."""
    import subprocess as real_subprocess

    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns, E2E)
    ns["ENV_ROOT"] = tmp_path / "uvroot"
    ns["NOTEBOOK_SOURCE"] = {"revision": "test"}
    src = _cell_with(E2E, "# @title Infrastructure: build (or reuse) the isolated")
    version = re.search(r"UV = ENV_ROOT / 'uv-([0-9.]+)'", src).group(1)
    ns["ENV_ROOT"].mkdir()
    uv = ns["ENV_ROOT"] / f"uv-{version}"
    uv.write_bytes(b"uv stand-in")
    (ns["ENV_ROOT"] / f"uv-{version}.sha256").write_text(hashlib.sha256(b"uv stand-in").hexdigest())
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append([str(c) for c in cmd])
        if cmd[1] == "venv":
            python = Path(cmd[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("")
        out = json.dumps({"python": "3.12.12", "torch": "x", "tabicl": "x", "numpy": "x", "pandas": "x", "scikit-learn": "x", "lightgbm": "x", "cuda": True})
        return types.SimpleNamespace(stdout=out + "\n", returncode=0)

    fake = types.SimpleNamespace(run=fake_run, Popen=real_subprocess.Popen, PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT)
    for attempt in range(2):
        ns["subprocess"] = fake
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src.replace("import zipfile\n", "import zipfile\nsubprocess = globals()['subprocess']\n", 1), "<install>", "exec"), ns)
        ns["subprocess"] = fake
        assert ns["environment_reused"] is (attempt == 1)
    builds = [c for c in commands if c[1] in ("venv", "pip")]
    assert len(builds) == 2, builds  # one venv + one install, on the first exec only
    assert ns["VENV"].name == "venv-" + ns["LOCK_SHA256"][:16]


# ---------------------------------------------------------------- TIC-M2: bounded fine-tuning on the default path


def test_M2_fine_tuning_runs_on_the_default_path_bounded_and_cpu_skip_is_recorded() -> None:
    """TIC-M2, superseded by R96-M1/M3 (2026-10-19): RUN_FINE_TUNING defaults to True and runs the pre-registered grid,
    selected on validation; a CPU runtime records why it was skipped instead of raising."""
    cell = _cell_with(E2E, "RUN_FINE_TUNING = True  # @param")
    assert "run_stage('condition', run_fine_tuning=RUN_FINE_TUNING" in cell
    assert not hasattr(STAGES, "FINE_TUNE") and [c["id"] for c in STAGES.GRID] == ["C1", "C2", "C3"]
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("PREREGISTERED: dict"):source.index("def wilson")]
    assert 'opts.get("run_fine_tuning", True)' in body
    assert 'fine_tune["skipped"] = "no CUDA device' in body and "raise RuntimeError('TabICLv2 fine-tuning requires CUDA')" not in body
    assert "time_limit=cfg[\"time_limit\"]" in body and "patience=cfg[\"patience\"]" in body
    assert "MIN_SELECTION_HOLDOUT_ROWS" in body and "select_config(pretrained_holdout, grid)" in body
    run_all = _nb(E2E)["cells"][0]
    assert "pre-registered three-config fine-tuning grid" in _src(run_all)


# ---------------------------------------------------------------- TIC-M3 / TICA-M3: guided layer

GUIDED = ("## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model → Output", "## Roadmap", "<summary><strong>Glossary</strong>", "Predict before running", "**What to notice:**", "Check your reasoning", "## Troubleshooting", "## Conclusion")


@pytest.mark.parametrize("path", [E2E], ids=["TIC-M3"])
def test_m3_guided_layer_and_collapsed_infrastructure(path: Path) -> None:
    nb = _nb(path)
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    for heading in GUIDED:
        assert heading in md, heading
    assert "{{" not in md and "{MODEL_ID}" not in md
    infra = [c for c in nb["cells"] if c["cell_type"] == "code" and _src(c).startswith("# @title Infrastructure:")]
    assert len(infra) == 4 and all(c["metadata"].get("cellView") == "form" for c in infra)
    assert _cell_with(path, "RUN_ACTIVITY = False  # @param")


def test_M3_wine_activity_is_the_guided_comparison_and_writes_only_to_its_directory() -> None:
    """TIC-M3: the DATA_SOURCE comparison is framed as the activity (Wine, 36 holdout rows) and leaves canonical outputs unchanged."""
    cell = _cell_with(E2E, "RUN_ACTIVITY = False  # @param")
    assert "ACTIVITY_DATA_SOURCE = 'Sample: Wine'  # @param" in cell
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_activity"):source.index("STAGES = {")]
    assert 'write_output(f"activity/' in body and "before != after" in body and "one_row_resolution" in body


def test_TICA_M3_every_external_artifact_refusal_has_a_troubleshooting_row() -> None:
    """TICA-M3 (carried into the E2E external-artifact section): the troubleshooting table names its refusals."""
    md = "\n".join(_src(c) for c in _nb(E2E)["cells"] if c["cell_type"] == "markdown")
    table = md[md.index("## Troubleshooting"):]
    for message in ("Trusted digest mismatch", "default data contract (SART10)", "external_artifact.json is missing", "is numeric except for"):
        assert message in table, message


@pytest.mark.parametrize("runner", ["tutorial_stages.py"])
def test_no_quality_assert_in_stage_runners(runner: str) -> None:
    tree = ast.parse((TOOLS / runner).read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]


# ---------------------------------------------------------------- TIC-m1: BYOD changes reported, not hidden


def _csv(tmp_path: Path, name: str, frame: pd.DataFrame) -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_m1_dropped_missing_targets_reach_the_input_manifest(tmp_path) -> None:
    """TIC-m1: a single-CSV upload with 7 blank targets reports 7 in the manifest (it used to report 0)."""
    frame = _breast_cancer().astype({"target": object})
    frame.loc[[3, 10, 50, 60, 70, 80, 90], "target"] = None
    path = _csv(tmp_path, "blank.csv", frame)
    run = _run(tmp_path / "run", tmp_path / "outputs", {"data_source": "Upload CSV", "use_byod": True, "target_column": "target", "paths": {"csv": str(path)}})
    _quiet(STAGES.stage_data, run)
    assert json.loads((run.state / "data.json").read_text())["dropped_missing_target_rows_before_split"] == {"csv": 7}
    _quiet(STAGES.stage_validate, run)
    manifest = json.loads((run.out / "tabiclv2_classifier_input_manifest.json").read_text())
    assert manifest["inputs"][0]["dropped_missing_target_rows_before_split"] == 7
    assert json.loads((run.state / "encoded.json").read_text())["dropped"]["before_split"] == 7


def test_m1_numeric_column_with_a_stray_string_is_refused_naming_column_and_value(tmp_path) -> None:
    """TIC-m1: `mean radius` with one `abc` stops in Section 4 naming the column and the value; TEXT_COLUMNS opts out."""
    frame = _breast_cancer().astype({"mean radius": object})
    frame.loc[5, "mean radius"] = "abc"
    path = _csv(tmp_path, "typo.csv", frame)
    run = _run(tmp_path / "run", tmp_path / "outputs", {"data_source": "Upload CSV", "use_byod": True, "target_column": "target", "paths": {"csv": str(path)}})
    with pytest.raises(ValueError, match=r"typo\.csv: column 'mean radius' is numeric except for 1 value\(s\) \['abc'\]"):
        _quiet(STAGES.stage_data, run)
    run.options["text_columns"] = ["mean radius"]
    _quiet(STAGES.stage_data, run)
    assert json.loads((run.state / "data.json").read_text())["text_columns"] == ["mean radius"]


# ---------------------------------------------------------------- TIC-m2: BYOD by path, TARGET_COLUMN field


def test_m2_target_column_field_and_missing_target_message(tmp_path) -> None:
    """TIC-m2: a CSV labelled `diagnosis` stops naming `target` and the header; TARGET_COLUMN='diagnosis' runs."""
    frame = _breast_cancer().rename(columns={"target": "diagnosis"})
    path = _csv(tmp_path, "diag.csv", frame)
    run = _run(tmp_path / "run", tmp_path / "outputs", {"data_source": "Upload CSV", "use_byod": True, "target_column": "target", "paths": {"csv": str(path)}})
    with pytest.raises(ValueError, match=r"diag\.csv: the target column 'target' is not in the header \['mean radius'.*'diagnosis'\]\. Set TARGET_COLUMN"):
        _quiet(STAGES.stage_data, run)
    run.options["target_column"] = "diagnosis"
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    assert json.loads((run.state / "data.json").read_text())["target"] == "diagnosis"


def test_m2_presplit_byod_by_path_without_colab(tmp_path, monkeypatch) -> None:
    """TIC-m2: the pre-split path fields run outside Colab; an empty path outside Colab, or a cancelled upload, gets a clear message."""
    cell = _cell_with(E2E, "BYOD_TRAIN_PATH = ''  # @param")
    presplit = _set(_set(cell, "USE_BYOD", True), "DATA_SOURCE", "Upload pre-split train/val/test")
    _no_colab(monkeypatch)
    ns, calls = _kernel(tmp_path)
    exec(compile(_set(_set(presplit, "BYOD_TRAIN_PATH", "/d/train.csv"), "BYOD_VAL_PATH", "/d/val.csv"), "<s4>", "exec"), ns)
    assert calls == [("data", {"data_source": "Upload pre-split train/val/test", "use_byod": True, "target_column": "target", "paths": {"train": "/d/train.csv", "val": "/d/val.csv", "test": ""}, "text_columns": []})]
    ns, calls = _kernel(tmp_path)
    with pytest.raises(RuntimeError, match="the upload dialog exists only in Google Colab: set the BYOD path field"):
        exec(compile(presplit, "<s4>", "exec"), ns)
    monkeypatch.undo()
    single = _set(_set(cell, "USE_BYOD", True), "DATA_SOURCE", "Upload CSV")
    with _colab([{}]) as files:
        ns, calls = _kernel(tmp_path)
        with pytest.raises(RuntimeError, match="cancelled or empty"):
            exec(compile(single, "<s4>", "exec"), ns)
        assert files.calls == 1 and calls == []
    ns, calls = _kernel(tmp_path)
    exec(compile(cell, "<s4>", "exec"), ns)  # default path: no colab import, no upload
    assert calls == [("data", {"data_source": "Sample: MAGIC Gamma Telescope", "use_byod": False, "target_column": "target", "paths": {}, "text_columns": []})]


def test_m2_presplit_byod_runs_through_validate(tmp_path) -> None:
    """TIC-m2: pre-split train/val CSVs given by path run through the model-free stages."""
    frame = _breast_cancer()
    train = _csv(tmp_path, "train.csv", frame.iloc[:400])
    val = _csv(tmp_path, "val.csv", frame.iloc[400:])
    run = _run(tmp_path / "run", tmp_path / "outputs", {"data_source": "Upload pre-split train/val/test", "use_byod": True, "target_column": "target", "paths": {"train": str(train), "val": str(val), "test": ""}})
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    data = json.loads((run.state / "data.json").read_text())
    assert (data["sample_kind"], data["train_rows"], data["holdout_rows"], data["test_rows"]) == ("BYOD", 400, 169, 0)
    missing = _run(tmp_path / "run2", tmp_path / "outputs2", {**run.options, "paths": {"train": str(tmp_path / "nope.csv"), "val": str(val)}})
    with pytest.raises(FileNotFoundError, match="BYOD train file .*nope.csv.* does not exist"):
        _quiet(STAGES.stage_data, missing)


# ---------------------------------------------------------------- TIC-m3: stated facts


def test_m3_no_unrecorded_development_numbers_and_validator_difference_is_stated() -> None:
    """TIC-m3: the previous revision's unrecorded 0.974 / 0.956 and the Python 3.11 floor are gone; the model card's
    stricter validator floors are stated as a difference."""
    text = E2E.read_text(encoding="utf-8")
    assert not re.search(r"0\.974(?!\d)|0\.956(?!\d)", text) and "Python 3.11" not in text
    assert "the production DIMER validator described in the model card is stricter" in text


# ---------------------------------------------------------------- TICA-M1: pinned sample bundle, default path
