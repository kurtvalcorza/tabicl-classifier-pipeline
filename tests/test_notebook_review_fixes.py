"""Regression tests for the 2026-10-05 notebook review (TIC-M1..M3, TIC-m1..m3; TICA-M1..M3, TICA-m1..m3).

They need only CI's dependencies (no torch, no tabicl, no lightgbm, no checkpoint): they exec the notebooks' own kernel
cells with stand-ins for `run_stage` and `google.colab`, run the stage runners' model-free parts (`data`, `validate`,
the sample-bundle assembly, `rows`, the trusted-digest and binding checks) in-process, and check the generated
notebooks statically. Each test names its finding.
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
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
E2E = ROOT / "tutorials" / "tabiclv2_classifier_colab.ipynb"
AI = ROOT / "tutorials" / "tabiclv2_classifier_artifact_inference_colab.ipynb"
SAMPLE = ROOT / "examples" / "sample-bundle"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("tic_tutorial_stages", TOOLS / "tutorial_stages.py")
AI_STAGES = _load("tic_tutorial_stages_ai", TOOLS / "tutorial_stages_artifact_inference.py")


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
    ns = {"ROOT": tmp_path / "run", "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o))}
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


@pytest.mark.parametrize("path", [E2E, AI], ids=["TIC-M1", "TICA-M2"])
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


@pytest.mark.parametrize("path", [E2E, AI], ids=["TIC-M1", "TICA-M2"])
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
    """TIC-M2: RUN_FINE_TUNING defaults to True; the run is bounded (3 epochs, 300 s, patience 2), selected on the holdout,
    and a CPU runtime records why it was skipped instead of raising."""
    cell = _cell_with(E2E, "RUN_FINE_TUNING = True  # @param")
    assert "run_stage('condition', run_fine_tuning=RUN_FINE_TUNING" in cell
    assert STAGES.FINE_TUNE == {"epochs": 3, "time_limit": 300, "patience": 2, "eval_metric": "accuracy"}
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_condition"):source.index("def stage_report")]
    assert 'opts.get("run_fine_tuning", True)' in body
    assert 'fine_tune["skipped"] = "no CUDA device' in body and "raise RuntimeError('TabICLv2 fine-tuning requires CUDA')" not in body
    assert "time_limit=FINE_TUNE[\"time_limit\"]" in body and "patience=FINE_TUNE[\"patience\"]" in body
    assert 'selection = f"holdout:{metric}"' in body and "MIN_SELECTION_HOLDOUT_ROWS" in body
    run_all = _nb(E2E)["cells"][0]
    assert "bounded to 3 epochs and 5 minutes" in _src(run_all)


# ---------------------------------------------------------------- TIC-M3 / TICA-M3: guided layer

GUIDED = ("## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model → Output", "## Roadmap", "<summary><strong>Glossary</strong>", "Predict before running", "**What to notice:**", "Check your reasoning", "## Troubleshooting", "## Conclusion")


@pytest.mark.parametrize("path", [E2E, AI], ids=["TIC-M3", "TICA-M3"])
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


def test_TICA_M3_every_bundle_refusal_has_a_troubleshooting_row() -> None:
    """TICA-M3: the troubleshooting table names the refusals the artifact stage can raise."""
    md = "\n".join(_src(c) for c in _nb(AI)["cells"] if c["cell_type"] == "markdown")
    table = md[md.index("## Troubleshooting"):]
    for message in ("Trusted digest mismatch", "Unexpected or missing artifact files", "pinned base checkpoint", "non-numeric value"):
        assert message in table, message


@pytest.mark.parametrize("runner", ["tutorial_stages.py", "tutorial_stages_artifact_inference.py"])
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
    assert calls == [("data", {"data_source": "Sample: Breast Cancer", "use_byod": False, "target_column": "target", "paths": {}, "text_columns": []})]


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


def test_M1_sample_bundle_is_reproduced_byte_for_byte() -> None:
    """TICA-M1: tools/build_sample_bundle.py reproduces examples/sample-bundle/ exactly (no model is run)."""
    builder = _load("tic_build_sample_bundle", TOOLS / "build_sample_bundle.py")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert builder.main(["--check"]) == 0


def test_M1_sample_rows_are_the_e2e_new_rows(tmp_path) -> None:
    """TICA-M1 / TIC-S2: the carried rows are the E2E notebook's first eight holdout rows, which the support set does not contain."""
    run = _run(tmp_path / "run", tmp_path / "outputs")
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    enc = run.read_state("encoded.json", "t")
    rows = STAGES.new_rows_frame(run, enc)
    pinned = pd.read_csv(SAMPLE / "new_rows.csv")
    pd.testing.assert_frame_equal(rows.reset_index(drop=True), pinned, check_dtype=False)
    assert not set(pinned["row_id"]) & set(enc["train_row_ids"])


def test_M1_companion_default_path_uses_the_sample_without_upload(tmp_path, monkeypatch) -> None:
    """TICA-M1: all fields at their defaults run the pinned sample bundle and rows; google.colab is never imported."""
    _no_colab(monkeypatch)
    ns, calls = _kernel(tmp_path)
    exec(compile(_cell_with(AI, "ARTIFACT_ZIP_PATH = ''  # @param"), "<s4>", "exec"), ns)
    exec(compile(_cell_with(AI, "NEW_DATA_PATH = ''  # @param"), "<s6>", "exec"), ns)
    assert calls == [("artifact", {"source": "sample", "zip_path": "", "expected_zip_sha256": ""}), ("rows", {"source": "sample", "path": ""})]


def _ai_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    shutil.copytree(SAMPLE, root / "sample-bundle")
    return root


def test_M1_sample_bundle_assembles_with_the_verified_base_checkpoint(tmp_path) -> None:
    """TICA-M1: the sample's artifact.json is checked against its pinned digest, and the base checkpoint becomes best.ckpt."""
    root = _ai_root(tmp_path)
    run = AI_STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {"source": "sample"})
    base = tmp_path / "weights" / "base.ckpt"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_bytes(b"verified base checkpoint stand-in")
    run.write_state("weights.json", {"checkpoint": str(base)})
    P = AI_STAGES.package(root)
    bundle = tmp_path / "bundle"
    trust = AI_STAGES.assemble_sample(run, P, bundle)
    record = json.loads((SAMPLE / "SAMPLE_BUNDLE.json").read_text())
    assert trust["artifact_json_sha256"] == record["artifact_sha256"]
    assert (bundle / "checkpoints" / "best.ckpt").read_bytes() == base.read_bytes()
    assert (bundle / "training_context.parquet").read_bytes() == (SAMPLE / "training_context.parquet").read_bytes()
    manifest = json.loads((bundle / "artifact.json").read_text())
    assert AI_STAGES.check_bundle_identity(P, manifest)["mode"] == "pretrained"
    (root / "sample-bundle" / "artifact.json").write_text("{}")
    with pytest.raises(ValueError, match="Trusted digest mismatch for artifact.json"):
        AI_STAGES.assemble_sample(run, P, tmp_path / "bundle2")


# ---------------------------------------------------------------- TICA-m1: ZIP by path + rows by upload


def test_m1_zip_by_path_with_rows_by_upload_has_no_name_error(tmp_path, monkeypatch) -> None:
    """TICA-m1: ARTIFACT_ZIP_PATH set, NEW_DATA_PATH empty: outside Colab a message naming NEW_DATA_PATH (no NameError);
    in Colab the dialog opens."""
    artifact_cell = _set(_cell_with(AI, "ARTIFACT_ZIP_PATH = ''  # @param"), "ARTIFACT_ZIP_PATH", "/d/bundle.zip")
    rows_cell = _cell_with(AI, "NEW_DATA_PATH = ''  # @param")
    _no_colab(monkeypatch)
    ns, calls = _kernel(tmp_path)
    exec(compile(artifact_cell, "<s4>", "exec"), ns)
    with pytest.raises(RuntimeError, match="set NEW_DATA_PATH"):
        exec(compile(rows_cell, "<s6>", "exec"), ns)
    monkeypatch.undo()
    with _colab([{"rows.csv": b"a\n1\n"}]) as files:
        ns, calls = _kernel(tmp_path)
        exec(compile(artifact_cell, "<s4>", "exec"), ns)
        exec(compile(rows_cell, "<s6>", "exec"), ns)
        assert files.calls == 1 and calls[1][0] == "rows" and calls[1][1]["source"] == "upload"
        assert calls[1][1]["path"].endswith("inputs/rows.csv")


# ---------------------------------------------------------------- TICA-M1 / TICA-m2: trusted digest, checkpoint binding


def test_M1_wrong_trusted_zip_digest_is_refused_before_extraction(tmp_path, monkeypatch) -> None:
    """TICA-M1 (trust): a ZIP whose SHA-256 differs from EXPECTED_ZIP_SHA256 is refused before anything is extracted."""
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="2.14.0", cuda=types.SimpleNamespace(is_available=lambda: False)))
    root = _ai_root(tmp_path)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(SAMPLE / "artifact.json", "artifact.json")
    run = AI_STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {"source": "path", "zip_path": str(archive), "expected_zip_sha256": "0" * 64})
    observed = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=f"EXPECTED_ZIP_SHA256 is {'0' * 64}, the supplied file's SHA-256 is {observed}.*nothing was extracted"):
        _quiet(AI_STAGES.stage_artifact, run)
    assert not (root / "state" / "bundle").exists()
    run.options["expected_zip_sha256"] = "abc"
    with pytest.raises(ValueError, match="must be 64 hexadecimal characters"):
        _quiet(AI_STAGES.stage_artifact, run)


def test_m2_pretrained_bundle_must_carry_the_base_checkpoint(tmp_path) -> None:
    """TICA-m2: a `pretrained` bundle whose checkpoint digest is not BASE_MODEL_SHA256 is refused naming the expected digest."""
    P = AI_STAGES.package(_ai_root(tmp_path))
    manifest = json.loads((SAMPLE / "artifact.json").read_text())
    swapped = json.loads(json.dumps(manifest))
    swapped["digests"]["checkpointSha256"] = "f" * 64
    with pytest.raises(RuntimeError, match=f"must be the pinned base checkpoint with SHA-256 {P.BASE_MODEL_SHA256}"):
        AI_STAGES.check_bundle_identity(P, swapped)
    swapped["mode"] = "fine-tuned"
    assert "EXPECTED_ZIP_SHA256" in AI_STAGES.check_bundle_identity(P, swapped)["checkpoint_binding"]
    other_base = json.loads(json.dumps(manifest))
    other_base["baseModelSha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="not produced on the pinned base checkpoint"):
        AI_STAGES.check_bundle_identity(P, other_base)


# ---------------------------------------------------------------- TICA-m3: numeric features, sample_kind


def _rows_run(tmp_path: Path, art_source: str, options: dict):
    root = _ai_root(tmp_path)
    run = AI_STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", options)
    manifest = json.loads((SAMPLE / "artifact.json").read_text())
    run.write_state("artifact.json", {"source": art_source, "manifest": manifest})
    return run, root


def test_m3_non_numeric_value_in_a_numeric_feature_is_refused_in_the_rows_stage(tmp_path) -> None:
    """TICA-m3: `abc` in `mean radius` stops in the rows stage naming the column and the value."""
    rows = pd.read_csv(SAMPLE / "new_rows.csv").astype({"mean radius": object})
    rows.loc[2, "mean radius"] = "abc"
    path = tmp_path / "mine.csv"
    rows.to_csv(path, index=False)
    run, root = _rows_run(tmp_path, "path", {"source": "path", "path": str(path)})
    with pytest.raises(ValueError, match=r"mine\.csv: numeric feature 'mean radius' has 1 non-numeric value\(s\), e\.g\. \['abc'\]"):
        _quiet(AI_STAGES.stage_rows, run)
    assert not (root / "state" / "rows.parquet").exists()


def test_m3_sample_kind_follows_the_input_source(tmp_path) -> None:
    """TICA-m3: sample rows on the sample bundle are `sample`; the user's rows are `BYOD`; sample rows on a user bundle are refused."""
    run, root = _rows_run(tmp_path, "sample", {"source": "sample"})
    _quiet(AI_STAGES.stage_rows, run)
    state = json.loads((root / "state" / "rows.json").read_text())
    assert (state["sample_kind"], state["rows"], state["extra_columns"]) == ("sample", 8, ["row_id"])
    path = tmp_path / "mine.csv"
    pd.read_csv(SAMPLE / "new_rows.csv").to_csv(path, index=False)
    run.options = {"source": "path", "path": str(path)}
    _quiet(AI_STAGES.stage_rows, run)
    assert json.loads((root / "state" / "rows.json").read_text())["sample_kind"] == "BYOD"
    run.write_state("artifact.json", {"source": "path", "manifest": json.loads((SAMPLE / "artifact.json").read_text())})
    run.options = {"source": "sample"}
    with pytest.raises(ValueError, match="The pinned sample rows match only the pinned sample bundle"):
        _quiet(AI_STAGES.stage_rows, run)
