"""Refusal display (NOTEBOOK_SPEC 3.0 DAT19/DAT20): a refused user input is one `Refused: <file>: <rule>` line.

Offline, with CI's dependencies. The browser kit's refusal runs end to end: the notebook's own `run_stage` starts the
carried stage runner in a real subprocess (this interpreter stands in for the isolated environment), streams its output
into the "cell" as the notebook does, and re-raises the refusal. IPython renders an exception through its
`_render_traceback_` method when it has one, so the cell shows exactly that line.
"""
# ruff: noqa: E501
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tutorials" / "tabiclv2_classifier_colab.ipynb"
_spec = importlib.util.spec_from_file_location("refusal_tutorial_stages", ROOT / "tools" / "tutorial_stages.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _install_cell() -> str:
    cells = ["".join(c["source"]) for c in json.loads(E2E.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    return next(c for c in cells if "def run_stage" in c)


def _kernel(tmp_path: Path) -> tuple[dict, Path]:
    """The notebook's Refused and run_stage, wired to a run directory that carries the real stage runner and package."""
    root = tmp_path / "run"
    (root / "state").mkdir(parents=True)
    shutil.copy(ROOT / "tools" / "tutorial_stages.py", root / "tutorial_stages.py")
    shutil.copytree(ROOT / "src", root / "src")  # a copy, not a symlink: Windows needs no extra privilege
    source = _install_cell()
    block = source[source.index("class Refused(RuntimeError):") : source.index("def load_record(name):")]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    ns = {"ROOT": root, "PYTHON": Path(sys.executable), "OUTPUTS": tmp_path / "outputs", "WEIGHTS": root / "weights", "VENV": tmp_path / "venv",
          "ENV": env, "json": json, "os": os, "subprocess": subprocess, "Path": Path}
    exec(compile(block, "<install cell>", "exec"), ns)  # noqa: S102 - the notebook's own runner
    return ns, root


def _render(ns: dict, stage: str, capsys, **options) -> tuple[str, bool]:
    """What the cell shows: the streamed stage output, then the error as IPython renders it."""
    halted = False
    try:
        ns["run_stage"](stage, **options)
    except Exception as exc:  # noqa: BLE001 - rendered exactly as the kernel would
        halted = True
        render = getattr(exc, "_render_traceback_", None)
        print("\n".join(render()) if render else f"Traceback (most recent call last):\n{type(exc).__name__}: {exc}")
    return capsys.readouterr().out, halted


def _iris(target: str) -> str:
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    frame = data.frame.rename(columns={"sepal length (cm)": "sepal_length", "sepal width (cm)": "sepal_width", "petal length (cm)": "petal_length", "petal width (cm)": "petal_width"})
    frame["target"] = frame["target"].map(dict(enumerate(data.target_names)))
    return frame.rename(columns={"target": target}).to_csv(index=False)


def test_the_browser_kit_refusal_is_one_line_with_no_stage_traceback(tmp_path, capsys) -> None:
    ns, root = _kernel(tmp_path)
    upload = root / "inputs" / "refuse.csv"  # where the notebook saves an upload
    upload.parent.mkdir()
    upload.write_text(_iris("variety"), encoding="utf-8")  # the kit's refuse.csv: the label renamed `variety`
    out, halted = _render(ns, "data", capsys, data_source="Upload CSV", use_byod=True, target_column="species", paths={"csv": str(upload)}, text_columns=[])
    refused = [line for line in out.splitlines() if "Refused:" in line]
    assert len(refused) == 1 and refused[0].startswith("Refused: refuse.csv: the target column 'species' is not in the header"), out
    assert refused[0].endswith("Nothing from the refused input was kept: fix it and re-run the Section 4 cell."), out
    assert "Traceback" not in out and halted, out  # the stage's traceback no longer streams into the cell
    error = json.loads((root / "state" / "data.error.json").read_text(encoding="utf-8"))
    assert error["refusal"] is True and error["name"] == "refuse.csv"


def test_the_browser_kit_accept_upload_passes_the_data_stage(tmp_path, capsys) -> None:
    ns, root = _kernel(tmp_path)
    upload = root / "inputs" / "accept.csv"
    upload.parent.mkdir()
    upload.write_text(_iris("species"), encoding="utf-8")
    out, halted = _render(ns, "data", capsys, data_source="Upload CSV", use_byod=True, target_column="species", paths={"csv": str(upload)}, text_columns=[])
    assert not halted and "Refused" not in out and "'name': 'accept.csv'" in out, out


def test_a_genuine_stage_failure_keeps_its_traceback(tmp_path, capsys) -> None:
    ns, _ = _kernel(tmp_path)
    out, halted = _render(ns, "fresh_test", capsys)  # no earlier stage ran: a genuine failure, not a refusal
    assert halted and "Traceback" in out and "Refused:" not in out, out


@pytest.mark.parametrize(
    "stage, options, exc, expected",
    [
        ("data", {}, ValueError("x"), True),
        ("predict", {"new_data_path": "rows.csv"}, FileNotFoundError("x"), True),
        ("external_artifact", {"source": "own", "zip_path": "mine.zip"}, ValueError("x"), True),
        ("external_artifact", {"source": "own", "zip_path": "mine.zip"}, zipfile.BadZipFile("x"), True),
        ("external_artifact", {"source": "pinned"}, ValueError("x"), False),  # the pinned asset failing is an integrity error
        ("external_predict", {"rows_path": "own_rows.csv"}, ValueError("x"), True),
        ("external_predict", {}, ValueError("x"), False),  # the sample rows failing is genuine
        ("data", {}, RuntimeError("x"), False),  # a bug is never a refusal
        ("condition", {}, ValueError("x"), False),
    ],
)
def test_which_failures_are_refusals(stage: str, options: dict, exc: BaseException, expected: bool) -> None:
    assert S.is_refusal(stage, options, exc) is expected


def test_a_refusal_names_the_file_the_message_starts_with(tmp_path) -> None:
    run = types.SimpleNamespace(state=tmp_path)
    assert S.refusal_name(run, {"paths": {"train": "/x/train.csv", "val": "/x/val.csv"}}, "val.csv: too few rows") == "val.csv"
    assert S.refusal_name(run, {"zip_path": "/x/mine.zip"}, "Trusted digest mismatch") == "mine.zip"
    assert S.refusal_name(run, {}, "anything") == "(no file)"


def test_kernel_side_upload_refusals_name_the_files(tmp_path, monkeypatch) -> None:
    cells = ["".join(c["source"]) for c in json.loads(E2E.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    cell = next(c for c in cells if "def external_upload(what):" in c)
    helper = cell[cell.index("def external_upload(what):") : cell.index("if ARTIFACT_SOURCE ==")]
    ns, _ = _kernel(tmp_path)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"a.zip": b"x", "b.zip": b"y"})
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    exec(compile(helper, "<section 10>", "exec"), ns)  # noqa: S102 - the notebook's own helper
    with pytest.raises(ns["Refused"]) as caught:
        ns["external_upload"]("your artifact ZIP")
    lines = caught.value._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith("Refused: a.zip, b.zip: Expected exactly one file") and "Traceback" not in lines[0]
