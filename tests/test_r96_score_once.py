"""Regression tests for the 2026-10-19 row-96 review fixes (R96-M1..M3, R96-m1..m12) and SCORE_ONCE_RULE items 1-8.

CI dependencies only (no torch, no tabicl, no checkpoint): the fresh-test stage runs in-process with the model replaced
by a stand-in that returns fixed probabilities, on the committed MAGIC archive. Each test names its finding.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tutorials" / "tabiclv2_classifier_colab.ipynb"
ZIP = ROOT / "examples" / "sample-data" / "magic-gamma.zip"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load("r96_tutorial_stages", ROOT / "tools" / "tutorial_stages.py")
FIRST, REPRO, EXPLORATORY = S.TEST_LOOKS
RECORD = S.RECORDED_EVALUATION  # the fixture below sets it to None; tests of the record restore it


def _grid() -> list[dict]:
    return [{"id": c["id"], "complete": True, "epochs_run": 3, "val_log_loss_per_epoch": [0.28, 0.29, 0.3], "baseline_val_log_loss": 0.27, "candidate_val": {"log_loss": 0.2634}, "param_delta_l2_best": 0.0, "param_delta_l2_last_epoch": 0.4, "optimizer_steps": 18} for c in S.GRID]


def _setup(tmp_path: Path, **settings_changes) -> S.Run:
    root = tmp_path / "run"
    (root / "inputs").mkdir(parents=True)
    shutil.copyfile(ZIP, root / "inputs" / "magic-gamma.zip")
    run = S.Run(root, tmp_path / "outputs", root / "weights", {})
    run.write_state("data.json", {"source": S.MAGIC_SOURCE, "use_byod": False, "archive_sha256": S.MAGIC_SHA256, "row_ids_sha256": S.MAGIC_ROW_IDS_SHA256})
    ckpt = root / "base.ckpt"
    ckpt.write_bytes(b"pretrained weights stand-in")
    settings = {**json.loads(json.dumps(S.PREREGISTERED)), "device": "Tesla T4", **settings_changes}
    grid = _grid()
    selection = "validation:log_loss; pretrained kept"
    run.write_state("condition.json", {"active_mode": "pretrained", "active_checkpoint": str(ckpt), "active_checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(), "selection": selection, "grid": grid, "grid_hash": S.grid_hash(grid, selection, (settings.get("determinism") or {}).get("sdpa_backend")), "settings": settings, "pretrained_holdout": {"log_loss": 0.2634}, "candidate_holdout": {"log_loss": 0.2634}, "device_name": settings["device"]})
    return run


@pytest.fixture
def scored(monkeypatch):
    calls = []

    def probabilities(run, P, cond, test):
        calls.append(len(test))
        y = test["target"].astype(str).to_numpy()
        p = np.where((y == "g")[:, None], [[0.8, 0.2]], [[0.3, 0.7]])
        return y, p, p, ["g", "h"], 1e-7

    monkeypatch.setattr(S, "fresh_test_probabilities", probabilities)
    monkeypatch.delenv(S.MASK_ENV, raising=False)
    monkeypatch.setattr(S, "RECORDED_EVALUATION", None)
    return calls


def _fresh(run: S.Run) -> Path:
    return run.out / S.FRESH_DIR


def _quiet(fn, *args):
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


# ---------------------------------------------------------------- SCORE_ONCE_RULE items 1, 3, 5, 7 (R96-m8)


def test_first_preregistered_scoring_then_reproduction(tmp_path, scored) -> None:
    run = _setup(tmp_path)
    _quiet(S.stage_fresh_test, run)
    report = json.loads((_fresh(run) / f"{S.STEM}_fresh_test_report.json").read_text(encoding="utf-8"))
    assert report["test_look"] == FIRST and report["settings_differing"] == [] and report["preregistered"] is True
    assert report["n"] == S.MAGIC_FRESH_TEST_ROWS and scored == [S.MAGIC_FRESH_TEST_ROWS]
    assert report["primary"]["verdict"] == "same model (Δ = 0)" and report["primary"]["delta"] == 0.0
    assert report["sensitivity_without_duplicated_rows"]["excluded_rows"] == 4  # 3 duplicated in train, 1 in validation
    _quiet(S.stage_fresh_test, run)
    repro = json.loads((_fresh(run) / f"{S.STEM}_fresh_test_reproduction_1.json").read_text(encoding="utf-8"))
    assert repro["test_look"] == REPRO
    assert json.loads((_fresh(run) / f"{S.STEM}_fresh_test_report.json").read_text(encoding="utf-8"))["test_look"] == FIRST  # never relabelled
    ledger = json.loads((_fresh(run) / "ledger.json").read_text(encoding="utf-8"))["entries"]
    assert [e["test_look"] for e in ledger] == [FIRST, REPRO]


def test_a_recorded_evaluation_makes_every_new_runtime_a_reproduction(tmp_path, scored, monkeypatch) -> None:
    monkeypatch.setattr(S, "RECORDED_EVALUATION", {"commit": "abc", "blob": "def", "date": "2026-10-20"})
    run = _setup(tmp_path)
    _quiet(S.stage_fresh_test, run)
    assert not (_fresh(run) / f"{S.STEM}_fresh_test_report.json").exists()
    assert json.loads((_fresh(run) / f"{S.STEM}_fresh_test_reproduction_1.json").read_text(encoding="utf-8"))["test_look"] == REPRO


@pytest.mark.parametrize("change", [{"grid_epochs_cap": 2}, {"grid_learning_rate": 1e-4}, {"random_state": 7}, {"data_source": "Sample: Breast Cancer"}, {"use_byod": True}, {"run_fine_tuning": False}, {"cpu_grid": True}, {"device": "cpu"}, {"device": "NVIDIA L4"}, {"determinism": None}], ids=lambda c: next(iter(c)))
def test_a_changed_setting_never_reads_the_fresh_test(tmp_path, scored, change) -> None:
    run = _setup(tmp_path, **change)
    _quiet(S.stage_fresh_test, run)
    record = json.loads((_fresh(run) / f"{S.STEM}_fresh_test_exploratory_1.json").read_text(encoding="utf-8"))
    assert record["test_look"] == EXPLORATORY and record["settings_differing"] and scored == []
    assert not (_fresh(run) / "ledger.json").exists()


def test_the_mask_is_reread_on_every_call_and_skips_everything(tmp_path, scored, monkeypatch) -> None:
    run = _setup(tmp_path)
    monkeypatch.setenv(S.MASK_ENV, "1")
    assert S.fresh_test_masked()
    _quiet(S.stage_fresh_test, run)
    assert scored == [] and not _fresh(run).exists()
    monkeypatch.delenv(S.MASK_ENV)
    assert not S.fresh_test_masked()


@pytest.mark.parametrize("damage", ["partial_grid", "grid_hash", "selected_model", "pool_digest"])
def test_a_failed_check_is_not_a_look(tmp_path, scored, damage) -> None:
    run = _setup(tmp_path)
    cond = run.read_state("condition.json", "test")
    if damage == "partial_grid":
        cond["grid"] = cond["grid"][:2]
    elif damage == "grid_hash":
        cond["grid_hash"] = "0" * 64
    elif damage == "selected_model":
        Path(cond["active_checkpoint"]).write_bytes(b"other weights")
    else:
        (run.root / "inputs" / "magic-gamma.zip").write_bytes(b"truncated")
    run.write_state("condition.json", cond)
    with pytest.raises(RuntimeError, match="not a look"):
        _quiet(S.stage_fresh_test, run)
    assert scored == [] and not (_fresh(run) / "ledger.json").exists()
    assert not list(_fresh(run).glob("*_report.json")) if _fresh(run).exists() else True


def test_an_interrupted_report_write_is_not_a_look(tmp_path, scored, monkeypatch) -> None:
    run = _setup(tmp_path)
    real = S.atomic_write_json

    def interrupted(path, value):
        if path.name.endswith("_fresh_test_report.json"):
            raise KeyboardInterrupt
        return real(path, value)

    monkeypatch.setattr(S, "atomic_write_json", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _quiet(S.stage_fresh_test, run)
    assert not (_fresh(run) / "ledger.json").exists() and not list(_fresh(run).glob("*.json"))
    monkeypatch.setattr(S, "atomic_write_json", real)
    _quiet(S.stage_fresh_test, run)
    assert json.loads((_fresh(run) / f"{S.STEM}_fresh_test_report.json").read_text(encoding="utf-8"))["test_look"] == FIRST


@pytest.mark.parametrize("content", ["", "{not json", "[]", '{"entries": 3}'])
def test_a_corrupt_ledger_never_crashes_or_claims_the_first_look(tmp_path, scored, content) -> None:
    run = _setup(tmp_path)
    _fresh(run).mkdir(parents=True)
    (_fresh(run) / "ledger.json").write_text(content, encoding="utf-8")
    _quiet(S.stage_fresh_test, run)
    assert (_fresh(run) / "ledger.corrupt-1.json").exists()
    assert json.loads((_fresh(run) / f"{S.STEM}_fresh_test_reproduction_1.json").read_text(encoding="utf-8"))["test_look"] == REPRO


def test_corrupt_condition_state_names_the_stage_to_rerun(tmp_path, scored) -> None:
    run = _setup(tmp_path)
    (run.state / "condition.json").write_text('{"active_mode": "pre', encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete or corrupt"):
        _quiet(S.stage_fresh_test, run)


# ---------------------------------------------------------------- reading rule and selection (R96-M2, R96-m6)


def _arrays(n=400, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.65, "g", "h")
    base = np.clip(np.where(y == "g", 0.75, 0.3) + rng.normal(0, 0.1, n), 0.02, 0.98)
    other = np.clip(base + (shift if shift else 0.0) * np.where(y == "g", 1, -1), 0.02, 0.98)
    return y, np.c_[base, 1 - base], np.c_[other, 1 - other]


def test_reading_rule_verdicts() -> None:
    y, a, _ = _arrays()
    assert S.fresh_test_reading(y, a, a, ["g", "h"], same_model=True)["primary"]["verdict"] == "same model (Δ = 0)"
    y, a, b = _arrays(shift=0.15)
    better = S.fresh_test_reading(y, a, b, ["g", "h"], same_model=False)
    assert better["primary"]["verdict"] == "improved" and better["primary"]["ci95"][1] < 0
    worse = S.fresh_test_reading(y, b, a, ["g", "h"], same_model=False)
    assert worse["primary"]["verdict"] == "worse" and worse["primary"]["ci95"][0] > 0
    y, a, b = _arrays(shift=0.001)
    assert S.fresh_test_reading(y, a, b, ["g", "h"], same_model=False)["primary"]["verdict"] == "no detectable difference"
    assert S.BOOTSTRAP == {"seed": 20261019, "n_boot": 10000}


def test_selection_keeps_pretrained_unless_beaten_by_the_tolerance() -> None:
    pre = {"log_loss": 0.2634}
    grid = [{"id": "C1", "complete": True, "weights_changed": True, "optimizer_steps": 48, "candidate_val": {"log_loss": 0.2600}}, {"id": "C2", "complete": True, "weights_changed": True, "optimizer_steps": 18, "candidate_val": {"log_loss": 0.2634}}]
    assert S.select_config(pre, grid)[0] is None  # 0.0034 < 0.005
    grid[0]["candidate_val"]["log_loss"] = 0.2500
    assert S.select_config(pre, grid)[0]["id"] == "C1"
    grid.append({"id": "C3", "complete": True, "weights_changed": True, "optimizer_steps": 6, "candidate_val": {"log_loss": 0.25005}})
    assert S.select_config(pre, grid)[0]["id"] == "C3"  # tie within 1e-4 goes to fewer steps


def test_grid_is_the_preregistered_one_and_takes_real_steps() -> None:
    doc = (ROOT / "docs" / "magic-finetune-preregistration.md").read_text(encoding="utf-8")
    assert [(c["id"], c["learning_rate"], c["epochs"], c["patience"]) for c in S.GRID] == [("C1", 1e-5, 30, 8), ("C2", 5e-5, 10, 3), ("C3", 1e-6, 10, 3)]
    assert S.GRID_COMMON["max_data_size"] == 500 and S.GRID_COMMON["use_lr_scheduler"] is False and S.GRID_COMMON["eval_metric"] == "log_loss"
    assert S.SELECTION_TOLERANCE == 0.005 and "more than 0.005" in doc and "**`max_data_size` 500**" in doc
    assert S.DETERMINISM == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "cudnn_deterministic": True, "cudnn_benchmark": False, "deterministic_algorithms": True, "warn_only": False, "seed": 42, "sdpa_backend": "MATH", "flash_attn3": False}
    assert S.MAGIC_ROW_IDS_SHA256["test"] == "8847cab4e87c8ff30dd6a27045464f7891b2e14dd39149c5e5648348f190f3e4"
    assert hashlib.sha256(ZIP.read_bytes()).hexdigest() == S.MAGIC_SHA256
    assert S.MAGIC_COMMIT in S.MAGIC_URL and S.RECORDED_EVALUATION is not None
    body = (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")
    assert "warn_only=True" not in body and "use_lr_scheduler=c[\"use_lr_scheduler\"]" in body


def test_settings_differing_covers_every_preregistered_key() -> None:
    base = {**json.loads(json.dumps(S.PREREGISTERED)), "device": "Tesla T4"}
    assert S.settings_differing(base) == []
    for key in S.PREREGISTERED:
        changed = dict(base, **{key: "changed"})
        assert key in S.settings_differing(changed), key


# ---------------------------------------------------------------- notebook: mask, refusals, m10, REL13/REL14, spec 2.5


def _install_cell() -> str:
    cells = ["".join(c["source"]) for c in json.loads(E2E.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    return next(c for c in cells if "def run_stage" in c)


def _run_stage_namespace(tmp_path, env_at_install: dict):
    source = _install_cell()
    start = source.index("class Refused(RuntimeError):")
    end = source.index("def load_record(name):")
    captured = {}

    class Popen:
        def __init__(self, command, env, **kwargs):
            captured["env"] = env
            self.stdout, self.returncode = iter(()), 1

        def wait(self):
            return 1

    import os as real_os

    root = tmp_path / "run"
    (root / "state").mkdir(parents=True)
    (root / "tutorial_stages.py").write_text("", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    ns = {"ROOT": root, "PYTHON": python, "OUTPUTS": tmp_path / "out", "WEIGHTS": tmp_path / "w", "VENV": tmp_path / "venv", "ENV": dict(env_at_install), "json": json, "os": real_os, "subprocess": types.SimpleNamespace(Popen=Popen, PIPE=-1, STDOUT=-2), "Path": Path}
    exec(source[start:end], ns)
    return ns, captured, root


def test_run_stage_rereads_the_mask_on_every_call(tmp_path, monkeypatch) -> None:
    ns, captured, _ = _run_stage_namespace(tmp_path, {"DIMER_MASK_FRESH_TEST": "1"})  # masked when the environment was built
    monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    with pytest.raises(RuntimeError):
        ns["run_stage"]("fresh_test")
    assert captured["env"]["DIMER_MASK_FRESH_TEST"] == ""  # unmasked now: the install-time copy does not leak
    monkeypatch.setenv("DIMER_MASK_FRESH_TEST", "1")
    with pytest.raises(RuntimeError):
        ns["run_stage"]("fresh_test")
    assert captured["env"]["DIMER_MASK_FRESH_TEST"] == "1" and captured["env"]["DIMER_ISOLATED_ENV"]


def test_a_refusal_is_one_message_without_a_traceback(tmp_path) -> None:
    ns, _, root = _run_stage_namespace(tmp_path, {})
    error = root / "state" / "data.error.json"

    def fail(*args, **kwargs):  # the stage writes its error file, then exits 1
        error.write_text(json.dumps({"type": "ValueError", "message": "train.csv: the target column 'y' is not in the header.", "refusal": True, "name": "train.csv"}), encoding="utf-8")
        return types.SimpleNamespace(stdout=iter(()), returncode=1, wait=lambda: 1)

    ns["subprocess"].Popen = fail
    with pytest.raises(ns["Refused"]) as caught:
        ns["run_stage"]("data")
    rendered = caught.value._render_traceback_()
    assert rendered == ["Refused: train.csv: the target column 'y' is not in the header. Nothing from the refused input was kept: fix it and re-run the Section 4 cell."]
    assert {"data", "validate", "predict"} == S.REFUSAL_STAGES


def test_probabilities_are_renormalised_and_the_deviation_is_bounded() -> None:
    import tabicl_classifier_pipeline as P

    proba, deviation = P.renormalise_probabilities([[0.6, 0.4000002], [0.3, 0.7]])
    assert np.allclose(proba.sum(axis=1), 1.0) and 0 < deviation < 1e-6
    with pytest.raises(RuntimeError, match="do not sum to one"):
        P.renormalise_probabilities([[0.6, 0.5]])


def test_notebook_text_is_labelled_and_current() -> None:
    nb = json.loads(E2E.read_text(encoding="utf-8"))
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert nb["metadata"]["dimer"]["notebook_spec"] == "3.0" and nb["metadata"]["dimer"]["notebook_kind"] == "TUTORIAL"
    assert nb["cells"][1]["cell_type"] == "markdown" and "".join(nb["cells"][1]["source"]).startswith("> **⚠️ Notice.**")
    for stale in ("No hosted run of this revision", "No recorded run of this revision", "the pipeline's production adaptation", "three epochs on the T4", "scored once"):
        assert stale not in text, stale  # R96-m1, R96-M1/M3, SCORE_ONCE_RULE item 6
    for quoted in ("0.9825", "0.9649", "0.9561", "0.2634"):  # R96-m2: every quoted number carries its run label
        for match in re.finditer(re.escape(quoted), text):
            window = text[max(0, match.start() - 400): match.end() + 200]
            assert "2026-10-10" in window or "2026-10-19" in window or "2026-10-20" in window, quoted
    for quoted in ("0.2628", "0.8993", "0.8831", "0.0006"):  # the hosted run of record's numbers carry its date
        assert quoted in text, quoted
        for match in re.finditer(re.escape(quoted), text):
            assert "2026-10-20" in text[max(0, match.start() - 600): match.end() + 200], quoted
    assert "CPU development run" in text and "hosted run of record" in text  # the two runs stay distinct
    assert "fine-tuning a strong in-context model often does not help" in text.lower()
    assert "[1] J. Qu et al." in text and "doi: 10.24432/C52C8B" in text  # IEEE-style references (s1)


def test_model_card_agrees_with_the_default() -> None:
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "MAGIC Gamma Telescope" in card and "does not improve on the in-context baseline" not in card  # R96-m11


# ---------------------------------------------------------------- provenance (R96-m3)


def test_stamp_names_a_commit_that_contains_every_carried_file() -> None:
    meta = json.loads(E2E.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]
    build = _load("r96_build_notebook", ROOT / "tools" / "build_notebook.py")
    sources = build.carried_sources(ROOT, build.load_template(ROOT / "tools" / "notebook_template.py"))
    revision = meta["revision"]
    found = subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=ROOT, capture_output=True)
    assert found.returncode == 0, f"stamped revision {revision} is not in this clone (CI checks out full history)"
    for dest, path in sources.items():
        blob = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=ROOT, check=True, capture_output=True).stdout
        assert hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() == meta["files"][dest], (dest, path, revision)


# ---------------------------------------------------------------- round 2 (peer review F1-F6; amendment 1)


def test_F5_a_restored_config_is_never_selected_as_fine_tuned() -> None:
    pre = {"log_loss": 0.2634}
    restored = {"id": "C1", "complete": True, "weights_changed": False, "optimizer_steps": 48, "candidate_val": {"log_loss": 0.2570}}
    chosen, basis = S.select_config(pre, [restored])
    assert chosen is None and "pretrained kept" in basis


class _Optimizer:
    def __init__(self, counter):
        self.counter = counter

    def step(self):
        self.counter.on_optimizer_step(self)  # what the global step-post hook does after a real step


class _Scaler:
    """GradScaler stand-in: skips optimizer.step() when the scaled gradients overflowed (inf/NaN)."""

    def __init__(self, overflow_on):
        self.overflow_on, self.calls = set(overflow_on), 0

    def step(self, optimizer):
        self.calls += 1
        if self.calls not in self.overflow_on:
            optimizer.step()


def test_F2_steps_are_counted_and_a_skipped_amp_step_is_reported() -> None:
    counter = S.StepCounter()
    scaler = _Scaler(overflow_on={1, 4})
    finetuner = type("F", (), {})()
    finetuner._make_amp = lambda device: (True, scaler, None)
    counter.instrument(finetuner)
    _, wrapped, _ = finetuner._make_amp("cuda")
    optimizer = _Optimizer(counter)
    for _ in range(6):
        wrapped.step(optimizer)
    assert counter.record(planned=6) == {"optimizer_steps_planned": 6, "optimizer_steps_attempted": 6, "optimizer_steps_taken": 4, "amp_skipped_steps": 2}
    body = (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")
    assert '"optimizer_steps": counter.taken' in body and "register_optimizer_step_post_hook(counter.on_optimizer_step)" in body
    assert 'key=lambda g: (g["optimizer_steps"], g["candidate_val"]["log_loss"])' in body  # tie rule uses the taken count


def test_F3_deleting_outputs_does_not_reset_the_first_look(tmp_path, scored) -> None:
    run = _setup(tmp_path)
    _quiet(S.stage_fresh_test, run)
    assert (_fresh(run) / f"{S.STEM}_fresh_test_report.json").exists() and S.look_marker(run).exists()
    shutil.rmtree(run.out)
    _quiet(S.stage_fresh_test, run)
    assert not (_fresh(run) / f"{S.STEM}_fresh_test_report.json").exists()
    assert json.loads((_fresh(run) / f"{S.STEM}_fresh_test_reproduction_1.json").read_text(encoding="utf-8"))["test_look"] == REPRO


def test_F6_negative_probability_entries_are_refused() -> None:
    import tabicl_classifier_pipeline as P

    with pytest.raises(RuntimeError, match="negative or undefined"):
        P.renormalise_probabilities([[1.5, -0.5]])
    with pytest.raises(RuntimeError):
        P.renormalise_probabilities([[float("nan"), 1.0]])


def test_F1_quoted_grid_numbers_come_from_the_committed_cpu_record() -> None:
    nb = json.loads(E2E.read_text(encoding="utf-8"))
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    row = next(line for line in record.splitlines() if line.startswith("| 2026-10-19 |") and "11cb0310" in line)
    for number in ("0.2634", "0.903", "0.2701", "0.2712–0.2762", "0.2864 / 0.3340 / 0.3027", "0.2702 / 0.2701 / 0.2700"):
        assert number in text and number in row, number
    assert "0.29–0.36" not in text and "every epoch *worse*" not in text and "every epoch scored worse" not in text
    assert "11cb0310" in text and "CPU development run" in text
    assert "suppose a config reports `weights_changed: False`" in text and "`weights_changed: True`" in text  # conditional, both outcomes


# ---------------------------------------------------------------- round 3 (amendment 2): MATH SDPA backend, warnings surfaced


@pytest.fixture
def fake_torch(monkeypatch):
    """Stand-ins for torch and tabicl's attention module (CI has neither): they record whether the MATH-only SDPA context
    and tabicl's FlashAttention-3 switch are active, as the real ones set global backend flags."""
    import contextlib
    import sys

    state = {"flash": True, "mem": True, "math": True, "cudnn": False, "det": False, "math_ctx": False}

    @contextlib.contextmanager
    def sdpa_kernel(backend):
        saved = dict(state)
        state.update(flash=False, mem=False, cudnn=False, math=(backend == "MATH"), math_ctx=True)
        try:
            yield
        finally:
            state.update(saved)

    attention = types.ModuleType("tabicl._model.attention")
    attention._use_flash_attn3 = True
    attention.HAS_FLASH_ATTN3 = True

    @contextlib.contextmanager
    def flash_attn3_toggle(enabled):
        old = attention._use_flash_attn3
        attention._use_flash_attn3 = enabled
        try:
            yield
        finally:
            attention._use_flash_attn3 = old

    attention.flash_attn3_toggle = flash_attn3_toggle
    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=True),
        cuda=types.SimpleNamespace(flash_sdp_enabled=lambda: state["flash"], mem_efficient_sdp_enabled=lambda: state["mem"], math_sdp_enabled=lambda: state["math"], cudnn_sdp_enabled=lambda: state["cudnn"]),
    )
    torch.use_deterministic_algorithms = lambda value: state.update(det=value)
    torch.are_deterministic_algorithms_enabled = lambda: state["det"]
    torch.is_deterministic_algorithms_warn_only_enabled = lambda: False
    torch.manual_seed = lambda seed: None
    nn = types.ModuleType("torch.nn")
    nn_attention = types.ModuleType("torch.nn.attention")
    nn_attention.SDPBackend = types.SimpleNamespace(MATH="MATH")
    nn_attention.sdpa_kernel = sdpa_kernel
    for name, module in {"torch": torch, "torch.nn": nn, "torch.nn.attention": nn_attention, "tabicl": types.ModuleType("tabicl"), "tabicl._model": types.ModuleType("tabicl._model"), "tabicl._model.attention": attention}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state, attention


@pytest.mark.parametrize("stage", ["condition", "fresh_test", "report", "predict", "export", "reload"])
def test_A2_every_model_stage_runs_inside_the_math_backend(tmp_path, fake_torch, monkeypatch, stage) -> None:
    state, attention = fake_torch
    seen = {}

    def fake_stage(run):  # what fine-tuning and every predict_proba inside the stage would see
        seen.update(math_ctx=state["math_ctx"], flash3=attention._use_flash_attn3 or attention.HAS_FLASH_ATTN3, det=state["det"], backend=S.attention_backend())

    monkeypatch.setitem(S.STAGES, stage, fake_stage)
    assert _quiet(S.main, ["--root", str(tmp_path / "run"), "--outputs", str(tmp_path / "out"), "--weights", str(tmp_path / "w"), "--stage", stage]) == 0
    assert seen == {"math_ctx": True, "flash3": False, "det": True, "backend": {"sdpa_backend": "MATH", "flash_attn3": False}}
    assert state["math_ctx"] is False and attention._use_flash_attn3 is True  # restored after the stage
    assert stage in S.MODEL_STAGES and "data" not in S.MODEL_STAGES


def test_A2_sdpa_backend_is_recorded_and_preregistered(fake_torch) -> None:
    with S.deterministic_kernels():
        record = S.apply_determinism()
    assert record["sdpa_backend"] == "MATH" and record["flash_attn3"] is False
    assert {k: record[k] for k in S.DETERMINISM if k != "CUBLAS_WORKSPACE_CONFIG"} == {k: v for k, v in S.DETERMINISM.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    base = {**json.loads(json.dumps(S.PREREGISTERED)), "device": "Tesla T4"}
    assert S.settings_differing(base) == []
    for other in ("FLASH+MATH+MEM_EFFICIENT", "MEM_EFFICIENT"):
        changed = json.loads(json.dumps(base))
        changed["determinism"]["sdpa_backend"] = other
        assert "determinism" in S.settings_differing(changed)
    grid = _grid()
    assert S.grid_hash(grid, "s", "MATH") != S.grid_hash(grid, "s", "MEM_EFFICIENT")  # the backend is a grid-hash input


def test_A2_a_determinism_warning_is_surfaced(tmp_path, fake_torch, monkeypatch, capsys) -> None:
    import warnings

    def warns(run):
        warnings.warn("Memory Efficient attention defaults to a non-deterministic algorithm.", UserWarning, stacklevel=1)
        warns.during = S.determinism_warnings()

    monkeypatch.setitem(S.STAGES, "condition", warns)
    S.main(["--root", str(tmp_path / "run"), "--outputs", str(tmp_path / "out"), "--weights", str(tmp_path / "w"), "--stage", "condition"])
    out = capsys.readouterr().out
    assert "DETERMINISM WARNING: UserWarning: Memory Efficient attention defaults to a non-deterministic algorithm." in out
    assert warns.during == ["Memory Efficient attention defaults to a non-deterministic algorithm."]


# ---------------------------------------------------------------- deterministic artifact ZIP (2026-10-20 record finding)


def test_the_artifact_zip_digest_depends_only_on_member_names_and_bytes(tmp_path) -> None:
    import os
    import zipfile

    digests = []
    for i, mtime in enumerate((1_000_000_000, 1_700_000_000)):
        art = tmp_path / f"art{i}"
        (art / "checkpoints").mkdir(parents=True)
        files = {"artifact.json": b"{}", "training_context.parquet": b"ctx", "checkpoints/best.ckpt": b"w" * 4096}
        for name, data in (files.items() if i == 0 else reversed(list(files.items()))):
            (art / name).write_bytes(data)
            os.utime(art / name, (mtime, mtime))
        archive = S.deterministic_zip(art, tmp_path / f"a{i}.zip")
        digests.append(hashlib.sha256(archive.read_bytes()).hexdigest())
        with zipfile.ZipFile(archive) as zf:
            assert zf.namelist() == sorted(files) and all(info.date_time == S.ZIP_DATE_TIME for info in zf.infolist())
    assert digests[0] == digests[1]
    body = (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")
    assert "make_archive(" not in body


# ---------------------------------------------------------------- evaluation of record (2026-10-20, hosted T4, 9ea0888)


def test_the_evaluation_of_record_is_the_hosted_t4_run_and_keeps_pretrained() -> None:
    rec = S.RECORDED_EVALUATION
    assert rec["commit"] == "9ea0888d9df24a9568613ec9c51dd1fe65cea443" and rec["notebook_blob"] == "947feb162abbf8126a8579188268024da3ca3ba8"
    assert rec["date"] == "2026-10-20" and rec["device"] == "Tesla T4" and rec["label"] == FIRST
    assert rec["grid_hash"] == "4b1bf5366b121dd2365c6c9b02408e1b825c247bb6636fee6d8fe35697b20692"
    assert S.RECORDED_SELECTION == "pretrained" and rec["selected"] == "pretrained"
    r = rec["reading"]
    assert r["primary.delta"] == 0.0 and r["primary.ci95"] == [0.0, 0.0] and r["primary.verdict"] == "same model (Δ = 0)"
    assert r["accuracy.pretrained"] == 0.8993 and r["accuracy.pretrained_wilson95"] == [0.8831, 0.9136] and r["roc_auc.selected"] == 0.95
    assert r["sensitivity.excluded_rows"] == 4 and r["sensitivity.n"] == 1496 and r["sensitivity.primary.verdict"] == "same model (Δ = 0)"
    evidence = ROOT / rec["evidence"]
    transcription = json.loads((evidence / "fresh_test_report_transcription.json").read_text(encoding="utf-8"))
    printed = transcription["printed_reading"]
    assert transcription["commit"] == rec["commit"] and transcription["grid_hash"] == rec["grid_hash"]
    assert printed["primary"]["delta"] == r["primary.delta"] and printed["primary"]["verdict"] == r["primary.verdict"]
    assert printed["accuracy"]["pretrained"] == r["accuracy.pretrained"] and printed["roc_auc"]["pretrained"] == r["roc_auc.pretrained"]
    assert printed["sensitivity_without_duplicated_rows"]["n"] == r["sensitivity.n"]
    assert (evidence / "tabiclv2_classifier_colab_9ea0888_colab-cli-t4.ipynb").is_file()


def test_a_new_runtime_is_a_reproduction_and_prints_matches_record(tmp_path, scored, monkeypatch, capsys) -> None:
    run = _setup(tmp_path)
    # The stand-in grid has its own (pre-checked) hash; give the record that hash so only the reading can differ.
    monkeypatch.setattr(S, "RECORDED_EVALUATION", {**RECORD, "grid_hash": run.read_state("condition.json", "test")["grid_hash"]})
    S.stage_fresh_test(run)
    out = capsys.readouterr().out
    assert not (_fresh(run) / f"{S.STEM}_fresh_test_report.json").exists()
    repro = json.loads((_fresh(run) / f"{S.STEM}_fresh_test_reproduction_1.json").read_text(encoding="utf-8"))
    assert repro["test_look"] == REPRO and "'matches_record'" in out
    comparison = repro["comparison_with_record"]
    assert comparison["reproduction_of"] == "2026-10-20 hosted T4, commit 9ea0888"
    # The stand-in's probabilities are not the model's, so its accuracy and AUC differ from the record and are named; the
    # selection, device and the "same model" reading match; the grid hash is never a difference (issue #28).
    assert comparison["matches_record"] is False
    assert {"selected", "grid_hash", "device", "reading.primary.delta", "reading.primary.verdict", "reading.sensitivity.n"}.isdisjoint(comparison["differs"])
    assert "reading.accuracy.pretrained" in comparison["differs"]


def test_compare_with_record_matches_the_recorded_values_and_names_each_difference(monkeypatch) -> None:
    rec = S.RECORDED_EVALUATION
    runner = rec["stage_runner_sha256"]
    same = S.compare_with_record("pretrained", rec["grid_hash"], "Tesla T4", dict(rec["reading"]), runner)
    assert same == {"reproduction_of": "2026-10-20 hosted T4, commit 9ea0888", "stage_runner_sha256": {"record": runner, "this_run": runner}, "same_stage_runner": True, "grid_hash": {"informational": S.GRID_HASH_NOTE, "record": rec["grid_hash"], "this_run": rec["grid_hash"], "equal": True}, "matches_record": True, "differs": []}
    changed = S.compare_with_record("fine-tuned", "0" * 64, "Tesla T4", {**rec["reading"], "primary.delta": -0.01}, runner)
    assert changed["matches_record"] is False and changed["differs"] == ["selected", "reading.primary.delta"]
    monkeypatch.setattr(S, "RECORDED_EVALUATION", None)
    assert S.compare_with_record("pretrained", rec["grid_hash"], "Tesla T4", {}) is None


# ---------------------------------------------------------------- issue #28: the grid hash is reported, never compared

OTHER_RUNNER = "6b64c59461035d497d164d8f6259c3673f5a6463c24f26cdf8a9db3de8b9e478"  # tools/tutorial_stages.py at d7426dc
OTHER_GRID = "2aa9e0cd" + "0" * 56


def test_the_record_names_the_stage_runner_that_produced_it() -> None:
    # Derived from git: sha256 of tools/tutorial_stages.py (LF) at 15c1ae4 / 9ea0888, which blob 947feb16 carries and stamps in
    # NOTEBOOK_SOURCE['files']['tutorial_stages.py'].
    assert S.RECORDED_EVALUATION["stage_runner_sha256"] == "cc604098aa235653c2ef5b36cffbca37332539a9ac74e1b52912e1ff71184823"


def test_stage_runner_sha256_is_the_digest_the_notebook_stamps() -> None:
    build = _load("r28_build_notebook", ROOT / "tools" / "build_notebook.py")
    assert S.stage_runner_sha256() == build.sha256_text(build._text(ROOT / "tools" / "tutorial_stages.py"))
    # This revision carries the record, so it is never the runner that produced it: its grid hash is reported as not comparable.
    assert S.stage_runner_sha256() != S.RECORDED_EVALUATION["stage_runner_sha256"]


def test_same_runner_with_a_different_grid_hash_and_equal_results_matches() -> None:
    # The d4172f5 and edbe94d hosted T4 runs: same notebook blob, same stage runner, grid hashes 2aa9e0cd... and 4b1bf536...,
    # identical selection and fresh-test values. That is a match.
    rec = S.RECORDED_EVALUATION
    out = S.compare_with_record("pretrained", OTHER_GRID, "Tesla T4", dict(rec["reading"]), rec["stage_runner_sha256"])
    assert out["same_stage_runner"] is True
    assert out["matches_record"] is True and out["differs"] == []
    assert out["grid_hash"] == {"informational": S.GRID_HASH_NOTE, "record": rec["grid_hash"], "this_run": OTHER_GRID, "equal": False}


def test_other_runner_with_a_different_grid_hash_and_equal_results_matches() -> None:
    rec = S.RECORDED_EVALUATION
    out = S.compare_with_record("pretrained", OTHER_GRID, "Tesla T4", dict(rec["reading"]), OTHER_RUNNER)
    assert out["matches_record"] is True and out["differs"] == [] and out["same_stage_runner"] is False
    assert out["grid_hash"]["equal"] is False and out["grid_hash"]["this_run"] == OTHER_GRID
    assert out["stage_runner_sha256"] == {"record": rec["stage_runner_sha256"], "this_run": OTHER_RUNNER}


@pytest.mark.parametrize("runner", ["record", "other"])
@pytest.mark.parametrize("grid", ["record", "other"])
def test_a_different_fresh_test_value_fails_whatever_the_runner_and_grid_hash(runner, grid) -> None:
    rec = S.RECORDED_EVALUATION
    ghash = rec["grid_hash"] if grid == "record" else OTHER_GRID
    rsha = rec["stage_runner_sha256"] if runner == "record" else OTHER_RUNNER
    out = S.compare_with_record("pretrained", ghash, "Tesla T4", {**rec["reading"], "accuracy.pretrained": 0.8987}, rsha)
    assert out["matches_record"] is False and out["differs"] == ["reading.accuracy.pretrained"]


def test_the_grid_hash_is_labelled_informational_in_the_output() -> None:
    rec = S.RECORDED_EVALUATION
    out = S.compare_with_record("pretrained", OTHER_GRID, "Tesla T4", dict(rec["reading"]), rec["stage_runner_sha256"])
    assert out["grid_hash"]["informational"].startswith("informational only, not compared")
    assert "issue #28" in out["grid_hash"]["informational"]
    assert "informational" in repr(out)  # the stage prints this dict as is
    assert not any("grid" in d for d in out["differs"])


def test_the_condition_stage_prints_the_grid_hash_as_informational() -> None:
    src = (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")
    assert 'print("grid hash (informational' in src
    assert "digit for digit" not in src and "bit-exact" not in src


def test_other_runner_with_a_different_selection_differs() -> None:
    rec = S.RECORDED_EVALUATION
    out = S.compare_with_record("fine-tuned", OTHER_GRID, "Tesla T4", dict(rec["reading"]), OTHER_RUNNER)
    assert out["matches_record"] is False and out["differs"] == ["selected"]


def test_without_an_explicit_runner_the_running_file_is_used() -> None:
    rec = S.RECORDED_EVALUATION
    out = S.compare_with_record("pretrained", rec["grid_hash"], "Tesla T4", dict(rec["reading"]))
    assert out["stage_runner_sha256"]["this_run"] == S.stage_runner_sha256()
