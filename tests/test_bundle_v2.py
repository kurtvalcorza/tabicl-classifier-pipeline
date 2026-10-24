# ruff: noqa: E501  -- single-line test fixtures and assertions are kept readable
"""Serving bundle v2: ``checkpoints/best.safetensors``; v1 ``checkpoints/best.ckpt`` bundles stay readable (deprecated).

The offline tests need no model library. The model-backed tests build a tiny TabICL network, write it in the layout
tabicl 2.1.1's fine-tuning loop uses for ``best.ckpt`` (``config``, ``state_dict``, optimizer and scheduler state), and
check the conversion, the equivalence of the two loads and both bundle layouts; they skip without ``tabicl``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from tabicl_classifier_pipeline import api as P

ROOT = Path(__file__).resolve().parents[1]
TINY_CONFIG = {"max_classes": 10, "embed_dim": 16, "col_num_blocks": 1, "col_nhead": 2, "col_num_inds": 8, "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2, "icl_num_blocks": 1, "icl_nhead": 2}


def _stages():
    spec = importlib.util.spec_from_file_location("bundle_v2_tutorial_stages", ROOT / "tools" / "tutorial_stages.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(fmt: str, checkpoint: str, **extra) -> dict:
    return {"artifactFormat": fmt, "checkpoint": checkpoint, "inference": {"modelPath": checkpoint}, "payloadFiles": [checkpoint, "training_context.parquet"], **extra}


# ---------------------------------------------------------------- offline: the format contract


def test_the_v2_contract_names_best_safetensors_and_keeps_v1_readable() -> None:
    assert P.ARTIFACT_FORMAT == "tabicl-dimer-classifier-v2" and P.LEGACY_ARTIFACT_FORMAT == "tabicl-dimer-classifier-v1"
    assert P.BUNDLE_CHECKPOINT == "checkpoints/best.safetensors" and P.LEGACY_BUNDLE_CHECKPOINT == "checkpoints/best.ckpt"
    assert P.SUPPORTED_ARTIFACT_FORMATS == (P.ARTIFACT_FORMAT, P.LEGACY_ARTIFACT_FORMAT)
    assert P.check_bundle_layout(_manifest(P.ARTIFACT_FORMAT, P.BUNDLE_CHECKPOINT)) == P.ARTIFACT_FORMAT
    assert P.check_bundle_layout(_manifest(P.LEGACY_ARTIFACT_FORMAT, P.LEGACY_BUNDLE_CHECKPOINT)) == P.LEGACY_ARTIFACT_FORMAT


@pytest.mark.parametrize(
    "manifest",
    [
        _manifest("tabicl-dimer-classifier-v2", "checkpoints/best.ckpt"),
        {**_manifest("tabicl-dimer-classifier-v2", "checkpoints/best.safetensors"), "inference": {"modelPath": "checkpoints/best.ckpt"}},
        {**_manifest("tabicl-dimer-classifier-v2", "checkpoints/best.safetensors"), "payloadFiles": ["checkpoints/best.ckpt", "training_context.parquet"]},
        _manifest("tabicl-dimer-classifier-v1", "checkpoints/best.safetensors"),
        _manifest("tabicl-dimer-classifier-v0", "checkpoints/best.safetensors"),
    ],
)
def test_a_mixed_or_unknown_layout_is_refused_before_any_file_is_read(manifest: dict, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        P.check_bundle_layout(manifest)
    with pytest.raises(ValueError):  # verify_artifact_bundle checks the layout before sizes, digests or loading
        P.verify_artifact_bundle(tmp_path, {**manifest, "trainingContext": "training_context.parquet"})


def test_the_export_manifest_points_checkpoint_model_path_and_payload_at_best_safetensors(tmp_path: Path) -> None:
    S = _stages()
    ckpt, ctx = tmp_path / "best.safetensors", tmp_path / "training_context.parquet"
    ckpt.write_bytes(b"w" * 16)
    ctx.write_bytes(b"c" * 8)
    enc = {"target": "target", "features": ["a"], "encoders": {}}
    manifest = S.build_manifest(P, enc=enc, classes=[0, 1], mode="pretrained", selection="s", metrics=None, n_estimators=8, checkpoint=ckpt, context=ctx, runtime={}, source={})
    assert manifest["artifactFormat"] == "tabicl-dimer-classifier-v2"
    assert manifest["checkpoint"] == manifest["inference"]["modelPath"] == "checkpoints/best.safetensors"
    assert manifest["payloadFiles"] == ["checkpoints/best.safetensors", "training_context.parquet"]
    assert manifest["digests"]["checkpointSha256"] == P.sha256_file(ckpt) and manifest["sizes"]["checkpoint"] == 16
    assert P.check_bundle_layout(manifest) == P.ARTIFACT_FORMAT


def test_a_pretrained_bundle_must_carry_the_pinned_base_in_its_own_layout() -> None:
    S = _stages()
    base = {"baseCheckpoint": P.BASE_CHECKPOINT_NAME, "baseModelRevision": P.MODEL_REVISION, "baseModelSha256": P.BASE_MODEL_SHA256, "mode": "pretrained"}
    v2 = {**base, "artifactFormat": P.ARTIFACT_FORMAT, "digests": {"checkpointSha256": P.CONVERTED_MODEL_SHA256}}
    v1 = {**base, "artifactFormat": P.LEGACY_ARTIFACT_FORMAT, "digests": {"checkpointSha256": P.BASE_MODEL_SHA256}}
    assert "safetensors conversion" in S.check_bundle_identity(P, v2)["checkpoint_binding"]
    assert "deprecated v1" in S.check_bundle_identity(P, v1)["checkpoint_binding"]
    for swapped in ({**v2, "digests": {"checkpointSha256": P.BASE_MODEL_SHA256}}, {**v1, "digests": {"checkpointSha256": P.CONVERTED_MODEL_SHA256}}):
        with pytest.raises(RuntimeError, match="must be the pinned base"):
            S.check_bundle_identity(P, swapped)


# ---------------------------------------------------------------- model-backed: conversion, equivalence, both layouts


@pytest.fixture()
def finetuned(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("tabicl")
    from tabicl._model.tabicl import TabICL

    torch.manual_seed(3)
    network = TabICL(**TINY_CONFIG)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    for parameter in network.parameters():
        parameter.grad = torch.randn_like(parameter) * 1e-3
    optimizer.step()
    ckpt = {"config": dict(TINY_CONFIG), "state_dict": {k: v.cpu() for k, v in network.state_dict().items()}, "optimizer_state": optimizer.state_dict(), "scheduler_state": None, "curr_step": 1, "epoch": 1, "best_metric": -0.5}
    out = tmp_path / "ft"
    out.mkdir()
    torch.save(ckpt, out / "best.ckpt")  # the layout tabicl's _save_checkpoint writes
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return {"torch": torch, "dir": out, "state": ckpt["state_dict"], "X": X, "y": y}


def _refuse_torch_load(monkeypatch: pytest.MonkeyPatch, torch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("torch.load called on the safetensors serving path")

    monkeypatch.setattr(torch, "load", refuse)


def test_a_finetune_checkpoint_is_converted_once_and_no_pickle_survives(finetuned) -> None:
    torch = finetuned["torch"]
    pickle_path = finetuned["dir"] / "best.ckpt"
    source_sha = P.sha256_file(pickle_path)
    keep = finetuned["dir"].parent / "copy" / "best.ckpt"  # same name: the source name is part of the metadata
    keep.parent.mkdir()
    shutil.copy2(pickle_path, keep)
    facts = P.convert_finetuned_checkpoint(pickle_path)
    target = Path(facts["path"])
    assert target == finetuned["dir"] / "best.safetensors" and target.is_file() and not pickle_path.exists()
    assert facts["source_sha256"] == source_sha and facts["converted_sha256"] == P.sha256_file(target)
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(target), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["format"] == P.CONVERSION_FORMAT and metadata["source_file"] == "best.ckpt" and metadata["source_sha256"] == source_sha
    assert json.loads(metadata[P.CONFIG_METADATA_KEY]) == TINY_CONFIG
    tensors = load_file(str(target))
    assert tensors.keys() == finetuned["state"].keys() and all(torch.equal(tensors[k], v) for k, v in finetuned["state"].items())
    again = P.convert_finetuned_checkpoint(keep, finetuned["dir"].parent / "again.safetensors", remove_source=False)
    assert again["converted_sha256"] == facts["converted_sha256"] and keep.exists()  # byte-reproducible


def test_the_converted_finetune_predicts_exactly_like_the_ckpt_with_torch_load_refused(finetuned, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = finetuned["torch"]
    X, y = finetuned["X"], finetuned["y"]
    pickle_path = finetuned["dir"] / "best.ckpt"
    keep = finetuned["dir"].parent / "kept.ckpt"
    shutil.copy2(pickle_path, keep)
    converted = Path(P.convert_finetuned_checkpoint(pickle_path)["path"])
    from tabicl import TabICLClassifier

    stock = TabICLClassifier(model_path=str(keep), allow_auto_download=False, n_estimators=2, random_state=0, device="cpu").fit(X[:40], y[:40])
    expected, labels = stock.predict_proba(X[40:]), stock.predict(X[40:])
    shim = P.create_classifier(model_path=converted, n_estimators=2, random_state=0, device="cpu")
    _refuse_torch_load(monkeypatch, torch)
    shim.fit(X[:40], y[:40])
    got, got_labels = shim.predict_proba(X[40:]), shim.predict(X[40:])
    monkeypatch.undo()
    assert np.array_equal(got, expected) and np.array_equal(got_labels, labels)


def _bundle(root: Path, fmt: str, member: str, weights: Path, X, y) -> dict:
    import pandas as pd

    (root / "checkpoints").mkdir(parents=True)
    shutil.copy2(weights, root / member)
    context = root / "training_context.parquet"
    pd.DataFrame({**{f"f{i}": X[:40, i] for i in range(X.shape[1])}, "target": y[:40]}).to_parquet(context, index=False)
    manifest = {**_manifest(fmt, member), "trainingContext": "training_context.parquet", "featureColumns": [f"f{i}" for i in range(X.shape[1])], "targetColumn": "target", "digests": {"checkpointSha256": P.sha256_file(root / member), "trainingContextSha256": P.sha256_file(context)}, "sizes": {"checkpoint": (root / member).stat().st_size, "trainingContext": context.stat().st_size}}
    (root / "artifact.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _serve(root: Path, manifest: dict):
    import pandas as pd

    members = P.verify_artifact_bundle(root, manifest)
    context = pd.read_parquet(members["training_context"])
    model = P.create_classifier(model_path=members["checkpoint"], n_estimators=2, random_state=0, device="cpu")
    return model.fit(context[manifest["featureColumns"]], context[manifest["targetColumn"]])


def test_both_bundle_layouts_serve_identical_predictions(finetuned, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = finetuned["torch"]
    X, y = finetuned["X"], finetuned["y"]
    keep = tmp_path / "kept.ckpt"
    shutil.copy2(finetuned["dir"] / "best.ckpt", keep)
    converted = Path(P.convert_finetuned_checkpoint(finetuned["dir"] / "best.ckpt")["path"])
    v1 = _bundle(tmp_path / "v1", P.LEGACY_ARTIFACT_FORMAT, P.LEGACY_BUNDLE_CHECKPOINT, keep, X, y)
    v2 = _bundle(tmp_path / "v2", P.ARTIFACT_FORMAT, P.BUNDLE_CHECKPOINT, converted, X, y)
    test_rows = X[40:]

    calls: list[dict] = []
    real_load = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    with pytest.warns(DeprecationWarning, match="tabicl-dimer-classifier-v1"):
        legacy = _serve(tmp_path / "v1", v1)
    assert type(legacy).__name__ == "LegacyCheckpointTabICLClassifier"
    assert calls and all(call.get("weights_only") is True for call in calls)
    monkeypatch.undo()
    expected = legacy.predict_proba(pd_frame(test_rows))

    _refuse_torch_load(monkeypatch, torch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        current = _serve(tmp_path / "v2", v2)
        got = current.predict_proba(pd_frame(test_rows))
    monkeypatch.undo()
    assert type(current).__name__ == "SafetensorsTabICLClassifier"
    assert np.array_equal(got, expected)


def pd_frame(rows):
    import pandas as pd

    return pd.DataFrame({f"f{i}": rows[:, i] for i in range(rows.shape[1])})
