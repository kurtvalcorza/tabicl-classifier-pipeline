"""Offline tests for the fleet snapshot helpers (NOTEBOOK_SPEC 1.1 ST3/ST4, MOD13).

No weights are downloaded and no model is loaded: the snapshot directory is a temporary one, the
downloader is injected, and ``BASE_MODEL_SHA256`` is redirected to the digest of the stand-in file
where the happy path needs the package constant and the manifest to agree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tabicl_classifier_pipeline import api as pipe_mod
from tabicl_classifier_pipeline.api import (
    BASE_CHECKPOINT_NAME,
    BASE_MODEL_SHA256,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    TabICLClassificationPipeline,
    stage_missing_files,
    verify_snapshot,
)

PAYLOAD = b"not-the-real-checkpoint"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def _manifest(sha: str) -> dict:
    return {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [{"path": BASE_CHECKPOINT_NAME, "bytes": len(PAYLOAD), "sha256": sha}],
        "totalBytes": len(PAYLOAD),
    }


def _snapshot(tmp_path: Path, *, manifest: dict | None = None, write_payload: bool = True) -> Path:
    root = tmp_path / "weights" / MODEL_KEY
    root.mkdir(parents=True)
    (root / MANIFEST_NAME).write_text(json.dumps(manifest or _manifest(DIGEST)), encoding="utf-8")
    if write_payload:
        (root / BASE_CHECKPOINT_NAME).write_bytes(PAYLOAD)
    return root


def test_identity_constants_are_the_fleet_shape() -> None:
    assert MODEL_ID == pipe_mod.BASE_MODEL_REPO == "jingang/TabICL"
    assert MODEL_REVISION == pipe_mod.BASE_MODEL_REVISION
    assert len(MODEL_REVISION) == 40 and all(c in "0123456789abcdef" for c in MODEL_REVISION)
    assert MODEL_LICENSE == "bsd-3-clause"
    assert pipe_mod.DEFAULT_WEIGHTS_DIR.name == MODEL_KEY
    assert pipe_mod.DEFAULT_WEIGHTS_DIR.parent.name == "weights"
    assert MANIFEST_NAME == "dimer-base-manifest.json"


def test_committed_manifest_matches_the_weight_constant() -> None:
    """The repository's own snapshot manifest must carry BASE_MODEL_SHA256 (never two truths)."""
    manifest_path = pipe_mod.DEFAULT_WEIGHTS_DIR / MANIFEST_NAME
    if not manifest_path.is_file():
        pytest.skip("snapshot manifest is not staged in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"]) == (MODEL_ID, MODEL_REVISION)
    digests = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    assert digests[BASE_CHECKPOINT_NAME] == BASE_MODEL_SHA256


def test_verify_snapshot_returns_the_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipe_mod, "BASE_MODEL_SHA256", DIGEST)
    root = _snapshot(tmp_path)
    result = verify_snapshot(root)
    assert result["modelId"] == MODEL_ID
    assert result["revision"] == MODEL_REVISION
    assert result["path"] == str(root)
    assert [entry["path"] for entry in result["files"]] == [BASE_CHECKPOINT_NAME]


def test_verify_snapshot_rejects_a_tampered_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipe_mod, "BASE_MODEL_SHA256", DIGEST)
    root = _snapshot(tmp_path)
    # Same byte count, different content: the size check passes and the digest check is what fires.
    (root / BASE_CHECKPOINT_NAME).write_bytes(PAYLOAD[:-1] + b"!")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)


def test_verify_snapshot_rejects_a_manifest_that_contradicts_the_weight_constant(tmp_path: Path) -> None:
    root = _snapshot(tmp_path, manifest=_manifest("0" * 64))
    with pytest.raises(ValueError, match="BASE_MODEL_SHA256"):
        verify_snapshot(root)


def test_verify_snapshot_rejects_a_foreign_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipe_mod, "BASE_MODEL_SHA256", DIGEST)
    manifest = {**_manifest(DIGEST), "modelId": "someone-else/TabDPT"}
    root = _snapshot(tmp_path, manifest=manifest)
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(root)
    manifest = {**_manifest(DIGEST), "revision": "f" * 40}
    root2 = _snapshot(tmp_path / "other", manifest=manifest)
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root2)


def test_verify_snapshot_reports_a_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipe_mod, "BASE_MODEL_SHA256", DIGEST)
    root = _snapshot(tmp_path, write_payload=False)
    with pytest.raises(FileNotFoundError, match="snapshot file missing"):
        verify_snapshot(root)


def test_stage_missing_files_is_a_no_op_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipe_mod, "BASE_MODEL_SHA256", DIGEST)
    root = _snapshot(tmp_path)
    assert stage_missing_files(root) == []


def test_stage_missing_files_refuses_without_allow_download(tmp_path: Path) -> None:
    root = _snapshot(tmp_path, write_payload=False)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)


def test_stage_missing_files_uses_the_injected_downloader(tmp_path: Path) -> None:
    root = _snapshot(tmp_path, write_payload=False)
    calls: list[tuple[str, Path]] = []

    def downloader(relative_path: str, destination: Path) -> None:
        calls.append((relative_path, destination))
        (destination / relative_path).write_bytes(PAYLOAD)

    fetched = stage_missing_files(root, allow_download=True, downloader=downloader)
    assert fetched == [BASE_CHECKPOINT_NAME]
    assert calls == [(BASE_CHECKPOINT_NAME, root)]
    assert (root / BASE_CHECKPOINT_NAME).read_bytes() == PAYLOAD


def test_stage_missing_files_refuses_a_foreign_manifest(tmp_path: Path) -> None:
    manifest = {**_manifest(DIGEST), "revision": "a" * 40}
    root = _snapshot(tmp_path, manifest=manifest, write_payload=False)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(root, allow_download=True, downloader=lambda *a: None)


def test_from_pretrained_pins_the_verified_weight_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipe_mod, "BASE_MODEL_SHA256", DIGEST)
    calls: list[dict] = []

    def fake_create_classifier(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(pipe_mod, "create_classifier", fake_create_classifier)
    root = _snapshot(tmp_path)
    pipe = TabICLClassificationPipeline.from_pretrained(
        weights_dir=root, n_estimators=3, random_state=7, device="cpu"
    )
    assert Path(pipe.model_path) == root / BASE_CHECKPOINT_NAME
    assert pipe.source == "local-snapshot"
    assert (pipe.n_estimators, pipe.random_state, pipe.device) == (3, 7, "cpu")
    assert calls == [
        {
            "model_path": root / BASE_CHECKPOINT_NAME,
            "n_estimators": 3,
            "random_state": 7,
            "device": "cpu",
            "allow_auto_download": False,
        }
    ]
    assert pipe.is_fitted is False
    with pytest.raises(RuntimeError, match="not conditioned"):
        pipe.predict([[1.0]])


def test_from_pretrained_refuses_an_unstaged_snapshot(tmp_path: Path) -> None:
    root = _snapshot(tmp_path, write_payload=False)
    with pytest.raises(FileNotFoundError):
        TabICLClassificationPipeline.from_pretrained(weights_dir=root, device="cpu")
