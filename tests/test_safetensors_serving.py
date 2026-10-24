# ruff: noqa: E501  -- single-line test fixtures and assertions are kept readable
"""The base model is served from a pinned safetensors conversion, never from the pickle checkpoint.

The offline tests need no model library. The model-backed tests build a tiny randomly initialised TabICL network,
save it in the upstream ``.ckpt`` layout, convert it and compare the upstream ``.ckpt`` load with the safetensors
shim; they skip when ``tabicl``/``safetensors`` are not installed (the static CI job installs neither). The last
test repeats the comparison on the real pinned checkpoint when it is staged under ``weights/``.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tabicl_classifier_pipeline import api as P

TINY_CONFIG = {
    "max_classes": 10,
    "embed_dim": 16,
    "col_num_blocks": 1,
    "col_nhead": 2,
    "col_num_inds": 8,
    "row_num_blocks": 1,
    "row_nhead": 2,
    "row_num_cls": 2,
    "icl_num_blocks": 1,
    "icl_nhead": 2,
}


def _header(data: bytes) -> tuple[int, bytes]:
    size = struct.unpack("<Q", data[:8])[0]
    return size, data[8 : 8 + size]


def test_converted_digest_is_pinned_and_distinct_from_the_source() -> None:
    assert len(P.CONVERTED_MODEL_SHA256) == 64 and int(P.CONVERTED_MODEL_SHA256, 16) >= 0
    assert P.CONVERTED_MODEL_SHA256 != P.BASE_MODEL_SHA256
    assert P.CONVERTED_CHECKPOINT_NAME.endswith(".safetensors")
    assert Path(P.CONVERTED_CHECKPOINT_NAME).stem == Path(P.BASE_CHECKPOINT_NAME).stem


def test_canonical_safetensors_sorts_the_header_and_keeps_the_data() -> None:
    header = {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "__metadata__": {"z": "1", "a": "2"}}
    text = json.dumps(header).encode()
    data = struct.pack("<Q", len(text)) + text + b"\x00\x00\x80\x3f"
    out = P.canonical_safetensors(data)
    size, raw = _header(out)
    assert size % 8 == 0
    assert raw.rstrip(b" ") == json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    assert out[8 + size :] == b"\x00\x00\x80\x3f"
    assert P.canonical_safetensors(out) == out


def test_ensure_converted_uses_a_pinned_file_without_reading_the_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / P.CONVERTED_CHECKPOINT_NAME).write_bytes(b"converted")
    monkeypatch.setattr(P, "CONVERTED_MODEL_SHA256", hashlib.sha256(b"converted").hexdigest())

    def refuse(*args, **kwargs):
        raise AssertionError("the source must not be read when the pinned conversion is present")

    monkeypatch.setattr(P, "convert_checkpoint", refuse)
    monkeypatch.setattr(P, "verify_snapshot", refuse)
    assert P.ensure_converted(tmp_path) == tmp_path / P.CONVERTED_CHECKPOINT_NAME


def test_ensure_converted_refuses_a_conversion_that_misses_the_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "stage_missing_files", lambda *a, **k: [])
    monkeypatch.setattr(P, "verify_snapshot", lambda *a, **k: {})

    def fake_convert(source, destination, **kwargs):
        Path(destination).write_bytes(b"other bytes")
        return {"converted_sha256": hashlib.sha256(b"other bytes").hexdigest()}

    monkeypatch.setattr(P, "convert_checkpoint", fake_convert)
    with pytest.raises(ValueError, match="converted sha256"):
        P.ensure_converted(tmp_path)
    assert not (tmp_path / P.CONVERTED_CHECKPOINT_NAME).exists()


def test_convert_refuses_a_source_that_misses_its_digest(tmp_path: Path) -> None:
    source = tmp_path / "x.ckpt"
    source.write_bytes(b"not the pinned checkpoint")
    with pytest.raises(ValueError, match="refusing to deserialise"):
        P.convert_checkpoint(source, tmp_path / "x.safetensors")
    assert not (tmp_path / "x.safetensors").exists()


@pytest.fixture()
def tiny(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("tabicl")
    from tabicl._model.tabicl import TabICL

    torch.manual_seed(0)
    network = TabICL(**TINY_CONFIG)
    source = tmp_path / "tiny.ckpt"
    torch.save({"config": dict(TINY_CONFIG), "state_dict": network.state_dict()}, source)
    digest = P.sha256_file(source)
    converted = tmp_path / "tiny.safetensors"
    facts = P.convert_checkpoint(source, converted, expected_sha256=digest, source_name="tiny.ckpt")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return {"torch": torch, "source": source, "digest": digest, "converted": converted, "facts": facts, "X": X, "y": y}


def test_conversion_is_byte_reproducible(tiny, tmp_path: Path) -> None:
    again = tmp_path / "again.safetensors"
    facts = P.convert_checkpoint(tiny["source"], again, expected_sha256=tiny["digest"], source_name="tiny.ckpt")
    assert again.read_bytes() == tiny["converted"].read_bytes()
    assert facts["converted_sha256"] == tiny["facts"]["converted_sha256"] == P.sha256_file(again)
    from safetensors import safe_open

    with safe_open(str(again), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["format"] == P.CONVERSION_FORMAT
    assert json.loads(metadata[P.CONFIG_METADATA_KEY]) == TINY_CONFIG
    assert metadata["source_sha256"] == tiny["digest"]


def test_shim_serves_without_torch_load_and_matches_the_ckpt_load(tiny, monkeypatch: pytest.MonkeyPatch) -> None:
    from tabicl import TabICLClassifier

    torch = tiny["torch"]
    X, y = tiny["X"], tiny["y"]
    stock = TabICLClassifier(model_path=str(tiny["source"]), allow_auto_download=False, n_estimators=2, random_state=0, device="cpu")
    stock.fit(X[:40], y[:40])
    expected = stock.predict_proba(X[40:])

    shim = P.create_classifier(model_path=tiny["converted"], n_estimators=2, random_state=0, device="cpu")
    assert type(shim).__name__ == "SafetensorsTabICLClassifier"

    def refuse(*args, **kwargs):
        raise AssertionError("torch.load called on the serving path")

    monkeypatch.setattr(torch, "load", refuse)
    shim.fit(X[:40], y[:40])
    got = shim.predict_proba(X[40:])
    monkeypatch.undo()
    assert np.array_equal(got, expected)
    assert np.array_equal(shim.predict(X[40:]), stock.predict(X[40:]))
    a, b = stock.model_.state_dict(), shim.model_.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_a_ckpt_path_still_goes_to_the_upstream_loader(tiny) -> None:
    estimator = P.create_classifier(model_path=tiny["source"], n_estimators=2, random_state=0, device="cpu")
    assert type(estimator).__name__ == "TabICLClassifier"


def test_the_finetuner_loads_the_pretrained_weights_through_the_shim(tiny, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = tiny["torch"]
    pytest.importorskip("transformers")  # tabicl's fine-tuning extra
    finetuner = P.create_finetuned_classifier(model_path=str(tiny["converted"]), device="cpu", epochs=1)
    assert type(finetuner).__name__ == "SafetensorsFinetunedTabICLClassifier"
    inner = finetuner._create_inner_estimator(n_estimators=2, device=torch.device("cpu"))
    assert type(inner).__name__ == "SafetensorsTabICLClassifier"
    monkeypatch.setattr(torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("torch.load")))
    finetuner._load_pretrained(torch.device("cpu"))
    assert finetuner.model_config_ == TINY_CONFIG
    monkeypatch.undo()
    reference = torch.load(tiny["source"], map_location="cpu", weights_only=True)["state_dict"]
    assert all(torch.equal(finetuner.model_.state_dict()[k], v) for k, v in reference.items())


def test_the_pinned_conversion_reproduces_the_ckpt_predictions(tmp_path: Path) -> None:
    """On the real pinned checkpoint, when staged: the conversion hits the pinned digest and predictions are identical."""
    source = P.DEFAULT_WEIGHTS_DIR / P.BASE_CHECKPOINT_NAME
    if not source.is_file():
        pytest.skip("the pinned checkpoint is not staged under weights/")
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabicl")
    from tabicl import TabICLClassifier

    converted = tmp_path / P.CONVERTED_CHECKPOINT_NAME
    assert P.convert_checkpoint(source, converted)["converted_sha256"] == P.CONVERTED_MODEL_SHA256
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] - X[:, 2] > 0).astype(int)
    stock = TabICLClassifier(model_path=str(source), allow_auto_download=False, n_estimators=2, random_state=0, device="cpu").fit(X[:150], y[:150])
    shim = P.create_classifier(model_path=converted, n_estimators=2, random_state=0, device="cpu").fit(X[:150], y[:150])
    assert np.array_equal(shim.predict_proba(X[150:]), stock.predict_proba(X[150:]))
    a, b = stock.model_.state_dict(), shim.model_.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
