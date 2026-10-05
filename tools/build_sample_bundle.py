#!/usr/bin/env python3
"""Build (or ``--check``) the pinned sample bundle the ARTIFACT-INFERENCE notebook uses by default.

TICA-M1. The sample is the serving bundle the E2E notebook exports on its default path
(``Sample: Breast Cancer``, pretrained mode), produced here with the E2E stage runner's own
``stage_data`` + ``stage_validate`` (the seeded stratified 60/20/20 split and the support-fitted
encoders) and its ``build_manifest``. Two things differ from a hosted export and are recorded in the
manifest and in ``SAMPLE_BUNDLE.json``: no model was run, so ``metrics`` is ``null`` and the
``runtime`` block names the pinned lock instead of an observed runtime; and the 110 MB checkpoint is
not stored. ``mode: pretrained`` means it IS the pinned base checkpoint, so the manifest records the
base's size and SHA-256 and the companion stages the digest-verified base file in its place (the
companion refuses any ``pretrained`` bundle whose checkpoint digest differs from
``BASE_MODEL_SHA256``). The E2E notebook's export prints ``matches_pinned_sample_context`` so a
hosted run confirms the support table byte for byte.

Writes ``examples/sample-bundle/``: ``artifact.json``, ``training_context.parquet``,
``new_rows.csv`` (the first eight holdout rows, unlabelled, with ``row_id``) and
``SAMPLE_BUNDLE.json``. Needs only the CI dependencies.

    python tools/build_sample_bundle.py           # write
    python tools/build_sample_bundle.py --check   # exit 1 unless the committed files are reproduced
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "sample-bundle"
FILES = ("artifact.json", "training_context.parquet", "new_rows.csv")


def _runner():
    spec = importlib.util.spec_from_file_location("tutorial_stages", ROOT / "tools" / "tutorial_stages.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path) -> dict:
    stages = _runner()
    P = stages.package(ROOT)
    base = json.loads((ROOT / "weights" / P.MODEL_KEY / P.MANIFEST_NAME).read_text(encoding="utf-8"))
    base_entry = next(f for f in base["files"] if f["path"] == P.BASE_CHECKPOINT_NAME)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        run = stages.Run(tmp_path / "run", tmp_path / "outputs", tmp_path / "weights",
                         {"data_source": "Sample: Breast Cancer"})
        with contextlib.redirect_stdout(io.StringIO()):
            stages.stage_data(run)
            stages.stage_validate(run)
        enc = run.read_state("encoded.json", "sample")
        train = run.read_frame("train_enc.parquet", "sample")
        out.mkdir(parents=True, exist_ok=True)
        context = out / "training_context.parquet"
        train[enc["features"] + [enc["target"]]].to_parquet(context, index=False)
        stages.new_rows_frame(run, enc).to_csv(out / "new_rows.csv", index=False)
    runtime = {
        "pythonVersion": "3.12.12",
        "torchVersion": "2.14.0",
        "note": ("offline build by tools/build_sample_bundle.py: no model was run and torch was not "
                 "imported; the versions are the pinned lock's"),
    }
    source = {"producer": "tools/build_sample_bundle.py",
              "repository": "kurtvalcorza/tabicl-classifier-pipeline"}
    manifest = stages.build_manifest(
        P, enc=enc, classes=enc["classes"], mode="pretrained",
        selection="default:pretrained (offline sample build; no model run)", metrics=None,
        n_estimators=P.DEFAULT_N_ESTIMATORS, checkpoint=None, context=context, runtime=runtime,
        source=source, checkpoint_facts=(base_entry["sha256"], base_entry["bytes"]))
    (out / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    record = {
        "description": (
            "Pinned sample bundle for the ARTIFACT-INFERENCE notebook's default path: the bundle the "
            "E2E notebook exports on its default path (Sample: Breast Cancer, mode pretrained), "
            "without its checkpoint, which is the pinned base checkpoint."),
        "producer": ("tools/build_sample_bundle.py: the E2E stage runner's stage_data + stage_validate"
                     " + build_manifest; no model was run (metrics: null)"),
        "checkpoint": {"is": "the pinned base checkpoint (mode pretrained)", "file": P.BASE_CHECKPOINT_NAME,
                       "bytes": base_entry["bytes"], "sha256": base_entry["sha256"]},
        "e2e_cross_check": ("the E2E notebook's export prints matches_pinned_sample_context "
                            "on its default path"),
        "data": ("scikit-learn load_breast_cancer (Wisconsin Diagnostic Breast Cancer, UCI; 569 rows), "
                 "seeded stratified 60/20/20 split, seed 42; the support (train) partition"),
        "support_rows": len(train),
        "artifact_sha256": sha256(out / "artifact.json"),
        "files": {name: {"bytes": (out / name).stat().st_size, "sha256": sha256(out / name)}
                  for name in FILES},
    }
    (out / "SAMPLE_BUNDLE.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.check:
        record = build(OUT)
        print(json.dumps({"artifact_sha256": record["artifact_sha256"], "files": record["files"]}, indent=2))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        build(Path(tmp))
        stale = [name for name in (*FILES, "SAMPLE_BUNDLE.json")
                 if (Path(tmp) / name).read_bytes() != (OUT / name).read_bytes()]
    if stale:
        print(f"STALE: {stale} differ from tools/build_sample_bundle.py output", file=sys.stderr)
        return 1
    print("OK: examples/sample-bundle/ is reproduced byte for byte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
