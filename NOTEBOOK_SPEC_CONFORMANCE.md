# DIMER Notebook Specification v1.0 conformance record

This record applies to the release-grade tutorials in `tutorials/`.

| Notebook | Profile | Spec |
|---|---|---|
| `tabiclv2_classifier_colab.ipynb` | `E2E` | `1.0` |
| `tabiclv2_classifier_artifact_inference_colab.ipynb` | `ARTIFACT-INFERENCE` | `1.0` |

## Applicable SHOULD deviations

- **SEC8 — per-file digest and size verification:** new tutorial-produced bundles record and verify both SHA-256 and byte size for `checkpoints/best.ckpt` and `training_context.parquet`. Existing `tabicl-dimer-classifier-v1` bundles produced by the current DIMER fine-tuner record SHA-256 but may omit byte sizes. The companion preserves compatibility by warning when the `sizes` block is absent while still enforcing the whole-archive expanded-size ceiling and component digests.
- **SEC9 — reject unexpected unlisted files:** current DIMER fine-tuner output can include side artifacts such as `evaluation/report.json` beside the three serving files, but `artifact.json` does not enumerate those sidecars. The companion therefore warns on unlisted files rather than rejecting them. Required serving files remain manifest-bound and digest-verified.

These deviations are retained specifically for compatibility with the existing DIMER v1 producer contract; they do not weaken traversal/symlink/containment checks or the checkpoint deserialization trust boundary.

## Dependency lock (ENV2)

The release-grade notebooks install from `tutorials/requirements-release.lock`, a fully resolved Python 3.12 graph generated from `tutorials/requirements-release.in`. Standalone Colab verifies lock SHA-256 `64e9a167567495263694f555195ca7df4488016cc76ab1e327e480509a4ac6d5` before installation. The lock includes transitive dependencies rather than relying only on exact top-level requirements.

## Automated verification

Pull-request CI performs two distinct layers:

1. `tutorial-checks` — notebook JSON/Python compilation plus repository-specific static and regression checks. Static validation is not described as execution evidence.
2. `notebook-release-execution` — executes the committed default `E2E` path in a clean Jupyter kernel, then starts a separate fresh kernel for `ARTIFACT-INFERENCE`, supplies the previously exported artifact plus a separate inference CSV, and writes `release-evidence/notebook-release-evidence.json` together with the executed notebook copies.

The evidence JSON records the tested PR head/revision, platform/Python identity, package/runtime versions, artifact SHA-256, fresh-kernel boundary, and pass/fail outcomes. Optional CUDA gradient fine-tuning is outside the default CPU release path; if a release specifically claims that branch, accelerator-specific execution evidence must be recorded separately.

A clean-kernel CI pass does not override a known Google Colab failure. Any known current-Colab failure is release-blocking until resolved.

## AI provenance

The v1.0 conformance migration and release-gate implementation were produced with substantial AI assistance using **GPT-5.6 Sol High**, via **OpenAI / ChatGPT**, under Agent Relay role **Builder**, with maintainer direction. Attribution is provenance, not independent sign-off.
