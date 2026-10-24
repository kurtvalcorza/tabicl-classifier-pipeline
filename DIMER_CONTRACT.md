# DIMER integration contract — TabICLv2 classifier

This document separates the contract owned by the TabICL repositories from changes that belong to DIMER Workbench. It reflects the 2026-08-23 contract audit against `dimer-backend:on-prem`.

## Repository-owned contract

### Pipeline manifest

`dimer-pipeline.json` is the version-controlled source for classification preprocessing and fine-tuning controls. The finetuner carries the same manifest and repo CI must keep the copies synchronized.

Task identity is `tabular_classification`.

### Validator

The validator supports the normal mounted dataset contract (`DIMER_DATASET_DIR`) and an on-prem compatibility path where an empty local dataset is staged from `S3_BUCKET` + `S3_DATASET_PREFIX`. Existing local files take precedence.

The validator consumes `DIMER_PREPROCESSING_ARGS_JSON` when DIMER supplies it and emits structured success/failure results with classification metadata including `classNames`.

### Fine-tuner

Normal executions retain the `/data` contract. In `GPU_BURST_MODE`, the worker stages the dataset into bounded scratch, runs the unchanged TabICL training core, verifies declared artifact descriptors, uploads the complete output tree, uploads `result.json` last, then calls the completion callback.

The deployable TabICL serving bundle (format `tabicl-dimer-classifier-v2`) requires at minimum:

- `checkpoints/best.safetensors`
- `training_context.parquet`
- `artifact.json`

#### Bundle format v1 → v2 (2026-10-24)

- **v2 (current, written by the worker and the tutorial notebook):** `artifact.json` has `artifactFormat`
  `tabicl-dimer-classifier-v2`, and its `checkpoint`, `inference.modelPath` and `payloadFiles` all name
  `checkpoints/best.safetensors`, whose SHA-256 and size are in `digests.checkpointSha256` and `sizes.checkpoint`. The
  file holds the network's `state_dict` as tensors and its `config` as sorted JSON string metadata (`tabicl_config`); it
  is loaded without `torch.load` (`SafetensorsTabICLClassifier`, which builds `TabICL(**config)` and calls
  `load_state_dict(strict=True)`). The worker's `result.json` `modelArtifact` names `best.safetensors` with its `sha256`.
- **Conversion:** tabicl 2.1.1's fine-tuning loop writes `best.ckpt` itself with `torch.save`. Right after training it is
  read once with `torch.load(weights_only=True)`, only `config` and `state_dict` are kept, and the pickle is deleted; only
  the safetensors file is packaged. A `pretrained` bundle carries the pinned safetensors conversion of the base checkpoint.
- **v1 (deprecated, still readable):** `tabicl-dimer-classifier-v1` bundles carry a pickle `checkpoints/best.ckpt`. Every
  consumer in the TabICL repositories still accepts them, loading the checkpoint only with `torch.load(weights_only=True)`
  and emitting a `DeprecationWarning`, until the v1 release artifacts are re-issued. Nothing loads with
  `weights_only=False`.
- **Shared code:** the conversion, the shim and the format checks live in `safetensors_shim.py` of
  `kurtvalcorza/tabicl-classifier-pipeline` (canonical); the worker vendors it byte for byte as `tabicl_safetensors_shim.py`
  and pins its SHA-256 in a test.

The checkpoint alone is not a complete serving artifact because TabICL reloads the saved support/context table before prediction.

### Base checkpoint

The default classifier base checkpoint is repository-pinned by immutable revision and SHA-256. `DIMER_BASE_MODEL_PATH`, when explicitly mounted by the platform/operator, overrides that default and its observed digest/source is recorded. `DIMER_MODEL_CONFIG_JSON` is not currently converted by this worker into a checkpoint path.

## DIMER-side requirements — documentation only

These are platform changes and are intentionally **not implemented** in the TabICL repositories.

1. **Validator preprocessing handoff.** Dataset-validation jobs must inject the resolved `DIMER_PREPROCESSING_ARGS_JSON` so validation and fine-tuning use the same `target_column`, `drop_columns`, `max_train_rows`, and `validation_split` values.
2. **First-class model bundles.** Export, promotion, download, and deployment must preserve `best.safetensors` (v2; `best.ckpt` in deprecated v1 bundles), `training_context.parquet`, `artifact.json`, and other required declared companions rather than treating `modelArtifact`/`best.pt` as the complete model.
3. **Generic artifact transfer.** Platform transport should consume `result.json.artifacts` roles instead of assuming YOLO file names. The repo-side GPU-burst adapter currently bridges the existing environment by publishing the complete tree and retaining the legacy model-key alias.
4. **Concrete base-model handoff.** If Workbench exposes multiple base models for this pipeline, the selected model must resolve to an actual mounted checkpoint (or another explicit worker contract); passing model metadata alone is not sufficient.
5. **Tabular registry metadata.** The DIMER database/registry must provide TabICL-specific dataset description/format, validation rules, training entrypoint, and artifact semantics so generic YOLO defaults are not surfaced for this tabular pipeline.
6. **GPU capacity.** Production fine-tuning requires a compatible CUDA GPU execution path; CPU fallback is intentionally not part of this pipeline.
7. **Schema/version validation.** DIMER should eventually validate/version worker result and artifact envelopes instead of relying on permissive dictionary access.

## Production acceptance boundary

Repository CI can prove worker semantics, transport adapters, artifact integrity, and reload behavior. Final production acceptance still requires an on-platform path test:

**upload → validator → GPU fine-tune → persisted bundle → fresh reload → DIMER deployment → API `predict` + `predict_proba` parity.**

No `dimer-backend` modification is included in the repo-side remediation work tracked by issue #7.
