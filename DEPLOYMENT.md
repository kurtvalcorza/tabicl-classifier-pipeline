# Deployment and Operations

## DIMER components

- Validator: `tabicl-classifier-dataset-validator`, CPU image.
- Fine-tuner: `tabicl-classifier-finetuner`, CUDA image.
- Base checkpoint: `tabicl-classifier-v2-20260212.ckpt`.
- Python package: `tabicl[finetune]==2.1.1`.

## Network

The first fine-tuning run downloads the upstream TabICLv2 checkpoint unless it is already cached or baked into the image. Production deployments should either permit the required model-download egress or bake/cache the checkpoint and validate its provenance.

## GPU

The DIMER fine-tuner intentionally requires CUDA. There is no CPU zero-shot fallback in this component because that would change the requested operation from fine-tuning to inference.

Start with a single modern CUDA GPU and conservative row/feature limits, then profile representative workloads before increasing `max_train_rows`, estimator counts, or epochs.

## Artifact serving

A successful training run produces:

- `checkpoints/best.ckpt`: downstream fine-tuned TabICLv2 checkpoint;
- `training_context.parquet`: the in-context support table used at inference time;
- `artifact.json`: paths, target/feature schema, package/checkpoint metadata.

The inference service must load `best.ckpt`, reconstruct `X_context`/`y_context` from `training_context.parquet`, call `TabICLClassifier.fit`, then serve `predict`/`predict_proba`.

Because the artifact intentionally contains the training context, treat the artifact with the same data-governance controls as the source training dataset.

`artifact.json` is the complete inference contract: it carries the target and feature-column ordering, the `inference` block (class, `nEstimators`, `randomState`, device, `allowAutoDownload`, `supportManyClasses`), the persisted `categoricalEncoders`, and component `digests`. A fresh-environment loader must be able to reconstruct the exact scored model from `artifact.json` plus the referenced files, with no additional configuration.

## Base model handoff

This pipeline is **fixed to the pinned TabICLv2 classifier checkpoint** `tabicl-classifier-v2-20260212.ckpt` at Hugging Face revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`, baked into the image and SHA-256-verified at fine-tune time. The finetuner records `baseModelRevision`, `baseModelSha256`, `baseModelSource`, and `baseMatchesPinned` in `result.json` and `artifact.json`.

Resolution precedence and provenance are exact:

| `baseModelSource` | when | verification | `baseModelRevision` |
|---|---|---|---|
| `dimer-provided` | `DIMER_BASE_MODEL_PATH` is set (DIMER operator override) | **must exist** — a configured-but-missing path fails the run; used as-is, SHA-256 recorded | pinned revision only if the bytes match the pinned default, else `null` |
| `pinned-baked` | default — the checkpoint baked into the image (`TABICL_BAKED_BASE_MODEL`) | SHA-256 hard-verified against the pinned value | pinned revision |
| `pinned-download` | no baked copy present | downloaded at the pinned revision, SHA-256 hard-verified | pinned revision |

So a DIMER-selected Base Model deterministically controls the checkpoint actually loaded (no silent fallback to the default), and a custom base never falsely claims the pinned revision. `model_id` is intentionally **not** a `dimer-pipeline.json` hyperparameter — every declared manifest key maps one-to-one to runtime behavior.

## Pre-enable checklist (release gate)

1. Build both repository images through DIMER.
2. Confirm validator pass/fail behavior on valid and adversarial sample archives (duplicate splits, nested/zip-bomb archives, path traversal, `target_column` in `drop_columns`, malformed runtime config).
3. Run a GPU fine-tune smoke test.
4. Confirm `best.ckpt` exists and the built-in reload smoke test — which reconstructs the model **from `artifact.json` plus its referenced files** (`checkpoints/best.ckpt`, `training_context.parquet`) — succeeds.
5. Confirm `result.json` records validation metrics, fine-tuned checkpoint SHA-256, training-context SHA-256, exact base-model provenance (revision + SHA-256 + source), and the input **dataset digest** (`provenance.dataset`).
6. Supply `test.csv` and confirm test metrics are reported but do not influence checkpoint selection.
7. **End-to-end serving acceptance (on-platform):** BYOD upload → validator → GPU fine-tune → artifact persistence/download → fresh-environment reload, then DIMER deployment → API inference. On a shared sample, compare both `predict` **and `predict_proba`** (including class ordering) across the training instance, the offline reload, and the API — they must all agree, so the served artifact reproduces the exact model whose metrics `result.json` reports.
8. Confirm each `dimer-pipeline.json` control changes the intended runtime behavior (spot-check e.g. `n_estimators_inference`, `seed`, `eval_metric`).
9. Load-test representative row/feature counts and set the production GPU/RAM profile from measured peak usage.

Do not production-enable the pipeline until step 7 is verified on-platform.
