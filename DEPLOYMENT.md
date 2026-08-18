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

## Pre-enable checklist

1. Build both repository images through DIMER.
2. Confirm validator pass/fail behavior on valid and adversarial sample archives.
3. Run a GPU fine-tune smoke test.
4. Confirm `best.ckpt` exists and the built-in reload smoke test succeeds.
5. Confirm `result.json` contains validation metrics, checkpoint SHA-256, and dataset digest.
6. Supply `test.csv` and confirm test metrics are reported but do not influence checkpoint selection.
7. Verify the DIMER inference-serving layer can reconstruct and serve the custom TabICL artifact.
8. Load-test representative row/feature counts and set the production GPU/RAM profile from measured peak usage.

Do not production-enable the pipeline until step 7 is verified on-platform.
