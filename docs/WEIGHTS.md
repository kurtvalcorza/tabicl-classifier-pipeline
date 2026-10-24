# Weight provenance and local snapshot layout

| Field | Value |
|---|---|
| Upstream model | `jingang/TabICL` (Hugging Face) |
| Checkpoint | `tabicl-classifier-v2-20260212.ckpt` (110,368,038 bytes) |
| Immutable revision | `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` |
| SHA-256 | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |
| Served file | `tabicl-classifier-v2-20260212.safetensors` (110,257,568 bytes), SHA-256 `09bc6836d32b92f3f6126b9b15994b353dcf31e039b8c2a64e2c5c9f05b9e917` (`CONVERTED_MODEL_SHA256`), produced locally by `convert_checkpoint` |
| License | BSD-3-Clause (upstream `tabicl` and the checkpoint) |
| Package constants | `MODEL_ID`, `MODEL_REVISION`, `MODEL_LICENSE`, `MODEL_KEY = "tabicl-classifier-v2"` in `src/tabicl_classifier_pipeline/api.py` (aliased by the published `BASE_MODEL_REPO`, `BASE_MODEL_REVISION`, `BASE_CHECKPOINT_NAME`, `BASE_MODEL_SHA256`) |

## Local snapshot (fleet scheme, NOTEBOOK_SPEC 1.1 ST3/ST4)

The pinned snapshot lives in `weights/tabicl-classifier-v2/` (the `MODEL_KEY`). The committed
`dimer-base-manifest.json` there lists the checkpoint's path, byte size and SHA-256 and is the parity anchor the
standalone tutorials carry inline; the checkpoint itself is git-ignored (`weights/**/*.ckpt`). `verify_snapshot`
re-hashes every manifest entry and asserts the manifest digest equals the package's `BASE_MODEL_SHA256`;
`stage_missing_files(allow_download=True)` fetches only the entries that are absent, from the Hub at the immutable
revision; `TabICLClassificationPipeline.from_pretrained(weights_dir=...)` runs both before constructing the estimator on
the verified file (`allow_auto_download=False`). To use an offline copy, place the `.ckpt` in that directory before
running — nothing is downloaded when every manifest entry is present and verified.

```bash
# Stage the pinned checkpoint through the package helpers (fetches only what is absent):
python -c "from tabicl_classifier_pipeline import stage_missing_files, verify_snapshot; print(stage_missing_files(allow_download=True)); print(verify_snapshot()['files'])"
```

## Served weights: the safetensors conversion

The checkpoint is a Lightning/PyTorch pickle. `ensure_converted` (called by `from_pretrained` and the tutorial's `weights` stage) reads it once, after the digest check, with `torch.load(map_location="cpu", weights_only=True)`, and `convert_checkpoint` writes its `state_dict` as contiguous CPU tensors and its `config` as sorted JSON in the string metadata (`tabicl_config`), with a canonical (sorted) header, to `tabicl-classifier-v2-20260212.safetensors` beside it. The conversion is byte-reproducible and must equal `CONVERTED_MODEL_SHA256`; a present file with that digest is used without reading the source. tabicl 2.1.1 loads `.ckpt` files itself in `TabICLClassifier._load_model`, so the package ships a small shim, `SafetensorsTabICLClassifier`, which overrides only that method: it builds `TabICL(**config)` from the file's metadata and calls `load_state_dict(strict=True)` (the fine-tuner's inner estimator uses the same shim). `tests/test_safetensors_serving.py` asserts the serving path never calls `torch.load` and that predictions equal the `.ckpt` load. Fine-tuned weights and serving bundles are safetensors too (bundle format `tabicl-dimer-classifier-v2`): tabicl's fine-tuning loop writes `best.ckpt` with `torch.save`, and `convert_finetuned_checkpoint` reads it once right after training with `torch.load(weights_only=True)`, keeps only `config` and `state_dict`, writes `best.safetensors` and deletes the pickle; the bundle carries `checkpoints/best.safetensors`, and a `pretrained` bundle carries the converted base (`CONVERTED_MODEL_SHA256`), never the source `.ckpt`. Deprecated v1 bundles (`checkpoints/best.ckpt`) stay readable: `LegacyCheckpointTabICLClassifier` loads them only with `torch.load(weights_only=True)` and emits a `DeprecationWarning`. The conversion, the shim and the format checks are in `src/tabicl_classifier_pipeline/safetensors_shim.py`, which the DIMER fine-tune worker vendors byte for byte (`python tools/print_shim_sha256.py` prints the digest it pins). The pinned sample artifact of the notebook's external-artifact section is the v2 release `sample-artifact-magic-v2` (SHA-256 `9694e374c5177e84244575b25fa14a858dd853eff59285b9c44f4a5e2c253579`): the re-issue of the v1 release `sample-artifact-magic-v1` (`2dc82d55161382602fa7a5d1e217e3a33b6c6188ba6ddf4f7ff6b087bf67114c`) with byte-identical training context, the converted base and a recorded `reissue` block, served without `torch.load`. The v1 asset stays published and readable through the deprecated path.

The `.ckpt` digest check establishes byte integrity
against this repository's pinned identity, not producer authenticity or model quality. The DIMER fine-tuner image bakes
the same file at build time (see `DEPLOYMENT.md`).
