# Weight provenance and local snapshot layout

| Field | Value |
|---|---|
| Upstream model | `jingang/TabICL` (Hugging Face) |
| Checkpoint | `tabicl-classifier-v2-20260212.ckpt` (110,368,038 bytes) |
| Immutable revision | `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` |
| SHA-256 | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |
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

The checkpoint is a Lightning/PyTorch `.ckpt` deserialised by `tabicl`; the digest check establishes byte integrity
against this repository's pinned identity, not producer authenticity or model quality. The DIMER fine-tuner image bakes
the same file at build time (see `DEPLOYMENT.md`).
