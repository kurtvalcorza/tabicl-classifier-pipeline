# Notebook release evidence

Release-grade notebook execution is verified by the `notebook-release-execution` job in `.github/workflows/ci.yml`.

For each pull-request head, the job writes and uploads:

- `release-evidence/tabiclv2_classifier_colab.executed.ipynb`
- `release-evidence/tabiclv2_classifier_artifact_inference_colab.executed.ipynb`
- `release-evidence/notebook-release-evidence.json`

The JSON record identifies the tested revision, clean-run environment, runtime/package versions, exported artifact SHA-256, the separate-kernel producer/consumer boundary, and each notebook outcome. The companion receives the producer artifact and a separate CSV only after a fresh kernel is started.

This automated CPU path covers the committed defaults. It does not claim execution of the optional CUDA fine-tuning branch. A release that specifically changes or claims CUDA fine-tuning behavior must record accelerator-specific execution evidence in the pull request, release note, or another durable test record.

Static validation remains a separate CI job and must not be represented as successful notebook execution.
