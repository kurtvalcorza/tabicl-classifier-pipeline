from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "tutorials" / "tabiclv2_classifier_colab.ipynb"
INF_PATH = ROOT / "tutorials" / "tabiclv2_classifier_artifact_inference_colab.ipynb"


def src(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def set_src(cell: dict, value: str) -> None:
    cell["source"] = value


def find_cell(nb: dict, needle: str, cell_type: str | None = None) -> dict:
    matches = [
        cell for cell in nb["cells"]
        if (cell_type is None or cell.get("cell_type") == cell_type) and needle in src(cell)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one cell containing {needle!r}; found {len(matches)}")
    return matches[0]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find {label}: {old!r}")
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one {label}; found {text.count(old)}")
    return text.replace(old, new, 1)


def write_nb(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def patch_main() -> None:
    nb = json.loads(MAIN_PATH.read_text(encoding="utf-8"))

    opening = src(nb["cells"][0])
    if "**Profile:** `E2E`" not in opening:
        opening = replace_once(
            opening,
            "\n\nYou have a labelled table",
            "\n\n**Profile:** `E2E` · **DIMER Notebook Specification:** `1.0`\n\nYou have a labelled table",
            "main profile insertion point",
        )
    if "This notebook does not demonstrate" not in opening:
        opening = replace_once(
            opening,
            "\n\n> Load checkpoints and bundles only from sources you trust.",
            "\n\n**This notebook does not demonstrate:** production deployment acceptance, benchmark-grade model ranking, or downstream consumption of an independently supplied serving artifact; use the companion `ARTIFACT-INFERENCE` notebook for the latter.\n\n> Load checkpoints and bundles only from sources you trust.",
            "main capability exclusions",
        )
    if "**External access:**" not in opening:
        opening = replace_once(
            opening,
            "\n\nCells with a form on the right",
            "\n- **External access:** PyPI for the pinned Python packages and Hugging Face for the immutable upstream checkpoint on the default path; no credentials are required for those public resources.\n\nCells with a form on the right",
            "main external access prerequisite",
        )
    set_src(nb["cells"][0], opening)

    install_md = find_cell(nb, "## 1. Install and inspect the runtime", "markdown")
    set_src(
        install_md,
        "## 1. Install and inspect the runtime\n\n"
        "The notebook pins every package it installs directly so the release path does not float between package releases. "
        "PyTorch is deliberately treated as part of the Colab/runtime substrate rather than reinstalled after import, which preserves the runtime's tested CPU/CUDA pairing; its exact version is printed and captured in release execution evidence. "
        "The cell verifies all direct pins and prints Python, TabICL, PyTorch, the principal data/ML libraries, and the effective accelerator.\n\n"
        "**What to look for:** `TabICL: 2.1.1`; every `...: <version>` line for a pinned package must match the declared pin. `CUDA: False` is valid for the default pretrained path; CUDA is required only when optional gradient fine-tuning is enabled.\n",
    )

    install_code = find_cell(nb, 'tabicl[finetune]==2.1.1', "code")
    set_src(
        install_code,
        '%pip -q install "tabicl[finetune]==2.1.1" "lightgbm==4.7.0" "pyarrow==25.0.1" "pandas==3.0.5" "numpy==2.5.2" "scikit-learn==1.9.0" "huggingface_hub==1.30.0"\n\n'
        'import importlib.metadata\n'
        'import sys\n\n'
        'import torch\n\n'
        'TABICL_VERSION = "2.1.1"\n'
        'PINNED_PACKAGES = {\n'
        '    "tabicl": TABICL_VERSION,\n'
        '    "lightgbm": "4.7.0",\n'
        '    "pyarrow": "25.0.1",\n'
        '    "pandas": "3.0.5",\n'
        '    "numpy": "2.5.2",\n'
        '    "scikit-learn": "1.9.0",\n'
        '    "huggingface-hub": "1.30.0",\n'
        '}\n'
        'for package, expected in PINNED_PACKAGES.items():\n'
        '    observed = importlib.metadata.version(package)\n'
        '    if observed != expected:\n'
        '        raise RuntimeError(f"Unexpected {package} version: {observed}; expected {expected}")\n'
        'print("Python:", sys.version.split()[0])\n'
        'print("TabICL:", TABICL_VERSION)\n'
        'print("PyTorch:", torch.__version__)\n'
        'for package in ("lightgbm", "pyarrow", "pandas", "numpy", "scikit-learn", "huggingface-hub"):\n'
        '    print(f"{package}:", PINNED_PACKAGES[package])\n'
        'print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")\n',
    )

    model_code = find_cell(nb, "def safe_single_ckpt", "code")
    model_text = src(model_code)
    if "MAX_INPUT_ARCHIVE_EXPANDED_BYTES" not in model_text:
        model_text = replace_once(
            model_text,
            'WORK_DIR = Path("/content/tabiclv2-classifier")\nWORK_DIR.mkdir(parents=True, exist_ok=True)\n',
            'WORK_DIR = Path("/content/tabiclv2-classifier")\nWORK_DIR.mkdir(parents=True, exist_ok=True)\nMAX_INPUT_ARCHIVE_MEMBERS = 1000\nMAX_INPUT_ARCHIVE_EXPANDED_BYTES = 2 * 1024**3\n',
            "main input archive limits",
        )
        model_text = replace_once(
            model_text,
            '    with zipfile.ZipFile(zip_path) as archive:\n        checkpoint_members = []\n        for info in archive.infolist():\n            name = info.filename.replace("\\\\", "/")\n',
            '    with zipfile.ZipFile(zip_path) as archive:\n        infos = archive.infolist()\n        if len(infos) > MAX_INPUT_ARCHIVE_MEMBERS:\n            raise ValueError(f"Archive has too many members: {len(infos)}")\n        expanded_bytes = sum(info.file_size for info in infos if not info.is_dir())\n        if expanded_bytes > MAX_INPUT_ARCHIVE_EXPANDED_BYTES:\n            raise ValueError(f"Archive expands to {expanded_bytes} bytes, above the {MAX_INPUT_ARCHIVE_EXPANDED_BYTES}-byte limit")\n        checkpoint_members = []\n        for info in infos:\n            if "\\\\" in info.filename:\n                raise ValueError(f"Ambiguous backslash path in ZIP member: {info.filename}")\n            name = info.filename\n',
            "main safe checkpoint archive loop",
        )
    set_src(model_code, model_text)

    data_md = find_cell(nb, "## 3. Load sample or BYOD data", "markdown")
    data_text = src(data_md)
    if "### BYOD schema and data handling" not in data_text:
        data_text += (
            "\n\n### BYOD schema and data handling\n\n"
            "Before upload, the expected schema is explicit: **Upload CSV** requires one labelled CSV containing `TARGET_COLUMN` (default `target`) plus one or more feature columns; **pre-split** mode requires `train.csv` and `val.csv`, with optional `test.csv`, and every split must carry the same feature names plus the target. The code rejects duplicate headers, missing targets/features, unseen target classes, and incompatible split schemas rather than guessing.\n\n"
            "**Data handling / privacy.** `files.upload()` sends selected files from your browser to the active Google Colab runtime. Uploaded rows are processed inside that runtime and are **not sent to Hugging Face or to an external inference API**; the Hugging Face network request downloads only the pinned model checkpoint. `files.download()` returns generated outputs from the runtime to your browser. Do not upload confidential, restricted, sensitive, or regulated data unless you are authorized to process it in the selected runtime.\n"
        )
    set_src(data_md, data_text)

    eval_md = find_cell(nb, "## 4. Evaluate pretrained TabICLv2", "markdown")
    eval_text = src(eval_md)
    if "Reproducibility and remaining variability" not in eval_text:
        eval_text += (
            "\n\n**Reproducibility and remaining variability.** `RANDOM_SEED` fixes the random split, TabICL ensemble seed, optional fine-tuning seed, and the companion tree-model seeds used for reported tutorial evidence. It does not guarantee bitwise identity across hardware: CUDA kernels, low-level numerical libraries, upstream framework implementations, scheduler/runtime load, and wall-clock latency can still vary. Treat small metric or latency differences across runtime images as run-to-run variation unless independently reproduced; release evidence records the effective environment.\n"
        )
    set_src(eval_md, eval_text)

    inference_code = find_cell(nb, "RUN_NEW_DATA_INFERENCE = False", "code")
    inference_text = src(inference_code)
    if "Expected inference CSV feature columns" not in inference_text:
        inference_text = replace_once(
            inference_text,
            'RUN_NEW_DATA_INFERENCE = False  # @param {type:"boolean"}\n\n',
            'RUN_NEW_DATA_INFERENCE = False  # @param {type:"boolean"}\nprint("Expected inference CSV feature columns:", FEATURE_COLUMNS)\n\n',
            "main inference schema print",
        )
    set_src(inference_code, inference_text)

    export_code = find_cell(nb, "ARTIFACT_DIR = Path", "code")
    export_text = src(export_code)
    if '"sizes":' not in export_text:
        export_text = replace_once(
            export_text,
            '    "digests":{"checkpointSha256":sha256_file(export_ckpt),"trainingContextSha256":sha256_file(context_path)},\n',
            '    "digests":{"checkpointSha256":sha256_file(export_ckpt),"trainingContextSha256":sha256_file(context_path)},\n    "sizes":{"checkpointBytes":export_ckpt.stat().st_size,"trainingContextBytes":context_path.stat().st_size},\n',
            "main artifact size manifest",
        )
    set_src(export_code, export_text)

    reload_code = find_cell(nb, "RELOAD_DIR = Path", "code")
    reload_text = src(reload_code)
    if "MAX_ARTIFACT_EXPANDED_BYTES" not in reload_text:
        reload_text = replace_once(
            reload_text,
            'RELOAD_DIR = Path("/content/tabiclv2-classifier-reload")\n',
            'MAX_ARTIFACT_MEMBERS = 1000\nMAX_ARTIFACT_EXPANDED_BYTES = 2 * 1024**3\n\nRELOAD_DIR = Path("/content/tabiclv2-classifier-reload")\n',
            "main reload archive limits",
        )
        reload_text = replace_once(
            reload_text,
            'with zipfile.ZipFile(archive_path) as archive:\n    root = RELOAD_DIR.resolve()\n    for info in archive.infolist():\n        name = info.filename.replace("\\\\", "/")\n',
            'with zipfile.ZipFile(archive_path) as archive:\n    infos = archive.infolist()\n    if len(infos) > MAX_ARTIFACT_MEMBERS:\n        raise ValueError(f"Artifact has too many members: {len(infos)}")\n    expanded_bytes = sum(info.file_size for info in infos if not info.is_dir())\n    if expanded_bytes > MAX_ARTIFACT_EXPANDED_BYTES:\n        raise ValueError(f"Artifact expands to {expanded_bytes} bytes, above the {MAX_ARTIFACT_EXPANDED_BYTES}-byte limit")\n    root = RELOAD_DIR.resolve()\n    for info in infos:\n        if "\\\\" in info.filename:\n            raise ValueError(f"Ambiguous backslash path in artifact member: {info.filename}")\n        name = info.filename\n',
            "main reload safe extraction loop",
        )
        reload_text = replace_once(
            reload_text,
            'served = json.loads((RELOAD_DIR / "artifact.json").read_text())\n',
            'observed_files = {path.relative_to(RELOAD_DIR).as_posix() for path in RELOAD_DIR.rglob("*") if path.is_file()}\nexpected_files = {"artifact.json", "checkpoints/best.ckpt", "training_context.parquet"}\nif observed_files != expected_files:\n    raise ValueError(f"Unexpected artifact file set: {sorted(observed_files ^ expected_files)}")\n\nserved = json.loads((RELOAD_DIR / "artifact.json").read_text())\n',
            "main reload file allowlist",
        )
        reload_text = replace_once(
            reload_text,
            'if sha256_file(served_ckpt) != served["digests"]["checkpointSha256"] or sha256_file(served_context) != served["digests"]["trainingContextSha256"]:\n    raise RuntimeError("Artifact digest mismatch")\n',
            'if sha256_file(served_ckpt) != served["digests"]["checkpointSha256"] or sha256_file(served_context) != served["digests"]["trainingContextSha256"]:\n    raise RuntimeError("Artifact digest mismatch")\nif served_ckpt.stat().st_size != served["sizes"]["checkpointBytes"] or served_context.stat().st_size != served["sizes"]["trainingContextBytes"]:\n    raise RuntimeError("Artifact size mismatch")\n',
            "main reload size verification",
        )
    set_src(reload_code, reload_text)

    nb.setdefault("metadata", {}).setdefault("dimer", {})
    nb["metadata"]["dimer"]["notebook_profile"] = "E2E"
    nb["metadata"]["dimer"]["notebook_spec"] = "1.0"
    write_nb(MAIN_PATH, nb)


def patch_inference() -> None:
    nb = json.loads(INF_PATH.read_text(encoding="utf-8"))

    opening = src(nb["cells"][0])
    if "**Profile:** `ARTIFACT-INFERENCE`" not in opening:
        opening = replace_once(
            opening,
            "\n\nSomeone hands you a TabICL serving bundle",
            "\n\n**Profile:** `ARTIFACT-INFERENCE` · **DIMER Notebook Specification:** `1.0`\n\nSomeone hands you a TabICL serving bundle",
            "inference profile insertion point",
        )
    if "This notebook does not demonstrate" not in opening:
        opening = replace_once(
            opening,
            "\n\n> **Trust boundary:**",
            "\n\n**This notebook does not demonstrate:** gradient fine-tuning, model selection, or creation of the artifact it consumes. The artifact must be supplied from outside this notebook execution.\n\n> **Trust boundary:**",
            "inference capability exclusions",
        )
    opening = opening.replace(
        "- Any Colab runtime; inference runs on CPU. About two minutes end to end.",
        "- **Runtime:** CPU is sufficient; when CUDA is available the estimator may use it automatically. About two minutes end to end on the small verification batch.",
    )
    set_src(nb["cells"][0], opening)

    install_md = find_cell(nb, "## 1. Install", "markdown")
    set_src(
        install_md,
        "## 1. Install and verify the matching runtime\n\n"
        "The directly installed packages are pinned exactly. PyTorch is treated as part of the runtime substrate so the notebook does not replace an already provisioned CPU/CUDA build; its effective version is printed. The artifact's recorded TabICL version is checked against the pinned estimator before deserialization.\n",
    )

    install_code = find_cell(nb, 'tabicl==2.1.1', "code")
    set_src(
        install_code,
        '%pip -q install "tabicl==2.1.1" "pyarrow==25.0.1" "pandas==3.0.5" "numpy==2.5.2" "scikit-learn==1.9.0"\n\n'
        'import importlib.metadata\n'
        'import sys\n\n'
        'import torch\n\n'
        'TABICL_VERSION = "2.1.1"\n'
        'PINNED_PACKAGES = {\n'
        '    "tabicl": TABICL_VERSION,\n'
        '    "pyarrow": "25.0.1",\n'
        '    "pandas": "3.0.5",\n'
        '    "numpy": "2.5.2",\n'
        '    "scikit-learn": "1.9.0",\n'
        '}\n'
        'for package, expected in PINNED_PACKAGES.items():\n'
        '    observed = importlib.metadata.version(package)\n'
        '    if observed != expected:\n'
        '        raise RuntimeError(f"Unexpected {package} version: {observed}; expected {expected}")\n'
        'print("Python:", sys.version.split()[0])\n'
        'print("TabICL:", TABICL_VERSION)\n'
        'print("PyTorch:", torch.__version__)\n'
        'for package in ("pyarrow", "pandas", "numpy", "scikit-learn"):\n'
        '    print(f"{package}:", PINNED_PACKAGES[package])\n'
        'print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")\n',
    )

    upload_md = find_cell(nb, "## 2. Upload and verify the artifact ZIP", "markdown")
    upload_text = src(upload_md)
    if "Data handling / privacy" not in upload_text:
        upload_text += (
            "\n\n**Data handling / privacy.** Selecting the artifact sends its bytes—including the persisted labelled training context—to the active Google Colab runtime. The notebook does not send the artifact, its embedded rows, or later inference rows to an external inference API or to Hugging Face; model auto-download is disabled. Package installation still contacts the configured Python package index. Do not upload confidential, restricted, sensitive, or regulated artifacts unless you are authorized to process them in the selected runtime.\n"
        )
    set_src(upload_md, upload_text)

    upload_code = find_cell(nb, "def safe_extract_zip", "code")
    upload_code_text = src(upload_code)
    if "MAX_ARCHIVE_EXPANDED_BYTES" not in upload_code_text:
        upload_code_text = replace_once(
            upload_code_text,
            'EXPECTED_ZIP_SHA256 = ""  # @param {type:"string"}\n',
            'EXPECTED_ZIP_SHA256 = ""  # @param {type:"string"}\nMAX_ARCHIVE_MEMBERS = 1000\nMAX_ARCHIVE_EXPANDED_BYTES = 2 * 1024**3\n',
            "inference archive limits",
        )
        upload_code_text = replace_once(
            upload_code_text,
            '    with zipfile.ZipFile(zip_path) as archive:\n        for info in archive.infolist():\n            name = info.filename.replace("\\\\", "/")\n',
            '    with zipfile.ZipFile(zip_path) as archive:\n        infos = archive.infolist()\n        if len(infos) > MAX_ARCHIVE_MEMBERS:\n            raise ValueError(f"Artifact has too many members: {len(infos)}")\n        expanded_bytes = sum(info.file_size for info in infos if not info.is_dir())\n        if expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES:\n            raise ValueError(f"Artifact expands to {expanded_bytes} bytes, above the {MAX_ARCHIVE_EXPANDED_BYTES}-byte limit")\n        for info in infos:\n            if "\\\\" in info.filename:\n                raise ValueError(f"Ambiguous backslash path in ZIP member: {info.filename}")\n            name = info.filename\n',
            "inference safe extraction loop",
        )
        upload_code_text = replace_once(
            upload_code_text,
            'if sha256_file(context_path) != manifest["digests"]["trainingContextSha256"]:\n    raise RuntimeError("Training-context digest mismatch")\nprint("✓ Artifact structure and digests verified")\n',
            'if sha256_file(context_path) != manifest["digests"]["trainingContextSha256"]:\n    raise RuntimeError("Training-context digest mismatch")\n\nsizes = manifest.get("sizes")\nif sizes is not None:\n    if ckpt.stat().st_size != sizes.get("checkpointBytes"):\n        raise RuntimeError("Checkpoint size mismatch")\n    if context_path.stat().st_size != sizes.get("trainingContextBytes"):\n        raise RuntimeError("Training-context size mismatch")\nelse:\n    print("⚠ Legacy DIMER v1 manifest has no checkpointBytes/trainingContextBytes; global archive limits and SHA-256 checks still apply.")\n\nroot_prefix = root.relative_to(extract_dir)\nexpected_files = {\n    (root_prefix / "artifact.json").as_posix(),\n    (root_prefix / Path(manifest["checkpoint"])).as_posix(),\n    (root_prefix / Path(manifest["trainingContext"])).as_posix(),\n}\nobserved_files = {path.relative_to(extract_dir).as_posix() for path in extract_dir.rglob("*") if path.is_file()}\nunexpected_files = sorted(observed_files - expected_files)\nif unexpected_files:\n    print("⚠ Unexpected unlisted artifact file(s) retained for DIMER v1 compatibility:", unexpected_files)\nprint("✓ Artifact structure and digests verified")\n',
            "inference digest/size/file-set verification",
        )
    set_src(upload_code, upload_code_text)

    reconstruct_md = find_cell(nb, "## 3. Reconstruct the in-context classifier", "markdown")
    reconstruct_text = src(reconstruct_md)
    if "remaining run-to-run variability" not in reconstruct_text:
        reconstruct_text += (
            "\n\nThe producer's recorded `randomState` is restored. Remaining run-to-run variability can come from the effective PyTorch/CUDA build, hardware-specific numerical kernels, and runtime scheduling; the seed does not imply bitwise identity across machines. No gradient optimization occurs in this notebook.\n"
        )
    set_src(reconstruct_md, reconstruct_text)

    reconstruct_code = find_cell(nb, "DEVICE =", "code")
    reconstruct_code_text = src(reconstruct_code)
    if "Expected inference CSV feature columns" not in reconstruct_code_text:
        reconstruct_code_text = replace_once(
            reconstruct_code_text,
            'model.fit(context[FEATURE_COLUMNS], context[TARGET_COLUMN])\nprint(f"✓ Loaded with {len(context)} context rows and {len(FEATURE_COLUMNS)} features; classes: {list(model.classes_)}")\n',
            'model.fit(context[FEATURE_COLUMNS], context[TARGET_COLUMN])\nprint(f"✓ Loaded with {len(context)} context rows and {len(FEATURE_COLUMNS)} features; classes: {list(model.classes_)}")\nprint("Effective inference device:", DEVICE)\nprint("Expected inference CSV feature columns:", FEATURE_COLUMNS)\n',
            "inference schema/device print",
        )
    set_src(reconstruct_code, reconstruct_code_text)

    predict_md = find_cell(nb, "## 4. Upload rows and predict", "markdown")
    predict_text = src(predict_md)
    if "Before upload, use the exact feature list" not in predict_text:
        predict_text += (
            "\n\nBefore upload, use the exact feature list printed by Step 3. The inference CSV does not need the target column. Uploading the CSV sends it to the active Google Colab runtime; prediction is performed locally in that runtime with `allow_auto_download=False`, and the only outbound action after scoring is the explicit download of `predictions.csv` back to your browser.\n"
        )
    set_src(predict_md, predict_text)

    final_md = nb["cells"][-1]
    final_text = src(final_md)
    if "## Interpretation and limits" not in final_text:
        final_text += (
            "\n\n## Interpretation and limits\n\n"
            "A successful run establishes that the externally supplied archive passed the documented path/member/expanded-size checks, that its manifest format and TabICL version are compatible, that the checkpoint and training-context digests match the manifest, that any recorded component sizes match, that the serving state reconstructs without network model fallback, and that a schema-compatible new CSV can be scored and exported. If you supplied an out-of-band `EXPECTED_ZIP_SHA256`, success also establishes that the whole archive matches that producer-provided digest.\n\n"
            "It **does not prove** that an artifact is trustworthy when no authentic out-of-band digest or trusted producer relationship exists; internal consistency can be forged together. It also does not establish calibration, fairness, production fitness, or distributional compatibility of the new rows—schema-valid data can still be out of distribution. Treat large unseen-category warnings as drift signals and validate decision thresholds and task-specific costs on appropriate held-out data before consequential use.\n\n"
            "**Next:** reproduce known holdout predictions from the producer, test representative new batches, investigate schema/drift warnings, and validate calibration/threshold behavior for the intended deployment population.\n"
        )
    set_src(final_md, final_text)

    nb.setdefault("metadata", {}).setdefault("dimer", {})
    nb["metadata"]["dimer"]["notebook_profile"] = "ARTIFACT-INFERENCE"
    nb["metadata"]["dimer"]["notebook_spec"] = "1.0"
    write_nb(INF_PATH, nb)


if __name__ == "__main__":
    patch_main()
    patch_inference()
    print("Applied DIMER NOTEBOOK_SPEC v1.0 migration to both TabICLv2 classifier notebooks")
