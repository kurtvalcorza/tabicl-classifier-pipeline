"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment) — E2E.

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, checkpoint
staging) from repository files; this template holds the learner-facing prose, the guided layer and the learner
cells. Every learner cell calls ``run_stage(...)``: the carried ``tools/tutorial_stages.py`` runs one stage per process
in an isolated, hash-locked environment, so nothing is installed into the notebook kernel. The ARTIFACT-INFERENCE
companion has its own template, ``tools/notebook_template_artifact_inference.py``.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "tabicl-classifier-pipeline"
BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/tabiclv2_classifier_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jingang%2FTabICL-ffcc4d?style=flat",
        "https://huggingface.co/jingang/TabICL",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-soda--inria%2Ftabicl-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/soda-inria/tabicl",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2602.11139-b31b1b.svg", "https://arxiv.org/abs/2602.11139"),
]


UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}
ENVIRONMENT = {
    "package": "tabicl_classifier_pipeline",
    "repo_name": REPO,
    "weights_key": "tabicl-classifier-v2",
    "modules": ["__init__.py", "api.py"],
    "entry_module": "api.py",
    "lock": "tutorials/requirements-colab.lock.txt",
    "managed_python": "3.12.12",
    "uv": UV,
    "disk_gib": {"weights": 0.2, "environment": 8.0},
    "runtime_modules": ["torch", "tabicl", "numpy", "pandas", "scikit-learn", "lightgbm"],
    "install_flags": ["--only-binary", ":all:"],
}

RUNTIME_PREREQ = (
    "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab with a **T4 GPU** is the documented runtime (the bounded fine-tuning stage needs CUDA); Kaggle or a Linux Jupyter kernel also work, and a CPU-only runtime runs everything except fine-tuning, which it skips with a printed note. The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`tabicl` 2.1.1 with its fine-tuning extras, `torch` 2.14.0 with its CUDA libraries, `numpy` 2.5.2, `pandas` 3.0.5, `scikit-learn` 1.9.0, `lightgbm` 4.7.0). About 0.2 GB of disk is needed for the checkpoint and about 8 GB for the isolated environment."
)

TEMPLATE = {
    **ENVIRONMENT,
    "stem": "tabiclv2_classifier",
    "notebook_name": "tabiclv2_classifier_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages.py",
    "carried_extra": {"sample-bundle/SAMPLE_BUNDLE.json": "examples/sample-bundle/SAMPLE_BUNDLE.json"},
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime (a T4 GPU is recommended) builds an isolated Python environment from the carried hash-locked requirements without touching the notebook kernel's own packages, then runs each stage below in its own process: it stages and digest-verifies the pinned TabICLv2 checkpoint, loads scikit-learn's bundled Breast Cancer Wisconsin table (no download), splits it 60/20/20 into support, holdout and independent test partitions with class coverage preserved, validates the tables into an input manifest and encodes them, evaluates the pretrained TabICLv2 classifier **adapted by in-context conditioning on the support rows**, then runs the pipeline's production adaptation — **gradient fine-tuning**, bounded to 3 epochs and 5 minutes, with the candidate selected on the holdout only if it beats the pretrained model — compares LightGBM, a random forest and a majority-class baseline, writes the evaluation report, scores new rows, exports a DIMER-style serving bundle and reloads it in a fresh process. On a runtime without CUDA the fine-tuning stage is skipped with a printed note (TabICL fine-tuning needs a GPU) and everything else runs. No repository clone, DIMER worker or service, credential, upload dialog, configuration edit or runtime restart is required (NOTEBOOK_SPEC 2.2 §5). No hosted run of this revision has been recorded yet; the fine-tuning branch has never been executed in a recorded run."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` and `DATA_SOURCE` to `Upload CSV` or `Upload pre-split train/val/test` in Section 4 and run Sections 4–9 again. Give the files by path (`BYOD_CSV_PATH`, or `BYOD_TRAIN_PATH` / `BYOD_VAL_PATH` / optional `BYOD_TEST_PATH` — these work in Colab, Kaggle and Jupyter), or leave the paths empty in Colab to use upload dialogs. Name your label column in `TARGET_COLUMN`. A missing label column is refused naming it; rows with a missing label are dropped **and counted in the input manifest**; a column that is numeric except for a few stray strings is refused naming the column and the values (list it in `TEXT_COLUMNS` if it really is categorical). `RUN_NEW_DATA_INFERENCE` in Section 8 scores your own unlabelled rows. Files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "title": "TabICLv2 Classifier — DIMER E2E tabular classification tutorial (standalone)",
    "badges": BADGES,
    "capability": "end-to-end TabICLv2 tabular classification: immutable checkpoint acquisition, validated support data with class coverage, in-context evaluation and bounded gradient fine-tuning against trivial and classical baselines, new-data inference with class probabilities, a DIMER-style serving bundle and its fresh reload",
    "intro": (
        "TabICLv2 (Qu et al.) is an **in-context tabular foundation model**: a transformer pretrained on synthetic tables "
        "to classify a query row by reading labelled example rows — the **support rows** or **context** — alongside it. "
        "`fit` registers the (encoded) support rows as context and every `predict_proba` call feeds context plus query rows "
        "through the transformer and reads off a class distribution per query row; no gradient step happens there. "
        "`n_estimators=8` averages eight passes with different feature and row permutations. The DIMER pipeline's production "
        "workflow goes one step further: it **fine-tunes** the checkpoint by gradient descent on the support rows (AdamW, "
        "early stopping on the holdout) and keeps the fine-tuned model only if it beats the pretrained one on the holdout. "
        "This notebook runs both: conditioning by default, and a bounded fine-tune when a GPU is present.\n\n"
        "The upstream project supplies the model and the checkpoint; the carried package adds the pinned snapshot scheme, "
        "the table preparation, class-coverage and encoding rules, the metric set, the `validate_inputs` / "
        "`majority_class_baseline` / `evaluation_report` helpers, and the serving-bundle safety checks. `predict` applies an "
        "implicit `argmax` over class probabilities that are raw ensemble outputs, **not calibrated probabilities**; the "
        "package ships no acceptance threshold. The default sample is scikit-learn's bundled Breast Cancer Wisconsin table; "
        "it is nearly separable (LightGBM and a random forest already score 0.96–0.98), so its metrics are tutorial sanity "
        "evidence, not a benchmark or production claim."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** the difference between in-context conditioning (registering support rows) and gradient fine-tuning, "
        "and why the candidate is selected on the holdout, never on the independent test (Section 6).\n"
        "2. **Diagnose** an invalid table from a validation refusal, and say what a dropped-row count in the input manifest "
        "means (Sections 4, 5).\n"
        "3. **Compare** TabICLv2 with a majority-class baseline, LightGBM and a random forest on the same partitions, using "
        "error counts and the one-row resolution (Section 7).\n"
        "4. **Interpret** accuracy, balanced accuracy, weighted F1, log loss and ROC-AUC, and why a blend weight chosen on the "
        "holdout is selection-biased (Section 7).\n"
        "5. **Apply** the selected model to new rows and read class-ordered, uncalibrated probabilities under an explicit "
        "`argmax` rule (Section 8).\n"
        "6. **Verify** that the exported bundle rebuilds an equivalent classifier in a fresh process (Section 9).\n"
        "7. **Predict**, run and **explain** how a three-class table with 36 holdout rows changes the one-row resolution and "
        "ROC-AUC, in an optional activity (Section 10).\n"
        "8. **Write** an evidence-based conclusion that names the baselines, the selection basis and the limits of one public "
        "table (Conclusion)."
    ),
    "exclusions": (
        "regression, forecasting, calibrated probabilities, or any deployment threshold. `prediction` is the `argmax` over "
        "uncalibrated class probabilities; fine-tuning runs only on CUDA, and its selection is evidence on one holdout, not "
        "a guarantee of better generalisation."
    ),
    "prerequisites": [
        RUNTIME_PREREQ,
        "- **Knowledge:** basic pandas, what a train/holdout/test split is, and how to read a printed Python dictionary. The metrics, in-context conditioning and fine-tuning are explained where they are first used, and the glossary collects them.",
        "- **Model file:** one checkpoint, `tabicl-classifier-v2-20260212.ckpt` (110 MB, released under BSD-3-Clause). It is a PyTorch/Lightning checkpoint: `tabicl` deserialises it, so only the digest-verified file is ever loaded.",
        "- **Data:** the default sample is scikit-learn's bundled Breast Cancer Wisconsin table (569 rows, 30 numeric features, two classes), loaded from the installed package, so nothing is downloaded and no private data is needed; `Sample: Wine` is the bundled three-class table and `Sample: Palmer Penguins` fetches the repository's pinned CC0 archive (mixed categorical/numeric features, pre-split) by immutable commit and verifies its SHA-256. BYOD (one CSV with a label column, or pre-split `train`/`val`/`test` CSVs) is off by default so the sample path runs top to bottom without interaction. The tutorial's own floors are `MIN_TRAIN_ROWS` = 50 support rows and `MIN_EVAL_ROWS` = 2 holdout rows, with no class ceiling; the production DIMER validator described in the model card is stricter (10 evaluation rows, at most 1,000 classes). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook (Google Colab or Jupyter) and read "
                "short Python and pandas, and who want to see a tabular foundation model used the way a production pipeline uses "
                "it: conditioned, optionally fine-tuned with honest holdout selection, compared with strong classical models, and "
                "packaged as a reusable bundle. No experience with transformers is assumed; each term is explained where it is "
                "first needed, and the glossary below collects them.\n\n"
                "**Running it.** In Colab choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The default "
                "path needs no edit, no upload, no account, no token and no runtime restart. Section 2 builds an isolated "
                "environment, which takes the longest (PyTorch and its CUDA libraries are a few GB); Section 6 then spends up to "
                "five minutes fine-tuning on a GPU. You can also run one cell at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files: the verified checkpoint, the partitions and fitted "
                "encoders, the fine-tuned checkpoint if one was selected, and JSON records.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–10) are the machine-learning workflow. *Infrastructure cells* "
                "(Sections 1–3: the runtime check, the carried code, the isolated environment and the checkpoint staging) are "
                "collapsed and titled **Infrastructure**. You may run them without studying their implementation: they exist for "
                "reproducibility and provenance, not as prerequisite machine-learning knowledge.\n\n"
                "**Form controls.** `DATA_SOURCE`, `USE_BYOD`, `TARGET_COLUMN`, the `BYOD_*_PATH` fields and `TEXT_COLUMNS` "
                "(Section 4); `RUN_FINE_TUNING`, `FINE_TUNE_EPOCHS` and `EVAL_METRIC` (Section 6); `RUN_NEW_DATA_INFERENCE` and "
                "`NEW_DATA_PATH` (Section 8); `RUN_ACTIVITY` and `ACTIVITY_DATA_SOURCE` (Section 10). Leave them at their defaults "
                "for the first run.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 6 and 7 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint. GPU kernels are not bitwise deterministic, so the last digits can vary between runs."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Validate and split** | a labelled table | `validate_inputs`, a stratified 60/20/20 split, support-fitted categorical encoders | an input manifest; 341 support, 114 holdout and 114 test rows |\n"
                "| **Condition** | the support rows | TabICLv2 reading them as context (`n_estimators=8`) | class probabilities for any query row |\n"
                "| **Fine-tune (GPU)** | support rows; holdout for early stopping | gradient fine-tuning of the checkpoint, ≤ 3 epochs / 5 min | a candidate checkpoint, kept only if it beats the pretrained model on the holdout |\n"
                "| **Evaluate** | holdout and test rows | `classification_metrics`, majority baseline, LightGBM, random forest, an in-memory blend | metrics, error counts, an evaluation report |\n"
                "| **Package** | selected checkpoint + support rows + settings | ZIP bundle, `safe_extract_zip`, `verify_artifact_bundle` | a bundle that rebuilds an equivalent classifier |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux x86_64, GPU and disk; a run directory | the machine |\n"
                "| 2. Carry the code, build the environment | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage and verify the model | [Engineering] | the 110 MB checkpoint downloaded at a fixed revision, digest-checked | identity and digest |\n"
                "| 4. Load the sample or your own data | [Concept] | the Breast Cancer table (or your files), split 60/20/20 | partition sizes, digest |\n"
                "| 5. Validate and encode | [Evaluation practice] | input manifest; a refusal probe; encoders; majority baseline | the manifest |\n"
                "| 6. Condition and fine-tune | [Concept] | pretrained metrics; bounded fine-tuning and holdout selection | the selection |\n"
                "| 7. Baselines, blend and report | [Evaluation practice] | LightGBM, random forest, blend; the evaluation report | the principal result |\n"
                "| 8. New-data inference | [Engineering] | eight rows (or your file) scored | the output contract |\n"
                "| 9. Export and fresh reload | [Engineering] | the bundle exported and rebuilt in a new process | the reload check and the digest |\n"
                "| 10. Optional activity | [Concept] | the three-class Wine table (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common failures and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Run all, then read Sections 6, 7 and 9 and the conclusion."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **In-context learning / conditioning** | Predicting from labelled rows shown to the model at prediction time; `fit` only registers them. |\n"
                "| **Support rows / context** | The labelled rows the model reads; here the 341-row support partition. |\n"
                "| **`n_estimators`** | How many passes with different feature/row permutations are averaged into one prediction. |\n"
                "| **Fine-tuning** | Updating the checkpoint's weights by gradient descent on the support rows. |\n"
                "| **Early stopping** | Stopping fine-tuning when the holdout metric stops improving. |\n"
                "| **Holdout selection** | Keeping the fine-tuned candidate only if it beats the pretrained model on the holdout. |\n"
                "| **Independent test** | Rows used by no selection at all: the only unbiased estimate here. |\n"
                "| **Selection bias** | A score looks better than it will on new data because a choice was made by looking at it. |\n"
                "| **One-row resolution** | 1 / (rows in the partition): the smallest possible change in accuracy. 114 rows → 0.0088. |\n"
                "| **Majority-class baseline** | Always predict the most frequent support class. |\n"
                "| **LightGBM / random forest** | Strong classical tree ensembles trained on the same support rows. |\n"
                "| **Blend** | A weighted average of TabICLv2's and LightGBM's probabilities, weight chosen on the holdout. |\n"
                "| **Accuracy / balanced accuracy / weighted F1** | Discrete correctness under the `argmax` rule; balanced accuracy averages per-class recall. |\n"
                "| **Log loss** | Average negative log probability of the true class; lower is better; punishes confident mistakes. |\n"
                "| **ROC-AUC (one-vs-rest)** | Ranking quality, threshold-free; for several classes, averaged over one-class-vs-rest problems. |\n"
                "| **Uncalibrated probability** | A score not guaranteed to match observed frequencies. |\n"
                "| **Categorical encoder** | Ordinal codes fitted on the support rows only; unseen values get an 'unknown' code and are counted. |\n"
                "| **Serving bundle** | A ZIP of `artifact.json`, `checkpoints/best.ckpt` and `training_context.parquet`. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes. |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n"
                "| **BYOD** | Bring Your Own Data. |\n\n"
                "</details>"
            ),
        ],
    },
    "cells": [
        {
            "md": (
                "## 4. Load the sample or your own data · [Concept]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`. `Sample: Breast Cancer` "
                "(default) is the bundled binary sanity check, split 60/20/20 with stratification into support, holdout and an "
                "independent test partition; `Sample: Wine` is the bundled three-class numeric table split the same way; `Sample: "
                "Palmer Penguins` is the repository's pinned CC0 archive with mixed categorical/numeric features and its own "
                "train/val/test split, fetched by immutable commit and digest-verified before it is read (see "
                "`examples/sample-data/DATASET_CARD.md`). With `USE_BYOD = True`: `Upload CSV` takes one CSV and carves a "
                "stratified 80/20 holdout (`check_stratifiable` needs at least `MIN_ROWS_PER_CLASS` rows per class); `Upload "
                "pre-split train/val/test` takes your own partitions (`test` optional). Give files by path, or leave the path "
                "fields empty in Colab to choose them in upload dialogs.\n\n"
                "The stage refuses — naming the file and the rule — duplicate header names, a `TARGET_COLUMN` absent from the "
                "header (the message lists it), and a column that is numeric except for a few stray strings (it would otherwise be "
                "encoded as categories without a word; list it in `TEXT_COLUMNS` if it really is categorical). Rows whose label is "
                "missing are **dropped and counted**: for a single CSV the count is carried into Section 5's input manifest as "
                "`dropped_missing_target_rows_before_split`. The stage first removes this notebook's earlier exports from "
                "`outputs/`, so a refused input never leaves an older result looking current.\n\n"
                "**BYOD privacy boundary.** Uploaded CSV bytes are read inside the current notebook runtime and are not sent "
                "by this notebook to an external inference or training service; the network requests on the default path "
                "are the pinned checkpoint download of Section 3 and, for `Sample: Palmer Penguins`, one pinned archive."
            ),
            "code": (
                "DATA_SOURCE = 'Sample: Breast Cancer'  # @param [\"Sample: Breast Cancer\", \"Sample: Wine\", \"Sample: Palmer Penguins\", \"Upload CSV\", \"Upload pre-split train/val/test\"]\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "TARGET_COLUMN = 'target'  # @param {{type:\"string\"}}\n"
                "BYOD_CSV_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_TRAIN_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_VAL_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_TEST_PATH = ''  # @param {{type:\"string\"}}\n"
                "TEXT_COLUMNS = []  # @param {{type:\"raw\"}}\n\n"
                "def upload_files(what):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'No path was given for {{what}}, and the upload dialog exists only in Google Colab: set the BYOD path field(s).') from None\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise RuntimeError(f'The upload was cancelled or empty: no file was received for {{what}}. Run this cell again, or set the path field(s).')\n"
                "    saved = {{}}\n"
                "    for upload_name, payload in uploaded.items():\n"
                "        path = ROOT / 'inputs' / Path(upload_name).name\n"
                "        path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        path.write_bytes(payload)\n"
                "        saved[path.name.lower()] = str(path)\n"
                "    return saved\n\n"
                "paths = {{}}\n"
                "if USE_BYOD and DATA_SOURCE == 'Upload CSV':\n"
                "    if BYOD_CSV_PATH:\n"
                "        paths = {{'csv': BYOD_CSV_PATH}}\n"
                "    else:\n"
                "        saved = upload_files('one labelled CSV')\n"
                "        if len(saved) != 1:\n"
                "            raise ValueError(f'Upload exactly one CSV; got {{sorted(saved)}}.')\n"
                "        paths = {{'csv': next(iter(saved.values()))}}\n"
                "elif USE_BYOD and DATA_SOURCE == 'Upload pre-split train/val/test':\n"
                "    if BYOD_TRAIN_PATH or BYOD_VAL_PATH:\n"
                "        paths = {{'train': BYOD_TRAIN_PATH, 'val': BYOD_VAL_PATH, 'test': BYOD_TEST_PATH}}\n"
                "    else:\n"
                "        saved = upload_files('train.csv and val.csv (test.csv optional)')\n"
                "        if not {{'train.csv', 'val.csv'}} <= set(saved):\n"
                "            raise ValueError(f'Upload train.csv and val.csv (test.csv optional); got {{sorted(saved)}}.')\n"
                "        paths = {{'train': saved['train.csv'], 'val': saved['val.csv'], 'test': saved.get('test.csv', '')}}\n"
                "run_stage('data', data_source=DATA_SOURCE, use_byod=USE_BYOD, target_column=TARGET_COLUMN, paths=paths, text_columns=TEXT_COLUMNS)"
            ),
        },
        {
            "md": (
                "**What to notice:** `sample_kind: 'sample'`, 341 support, 114 holdout and 114 test rows, and the CSV digest. Every "
                "later number is measured on these exact partitions."
            ),
        },
        {
            "md": (
                "## 5. Validate the tables → input manifest, then encode · [Evaluation practice]\n\n"
                "`validate_inputs` is the package's public validation stage: it applies exactly the checks "
                "`prepare_classification_table` applies — unique column names, the target present, missing-target rows dropped "
                "with the count recorded, at least `MIN_TRAIN_ROWS` support rows (`MIN_EVAL_ROWS` for a holdout), at most "
                "`MAX_TRAIN_ROWS` rows and `MAX_FEATURES` columns, at least `MIN_CLASSES` classes — and returns an **input "
                "manifest** naming the schema, the observed structure (categorical columns, missing values, the class list and "
                "class counts, the majority-class share) and the verdict. It is written to `outputs/{stem}_input_manifest.json`. "
                "To show what rejection looks like, the stage also validates a probe with too few rows and records the package's "
                "own error message as a finding. The ceilings and the decision rule are printed before any model runs.\n\n"
                "The holdout and test partitions are then aligned to the support schema **and to the support class set** "
                "(`align_to_schema` refuses a class the model never saw — an in-context classifier can only predict classes present "
                "in its context), the categorical encoder is fitted on the support split only and applied everywhere (unseen values "
                "counted), and the trivial baseline is computed with `majority_class_baseline`: always predict the most frequent "
                "support class."
            ),
            "code": "run_stage('validate')",
        },
        {
            "md": (
                "**What to notice:** the ceilings; the manifest entry for the support partition (30 numeric features, no categorical "
                "columns, two classes, a majority share near 0.63); the finding `too-few-rows-probe` rejected with the package's "
                "message; `dropped_missing_target_rows` all zero for the sample; and the majority-class baseline on the holdout — "
                "accuracy 0.6228, ROC-AUC `nan` (a constant score ranks nothing).\n\n"
                "**Checkpoint:** a BYOD CSV has 7 rows with no label. Where does that number appear, and why must it?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "In the input manifest, as `dropped_missing_target_rows_before_split` on the support entry (and in the printed "
                "`dropped_missing_target_rows`). Those rows cannot be used — there is nothing to learn or score them against — but "
                "dropping them silently would change the data you think you evaluated: the partitions would be smaller and could "
                "be biased if the unlabelled rows are not random. Recording the count makes the change visible and auditable; the "
                "earlier version of this notebook dropped them before validation and reported 0.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Condition TabICLv2, then fine-tune it (GPU) · [Concept]\n\n"
                "`fit` does not train anything; it registers the support rows as the model's context (the checkpoint pinned in "
                "Section 3 is deserialised by `tabicl` here). `classification_metrics` scores the holdout and the independent "
                "test: **accuracy**, **balanced accuracy** and **weighted F1** are discrete correctness under the implicit "
                "`argmax` rule; **log loss** scores the class probabilities and penalises confident mistakes; **ROC-AUC** "
                "(binary, or one-vs-rest) scores ranking quality independent of any threshold and is `NaN` when undefined.\n\n"
                "**Fine-tuning (on by default; needs CUDA).** With `RUN_FINE_TUNING` ticked on a GPU runtime, "
                "`create_finetuned_classifier` runs the upstream gradient fine-tuning for at most `FINE_TUNE_EPOCHS` (default 3) "
                "epochs and 300 seconds, with early stopping on the holdout and `EVAL_METRIC`; the best checkpoint is reloaded into "
                "an ordinary classifier with the same inference ensemble for a fair comparison, and the candidate replaces the "
                "pretrained model **only** if it beats it on the holdout (`compare_metric`) and the holdout has at least "
                "`MIN_SELECTION_HOLDOUT_ROWS` = 50 rows. The independent test is evidence only; a worse test result is printed as a "
                "warning and never changes the selection. TabICL writes one checkpoint per epoch; the stage keeps `best.ckpt` only. "
                "The fine-tuned `best.ckpt` may remain larger than the base checkpoint because upstream training state can be "
                "embedded; the format is kept for checkpoint compatibility. On a CPU runtime the stage skips fine-tuning and records "
                "why in the result.\n\n"
                "**Reproducibility boundary.** The split seed and the TabICL ensemble seed are fixed; that makes the partitioning and "
                "the estimator configuration repeatable, but not bitwise-identical floating-point results across devices, library "
                "builds or kernels.\n\n"
                "**Predict before running:** the majority baseline scores 0.62 on the holdout and LightGBM 0.9649 (Section 7). "
                "Will the pretrained model's holdout accuracy be closer to LightGBM's or higher — and will three epochs of fine-tuning "
                "change it by more than one row (0.0088)?"
            ),
            "code": (
                "RUN_FINE_TUNING = True  # @param {{type:\"boolean\"}}\n"
                "FINE_TUNE_EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "EVAL_METRIC = 'accuracy'  # @param [\"accuracy\", \"log_loss\", \"roc_auc\"]\n"
                "run_stage('condition', run_fine_tuning=RUN_FINE_TUNING, fine_tune_epochs=FINE_TUNE_EPOCHS, eval_metric=EVAL_METRIC)"
            ),
        },
        {
            "md": (
                "**What to notice:** the pretrained holdout and test metrics; on a GPU the fine-tuning log, the candidate's holdout "
                "metrics, and `recommended_for_export` with its `selection_basis` (`holdout:accuracy` when the holdout has ≥ 50 rows); "
                "on a CPU `fine_tune.skipped` with the reason. No recorded run of this revision exists yet, so read your own numbers "
                "against the baselines of Section 7.\n\n"
                "**Checkpoint:** why is the candidate selected on the holdout and never on the independent test — and what would "
                "go wrong if the test were used?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Any partition used to make a choice stops being an unbiased estimate of performance on new data: picking the "
                "better of two models on a partition favours whichever happened to fit that partition's quirks. The holdout pays "
                "that price so the test does not. If the test chose the model, the reported test score would be optimistic and "
                "there would be no clean estimate left. That is why the stage only *warns* when the test disagrees with the "
                "holdout's choice: the disagreement is evidence about selection noise, not a reason to override it.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Classical baselines, blending and the evaluation report · [Evaluation practice]\n\n"
                "LightGBM and a random forest are fitted on the exact same encoded support rows and scored on the exact same "
                "holdout and test partitions, so the comparison is fair. Their probability columns are mapped onto the TabICLv2 "
                "class order with `align_probabilities` **before** any blending, so a multiclass label permutation can never pass "
                "silently. A convex blend of TabICLv2 and LightGBM probabilities is chosen by maximising holdout accuracy over a "
                "101-point grid (ties broken toward the foundation model by the upper-median tied weight); it is evaluated in memory "
                "only, its tied weight range is printed, and the exported bundle in Section 9 is the unchanged selected model. Each "
                "model is reported with its holdout error count beside the one-row resolution ($1/N$). Latencies are medians of five "
                "warmed runs on this runtime.\n\n"
                "`evaluation_report` is the package's public evaluation stage and always produces a report. Here it carries the "
                "selected model's holdout metrics (`accuracy`, `balanced_accuracy`, `f1_weighted`, `log_loss`, `roc_auc`), the "
                "independent-test metrics, the majority-class baseline, the classical baselines, the error counts, the blend and "
                "the fine-tuning record, with the verdict `sample-sanity`: one seeded stratified split with no dispersion estimate. "
                "Without a labelled holdout the verdict would be `not-measurable`. The report is written to "
                "`outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** on 114 holdout rows, can a blend gain more than one row over TabICLv2 alone, and would "
                "you trust that gain?"
            ),
            "code": "run_stage('report')",
        },
        {
            "md": (
                "**What to notice:** `one_row_resolution_pct: 0.88`; LightGBM at 0.9649 (4 errors) and the random forest at 0.9561 "
                "(5 errors) on the holdout — and 0.9737 and 0.9825 on the independent test; TabICLv2's line from your run; the "
                "blend's tied weight range (when it is wide, the chosen weight is not informative); "
                "and the report's `verdict: sample-sanity`.\n\n"
                "**Checkpoint:** the blend scores one row better than TabICLv2 on the holdout but no better on the test. What do you "
                "report?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "That the blend shows no evidence of improvement. The blend weight was chosen to maximise holdout accuracy, so its "
                "holdout score is selection-biased by construction, and one row is the smallest change the holdout can show. The "
                "test, untouched by the choice, shows no gain — the stage prints exactly this *mixed evidence* warning. Report the "
                "test numbers, the error counts and the one-row resolution, and do not export the blend.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. New-data inference with class probabilities · [Engineering]\n\n"
                "By default the `predict` stage scores the first eight holdout rows — already scored in Section 7, so this "
                "demonstrates the output contract, not a new evaluation — and writes their unlabelled copy, with their original "
                "`row_id`, to `outputs/{stem}_new_rows.csv`, the input the companion artifact-inference notebook accepts. Tick "
                "`RUN_NEW_DATA_INFERENCE` and set `NEW_DATA_PATH` (or, in Colab, leave it empty to upload) to score your own "
                "unlabelled CSV: `read_inference_csv` and `validate_inputs(..., target_column=None, feature_columns=...)` require a "
                "unique header, no pre-existing `prediction`/`probability_*` columns and every feature present; extra columns are "
                "kept in the output and not passed to the model; the support-fitted encoder is applied (unseen values counted). The "
                "output adds `prediction` (the `argmax` label) and one `probability_<class>` column per class **in the fitted class "
                "order**; the probabilities are uncalibrated and no threshold is shipped, so a cost-sensitive decision must "
                "threshold the relevant probability on the caller's own labelled data. `outputs/{stem}_result.json` records the "
                "predictions, metrics, baselines, the fine-tuning record, the evaluation report, the input manifest, the sample "
                "digest, the notebook's source, the model identity, revision and licence, and the runtime identity as a provenance "
                "record."
            ),
            "code": (
                "RUN_NEW_DATA_INFERENCE = False  # @param {{type:\"boolean\"}}\n"
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "new_data_file = ''\n"
                "if RUN_NEW_DATA_INFERENCE:\n"
                "    new_data_file = NEW_DATA_PATH or next(iter(upload_files('one unlabelled CSV').values()))\n"
                "run_stage('predict', new_data_path=new_data_file)"
            ),
        },
        {
            "md": (
                "**What to notice:** eight rows with their `row_id`, the `prediction` and one `probability_<class>` column per class, "
                "summing to 1 per row."
            ),
        },
        {
            "md": (
                "## 9. Export a DIMER-style serving bundle, then prove a fresh reload · [Engineering]\n\n"
                "The ZIP carries the minimum serving contract the DIMER pipeline expects: `checkpoints/best.ckpt`, "
                "`training_context.parquet`, and `artifact.json`. The training context is *required*: TabICL is still an in-context "
                "learner at serve time, so whoever loads the bundle must hand the model the same rows you evaluated with, and the "
                "context inherits the source data's confidentiality, licensing, retention and disclosure obligations. "
                "`artifact.json` records the feature and target columns, the class labels, the base-model identity and digest, the "
                "selected mode and its basis, the metrics, the inference settings including the fitted categorical encoders, "
                "per-file sizes and SHA-256 digests, the payload allowlist and the producer runtime. The `export` stage prints the "
                "ZIP's SHA-256 — the **trusted digest** to give the companion notebook (`EXPECTED_ZIP_SHA256`) — and, on the default "
                "path, whether the support table equals the companion's pinned sample bundle (`matches_pinned_sample_context`).\n\n"
                "The `reload` stage then runs in a **fresh process**: it extracts the ZIP with `safe_extract_zip` (member by member, "
                "after path, symlink and expanded-size checks — the same function the companion applies), verifies the allowlist, "
                "sizes and digests with `verify_artifact_bundle`, rebuilds the classifier from the bundle alone (checkpoint + "
                "context + recorded settings), and checks that its labels equal and its probabilities agree with the exporting "
                "process's on eight holdout rows (`rtol=1e-5`, `atol=1e-7`).\n\n"
                "**Predict before running:** if Section 6 selected the fine-tuned candidate, which checkpoint is in the bundle, and "
                "what does the companion then check about it?"
            ),
            "code": "run_stage('export')\nrun_stage('reload')",
        },
        {
            "md": (
                "**What to notice:** the ZIP path, its `zip_sha256`, the `mode`, the checkpoint size, "
                "`matches_pinned_sample_context: True` on the default path; then `labels_identical: True` and the `PASS` line.\n\n"
                "**Checkpoint:** the bundle's checkpoint is code-capable (a PyTorch/Lightning checkpoint). What protects the person "
                "who loads it?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "For a `pretrained` bundle, the companion requires the checkpoint's SHA-256 to equal the pinned base checkpoint's, so "
                "it is byte for byte the file the model's publisher released and Section 3 verified. For a `fine-tuned` bundle no "
                "such reference exists: the only protection is that the ZIP's digest matches a trusted digest received from the "
                "producer through a separate channel (`EXPECTED_ZIP_SHA256`). Internal digests inside `artifact.json` protect "
                "against corruption, not against someone who rewrites the manifest too. So: load fine-tuned bundles only from "
                "producers you trust, with their printed digest.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 10. Optional activity: a three-class table with 36 holdout rows · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** on `Sample: Wine` (178 rows, three classes, a 60/20/20 "
                "split leaves 36 holdout rows), what is the one-row resolution, and will one-vs-rest ROC-AUC be more or less "
                "informative than accuracy? **Change:** tick `RUN_ACTIVITY` (keep `ACTIVITY_DATA_SOURCE = 'Sample: Wine'`). **Run** "
                "this cell. **Observe** the activity record beside the canonical one. **Explain** what 36 rows can and cannot "
                "show. The activity runs the pretrained model only, writes to `outputs/activity/`, and stops if any canonical output "
                "changed."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_DATA_SOURCE = 'Sample: Wine'  # @param [\"Sample: Wine\", \"Sample: Palmer Penguins\"]\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', data_source=ACTIVITY_DATA_SOURCE)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** `one_row_resolution` 0.0278 against the canonical 0.0088; the majority baseline "
                "near 0.39; on this split LightGBM and a random forest each miss one holdout row (0.9722) — compare TabICLv2's line. "
                "`canonical_outputs_unchanged: True`.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "With 36 rows one mistake moves accuracy by 2.8 points, so models that differ by one row are indistinguishable. "
                "ROC-AUC uses the full probability ranking, so it can separate models whose `argmax` labels agree, but on a table "
                "this easy it saturates near 1 too. The honest reading is that every reasonable model solves Wine and that 36 rows "
                "cannot rank them.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle, or a Linux x86_64 Jupyter kernel. |\n"
        "| `No CUDA GPU detected`; later `Fine-tuning skipped: no CUDA device` | a CPU runtime | Everything else runs. For the documented runtime choose *Runtime → Change runtime type → T4 GPU*. |\n"
        "| `Not enough free disk` | the isolated environment needs about 8 GB | Start a fresh runtime; an environment built from the same lock is reused. |\n"
        "| `Carried file integrity failure` | a carried file was edited in the notebook | Open a fresh copy from the repository. |\n"
        "| `uv … mismatch`, `URLError`, or `CalledProcessError` from `uv` | network or a transient PyPI error | Re-run the Section 2 install cell. Never remove a pin or a hash. |\n"
        "| `The run directory … has no carried files, or the isolated environment is gone` | Section 1 run with `NEW_RUN_DIRECTORY` ticked | Run Sections 1–3 again, or *Run all*. |\n"
        "| `RuntimeError: Stage '…' failed (exit 1): …` | the stage's own error follows the colon | Find it below; fix the cause and re-run from that cell. |\n"
        "| A Hub download error in Section 3, or `… sha256 … != manifest` | a transient failure or a corrupted download | Re-run Section 3; delete the partial file under `weights/` if the digest fails. Never edit the manifest. |\n"
        "| `CUDA out of memory` in Section 6 | a large BYOD table | Untick `RUN_FINE_TUNING`, or restart on a fresh T4. |\n"
        "| `Fine-tuning did not produce best.ckpt` | the time limit ended before the first checkpoint | Re-run Section 6, or lower `FINE_TUNE_EPOCHS`; report it if it repeats. |\n"
        "| `the target column … is not in the header` | your label column has another name | Set `TARGET_COLUMN`. |\n"
        "| `column … is numeric except for N value(s)` | stray text in a numeric column | Fix the values, or list the column in `TEXT_COLUMNS`. |\n"
        "| `need at least 50 labelled rows` / `Every class needs >= 2 rows` | a table below the floors | Supply more rows or merge rare classes. |\n"
        "| `target classes unseen in training` | a holdout/test class missing from the support rows | Re-split so every class is in the support partition. |\n"
        "| `No path was given …` outside Colab, or `The upload was cancelled or empty` | no file supplied | Set the path field(s), or run the cell again and choose files. |\n"
        "| `Palmer Penguins sample archive digest mismatch` | the pinned archive could not be fetched intact | Retry; never change the pinned digest. |\n\n"
        "## Interpretation and limits\n\n"
        "`prediction` is the `argmax` of uncalibrated class probabilities over the fitted class set; the package ships no "
        "threshold, and any deployment cut-off must be chosen on the caller's own labelled, domain-representative data. The "
        "evaluation report's `sample-sanity` verdict names what it is: one seeded stratified split of a public sample with no "
        "dispersion estimate — tutorial evidence that must not be generalised. On the Breast Cancer sample LightGBM and a "
        "random forest already reach 0.96–0.98 on these partitions, and one row is 0.0088 of accuracy, so differences of a "
        "row or two between models rank nothing. No recorded run of this notebook revision exists yet; read the numbers your "
        "run printed. The fine-tuning selection and the blend weight are chosen on the holdout and are therefore "
        "selection-biased; only the independent test is unbiased. Rows that are not independent, classes absent from the "
        "support rows (refused, not predicted), unseen categories and support sets near the ceilings all change results in "
        "ways these metrics do not measure.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire "
        "and digest-verify the pinned checkpoint, validate the demonstrated tables with class coverage, condition on "
        "classification support rows, fine-tune with holdout-based selection on a GPU, compute sample metrics against trivial "
        "and classical baselines with strict label alignment, write the input manifest and the evaluation report, score rows "
        "with class probabilities under an explicit `argmax` rule, export a DIMER-style serving bundle and rebuild an "
        "equivalent classifier from that bundle alone in a fresh process — without the repository being reachable. It does "
        "**not** establish benchmark superiority, domain generalisation, fairness, robustness, probability calibration, "
        "production safety, or deployment fitness.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three to five sentences, using the numbers your run printed:\n\n"
        "1. **Result:** the selected model's holdout and test accuracy, log loss and error counts beside the majority "
        "baseline, LightGBM and the random forest.\n"
        "2. **Adaptation:** whether fine-tuning ran, what it changed on the holdout, and which model was selected and why.\n"
        "3. **Reading:** whether any difference exceeds one or two rows, and what the blend's tied weight range tells you.\n"
        "4. **Reuse and limits:** what the Section 9 reload proved, and the one limitation you would fix first.\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "On the 114-row holdout and 114-row independent test of the Breast Cancer table, the selected TabICLv2 model scored "
        "the accuracy and log loss printed in Sections 6 and 7, against 0.62 for the majority class and 0.96–0.98 for LightGBM "
        "and a random forest on the same partitions. Fine-tuning (three epochs on the T4) was kept only if it beat the "
        "pretrained model on the holdout; either way the gap between the foundation model and the tree ensembles is a few "
        "rows at most, within the one-row resolution of 0.0088 and the split noise, so the run shows that TabICLv2 matches "
        "strong classical models here without hyperparameter search, not that it beats them. If the blend's tied weight "
        "range is wide, the holdout cannot choose between them either. The bundle rebuilt an identical classifier in a fresh process. "
        "Before deployment I would evaluate on repeated splits of a larger, domain-representative table and calibrate the "
        "probabilities.\n\n"
        "</details>\n\n"
        "**Next experiments:** switch `DATA_SOURCE` to `Sample: Palmer Penguins` (categorical features: watch the encoder and "
        "unseen-value counts); untick `RUN_FINE_TUNING` and compare the selection record; upload your own pre-split partitions; "
        "feed `outputs/{stem}_artifact.zip`, its printed digest and `outputs/{stem}_new_rows.csv` to the companion "
        "artifact-inference notebook in a separate session.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        f"- Sample dataset card: https://github.com/kurtvalcorza/{REPO}/blob/main/examples/sample-data/DATASET_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/soda-inria/tabicl\n"
        "- TabICLv2 paper: https://arxiv.org/abs/2602.11139\n"
        "- TabICL paper: https://arxiv.org/abs/2502.05564"
    ),
}
