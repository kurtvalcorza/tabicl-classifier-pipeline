"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.5 §4 standalone, §25.13 isolated environment) — E2E.

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, checkpoint
staging) from repository files; this template holds the learner-facing prose, the guided layer and the learner
cells. Every learner cell calls ``run_stage(...)``: the carried ``tools/tutorial_stages.py`` runs one stage per process
in an isolated, hash-locked environment, so nothing is installed into the notebook kernel. Since NOTEBOOK_SPEC 3.0 the
notebook closes with the external-artifact section (Section 10, §19.1) that replaced the separate artifact-inference
companion notebook.
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
    "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab with a **T4 GPU** is the documented runtime (the pre-registered fine-tuning grid needs CUDA, and only a Tesla T4 run scores the fresh test); Kaggle or a Linux Jupyter kernel also work, and a CPU-only runtime runs everything except the grid, which it skips with a printed note (that run is exploratory). The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`tabicl` 2.1.1 with its fine-tuning extras, `torch` 2.14.0 with its CUDA libraries, `numpy` 2.5.2, `pandas` 3.0.5, `scikit-learn` 1.9.0, `lightgbm` 4.7.0). About 0.2 GB of disk is needed for the checkpoint and about 8 GB for the isolated environment."
)

TEMPLATE = {
    **ENVIRONMENT,
    "stem": "tabiclv2_classifier",
    "notebook_name": "tabiclv2_classifier_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages.py",
    "notebook_spec": "3.0",
    "notebook_kind": "TUTORIAL",
    "e2e_runner_extras": True,
    "use_notice": (
        "> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,\n"
        "> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party\n"
        "> model weights and datasets under their own licenses (see the model card) and uses compute\n"
        "> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are\n"
        "> responsible for having the right to use it. Results here are tutorial evidence, not\n"
        "> benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum\n"
        "> extent permitted by law, the maintainers and the DIMER platform accept no liability for\n"
        "> its use."
    ),
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime (a T4 GPU is the documented runtime) builds an isolated Python environment from the carried hash-locked requirements without touching the notebook kernel's own packages, then runs each stage below in its own process: it stages and digest-verifies the pinned TabICLv2 checkpoint, fetches the pinned MAGIC Gamma Telescope split archive (3,000 support, 1,000 validation and a 1,500-row fresh test) by immutable commit and verifies its SHA-256, validates and encodes the tables, evaluates the pretrained TabICLv2 classifier **adapted by in-context conditioning on the support rows**, runs a **pre-registered three-config fine-tuning grid** that takes real optimizer steps and keeps the pretrained model unless a config beats it on validation log loss by more than 0.005, compares LightGBM, a random forest and a majority-class baseline, scores the fresh test **only under the pre-registered settings** (a different setting writes an exploratory record and reads nothing), scores new rows, exports a DIMER-style serving bundle and reloads it in a fresh process, and finally, in the closing external-artifact section, downloads the pinned sample artifact that a recorded earlier run of this notebook exported (a GitHub release asset, accepted only at its pinned SHA-256 and never this run's export), validates it, rebuilds serving state from its files alone and scores eight MAGIC rows. On a runtime without CUDA the grid is skipped with a printed note and the run is exploratory. Set `DIMER_MASK_FRESH_TEST=1` to skip the fresh test. No repository clone, DIMER worker or service, credential, upload dialog, configuration edit or runtime restart is required (NOTEBOOK_SPEC 3.0 §5). Recorded runs of this notebook are listed in `docs/release-verification.md`; this revision's own run is recorded there once it exists."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` and `DATA_SOURCE` to `Upload CSV` or `Upload pre-split train/val/test` in Section 4 and run Sections 4–9 again. Give the files by path (`BYOD_CSV_PATH`, or `BYOD_TRAIN_PATH` / `BYOD_VAL_PATH` / optional `BYOD_TEST_PATH` — these work in Colab, Kaggle and Jupyter), or leave the paths empty in Colab to use upload dialogs. Name your label column in `TARGET_COLUMN`. A missing label column is refused naming it; rows with a missing label are dropped **and counted in the input manifest**; a column that is numeric except for a few stray strings is refused naming the column and the values (list it in `TEXT_COLUMNS` if it really is categorical). `RUN_NEW_DATA_INFERENCE` in Section 8 scores your own unlabelled rows; in Section 10, `USE_OWN_ROWS` scores your own rows with the external artifact, and `ARTIFACT_SOURCE` can load your own artifact ZIP instead of the pinned sample (both off by default). Files stay inside this runtime. BYOD is optional, never part of the default path, and always exploratory: a BYOD run never reads the fresh test."
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
        "This notebook runs both: conditioning, and a pre-registered fine-tuning grid on a GPU — and it shows honestly that "
        "**fine-tuning a strong in-context model often does not help**.\n\n"
        "The upstream project supplies the model and the checkpoint; the carried package adds the pinned snapshot scheme, "
        "the table preparation, class-coverage and encoding rules, the metric set, the `validate_inputs` / "
        "`majority_class_baseline` / `evaluation_report` helpers, and the serving-bundle safety checks. `predict` applies an "
        "implicit `argmax` over class probabilities that are raw ensemble outputs, **not calibrated probabilities**; the "
        "package ships no acceptance threshold. The default sample is the UCI MAGIC Gamma Telescope table [3]: harder than the "
        "earlier Breast Cancer default, with a 1,500-row fresh test that resolves differences of a few tenths of a percent. Its "
        "metrics are tutorial evidence, not a benchmark or production claim."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** the difference between in-context conditioning (registering support rows) and gradient fine-tuning, "
        "read a fine-tuning curve, the optimizer-step count and a parameter delta, and say why a pretrained model that "
        "tabicl restores is a finding, not a failure (Section 6).\n"
        "2. **Diagnose** an invalid table from a validation refusal, and say what a dropped-row count in the input manifest "
        "means (Sections 4, 5).\n"
        "3. **Compare** TabICLv2 with a majority-class baseline, LightGBM and a random forest on the same partitions, using "
        "error counts and the one-row resolution (Section 7).\n"
        "4. **Interpret** accuracy, balanced accuracy, weighted F1, log loss and ROC-AUC, and why a blend weight chosen on "
        "validation is selection-biased (Section 7); read a pre-registered fresh-test comparison with its interval, and say "
        "why changing a setting makes a run exploratory (Section 7b).\n"
        "5. **Apply** the selected model to new rows and read class-ordered, uncalibrated probabilities under an explicit "
        "`argmax` rule (Section 8).\n"
        "6. **Verify** that the exported bundle rebuilds an equivalent classifier in a fresh process (Section 9).\n"
        "7. **Validate and serve** an artifact made outside this run: check its pinned digest before opening it, read its "
        "provenance and format, rebuild serving state from its files, and say how that differs from the fresh reload (Section 10).\n"
        "8. **Predict**, run and **explain** how a three-class table with 36 holdout rows changes the one-row resolution and "
        "ROC-AUC, in an optional activity (Section 11).\n"
        "9. **Write** an evidence-based conclusion that names the baselines, the selection basis and the limits of one public "
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
        "- **Model file:** one checkpoint, `tabicl-classifier-v2-20260212.ckpt` (110 MB, released under BSD-3-Clause). It is a PyTorch/Lightning pickle checkpoint, so Section 3 reads the digest-verified file once with `torch.load(weights_only=True)` and converts it to `tabicl-classifier-v2-20260212.safetensors`, whose SHA-256 is pinned; the model is then built from that file's config and loaded from it with `load_state_dict(strict=True)`. Every stage serves the base model from the safetensors file, never from the pickle.",
        "- **Data:** the default sample is the UCI MAGIC Gamma Telescope table (19,020 simulated events, 10 numeric features, two classes, CC BY 4.0) as the repository's pinned split archive: 3,000 support, 1,000 validation and a 1,500-row fresh test, plus a 13,520-row reserve the notebook never reads; it is fetched by immutable commit and its SHA-256 is verified (`examples/sample-data/DATASET_CARD.md`). `Sample: Breast Cancer` (the earlier default, scikit-learn's bundled table) and `Sample: Wine` need no download; `Sample: Palmer Penguins` fetches the repository's pinned CC0 archive. BYOD (one CSV with a label column, or pre-split `train`/`val`/`test` CSVs) is off by default so the sample path runs top to bottom without interaction. The tutorial's own floors are `MIN_TRAIN_ROWS` = 50 support rows and `MIN_EVAL_ROWS` = 2 evaluation rows, with no class ceiling; the production DIMER validator described in the model card is stricter. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
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
                "environment, which takes the longest (PyTorch and its CUDA libraries are a few GB); Section 6 then runs the "
                "pre-registered fine-tuning grid on the GPU (the whole notebook took about 7.5 minutes on a hosted T4 in the 2026-10-20 run of record). You can also run one cell at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files: the verified checkpoint, the partitions and fitted "
                "encoders, the fine-tuned checkpoint if one was selected, and JSON records.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–11) are the machine-learning workflow. *Infrastructure cells* "
                "(Sections 1–3: the runtime check, the carried code, the isolated environment and the checkpoint staging) are "
                "collapsed and titled **Infrastructure**. You may run them without studying their implementation: they exist for "
                "reproducibility and provenance, not as prerequisite machine-learning knowledge.\n\n"
                "**Form controls.** `DATA_SOURCE`, `USE_BYOD`, `TARGET_COLUMN`, the `BYOD_*_PATH` fields and `TEXT_COLUMNS` "
                "(Section 4); `RUN_FINE_TUNING`, `GRID_EPOCHS_CAP`, `GRID_LEARNING_RATE` and `RANDOM_STATE` (Section 6); `RUN_NEW_DATA_INFERENCE` and "
                "`NEW_DATA_PATH` (Section 8); `ARTIFACT_SOURCE`, `ARTIFACT_ZIP_PATH`, `EXPECTED_ZIP_SHA256`, `USE_OWN_ROWS` and "
                "`EXTERNAL_ROWS_PATH` (Section 10); `RUN_ACTIVITY` and `ACTIVITY_DATA_SOURCE` (Section 11). Leave them at their defaults "
                "for the first run: changing any of them makes the run exploratory (validation only; the fresh test is not read). "
                "The environment variable `DIMER_MASK_FRESH_TEST=1` skips the fresh test; stages run with "
                "`DIMER_ISOLATED_ENV` set to the isolated environment's path.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 6 and 7 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint. The stages switch on PyTorch's deterministic algorithms and fixed seeds; Section 6 prints a grid hash for "
                "information: on a T4 it can differ between two runs of the same notebook revision, while the selection and the "
                "fresh-test values match exactly."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Validate** | the pinned MAGIC split | `validate_inputs`, support-fitted categorical encoders | an input manifest; 3,000 support and 1,000 validation rows (the 1,500-row fresh test is not loaded yet) |\n"
                "| **Condition** | the support rows | TabICLv2 reading them as context (`n_estimators=8`) | class probabilities for any query row |\n"
                "| **Fine-tune (GPU)** | support rows; validation for early stopping | a pre-registered three-config grid, 6 optimizer steps per epoch | a selection on validation log loss; pretrained kept unless beaten by > 0.005 |\n"
                "| **Fresh test** | 1,500 never-scored rows | the pre-registered reading rule, score-once and fail-closed | Δ log loss (selected − pretrained) with a paired bootstrap interval |\n"
                "| **Evaluate** | validation rows | `classification_metrics`, majority baseline, LightGBM, random forest, an in-memory blend | metrics, error counts, an evaluation report |\n"
                "| **Package** | selected checkpoint + support rows + settings | ZIP bundle, `safe_extract_zip`, `verify_artifact_bundle` | a bundle that rebuilds an equivalent classifier |\n"
                "| **Serve elsewhere** | a pinned artifact from an earlier recorded run + eight MAGIC rows | digest check, `safe_extract_zip`, `validate_artifact_runtime`, `verify_artifact_bundle` | predictions under `outputs/external_artifact/` |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux x86_64, GPU and disk; a run directory | the machine |\n"
                "| 2. Carry the code, build the environment | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage and verify the model | [Engineering] | the 110 MB checkpoint downloaded at a fixed revision, digest-checked, converted once to pinned safetensors | identity, both digests |\n"
                "| 4. Load the sample or your own data | [Concept] | the pinned MAGIC split (or your files) | partition sizes, digests, the mask |\n"
                "| 5. Validate and encode | [Evaluation practice] | input manifest; a refusal probe; encoders; majority baseline | the manifest |\n"
                "| 6. Condition and fine-tune | [Concept] | pretrained metrics; the pre-registered grid; validation selection | curves, steps, parameter deltas, the selection |\n"
                "| 7. Baselines, blend and report | [Evaluation practice] | LightGBM, random forest, blend; the evaluation report | the validation comparison |\n"
                "| 7b. Fresh test | [Evaluation practice] | score-once, pre-registered reading rule | the principal result and its `test_look` |\n"
                "| 8. New-data inference | [Engineering] | eight rows (or your file) scored | the output contract |\n"
                "| 9. Export and fresh reload | [Engineering] | the bundle exported and rebuilt in a new process | the reload check and the digest |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n"
                "| 10. Using an artifact made elsewhere | [Engineering] | the pinned sample artifact verified, rebuilt and used to score MAGIC rows | provenance, digests, predictions |\n"
                "| 11. Optional activity | [Concept] | the three-class Wine table (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common failures and what to do | when something fails |\n"
                "| References | | sources and repository documents | |\n\n"
                "**Fast path.** Run all, then read Sections 6, 7b, 9 and 10 and the conclusion."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **In-context learning / conditioning** | Predicting from labelled rows shown to the model at prediction time; `fit` only registers them. |\n"
                "| **Support rows / context** | The labelled rows the model reads; here the 3,000-row support partition. |\n"
                "| **`n_estimators`** | How many passes with different feature/row permutations are averaged into one prediction. |\n"
                "| **Fine-tuning** | Updating the checkpoint's weights by gradient descent on the support rows. |\n"
                "| **Early stopping** | Stopping fine-tuning when the validation metric stops improving; tabicl then restores the best state. |\n"
                "| **Validation (holdout)** | The 1,000 rows that early stopping and the grid selection look at. |\n"
                "| **Fresh test** | 1,500 rows no choice has seen, scored only under the pre-registered settings. |\n"
                "| **Exposed development test** | A test whose numbers were already seen in earlier runs (the Breast Cancer 114 rows). |\n"
                "| **Pre-registration** | Writing down settings, selection and reading rule before running, so no result can steer them. |\n"
                "| **Exploratory run** | Any run whose settings differ from the pre-registration: compared on validation only. |\n"
                "| **Optimizer step / parameter delta** | One AdamW update; the L2 norm of how far the weights moved from the pretrained ones. |\n"
                "| **Paired bootstrap interval** | Resampling the same rows for both models to bound their difference. |\n"
                "| **Selection bias** | A score looks better than it will on new data because a choice was made by looking at it. |\n"
                "| **One-row resolution** | 1 / (rows in the partition): the smallest possible change in accuracy. 1,000 rows → 0.001. |\n"
                "| **Majority-class baseline** | Always predict the most frequent support class. |\n"
                "| **LightGBM / random forest** | Strong classical tree ensembles trained on the same support rows. |\n"
                "| **Blend** | A weighted average of TabICLv2's and LightGBM's probabilities, weight chosen on the holdout. |\n"
                "| **Accuracy / balanced accuracy / weighted F1** | Discrete correctness under the `argmax` rule; balanced accuracy averages per-class recall. |\n"
                "| **Log loss** | Average negative log probability of the true class; lower is better; punishes confident mistakes. |\n"
                "| **ROC-AUC (one-vs-rest)** | Ranking quality, threshold-free; for several classes, averaged over one-class-vs-rest problems. |\n"
                "| **Uncalibrated probability** | A score not guaranteed to match observed frequencies. |\n"
                "| **Categorical encoder** | Ordinal codes fitted on the support rows only; unseen values get an 'unknown' code and are counted. |\n"
                "| **Serving bundle** | A ZIP (format `tabicl-dimer-classifier-v2`) of `artifact.json`, `checkpoints/best.safetensors` and `training_context.parquet`. |\n"
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
                "From here on, every code cell runs one stage of the carried runner with `run_stage`. The default, `Sample: MAGIC "
                "Gamma Telescope`, is the UCI MAGIC table [3] (19,020 simulated telescope events, 10 numeric features, two classes: "
                "`g` gamma, `h` hadron, CC BY 4.0). The repository commits it as a pinned split archive "
                "(`examples/sample-data/magic-gamma.zip`, fetched by immutable commit and digest-verified): **3,000 support rows, "
                "1,000 validation rows, and a 1,500-row fresh test** that was carved first and has never been fitted, predicted or "
                "scored by anyone. The 13,520 rows left over are an untouched reserve. This stage reads only the support and "
                "validation CSVs; the fresh test is read later by one stage only, under the pre-registered settings (Section 7b).\n\n"
                "The other samples are exploratory: `Sample: Breast Cancer` is the **earlier default**, kept as a labelled "
                "earlier-run reference (its 114-row test was scored in recorded runs, so here it is an *exposed development test*); "
                "`Sample: Wine` is a small three-class table; `Sample: Palmer Penguins` has mixed categorical/numeric features. With "
                "`USE_BYOD = True`: `Upload CSV` takes one CSV and carves a stratified 80/20 validation split; `Upload pre-split "
                "train/val/test` takes your own partitions (`test` optional). Give files by path, or leave the path fields empty in "
                "Colab to choose them in upload dialogs. Any non-default source makes the run **exploratory**: it never scores the "
                "fresh test.\n\n"
                "The stage refuses, naming the file and the rule, duplicate header names, a `TARGET_COLUMN` absent from the header, "
                "and a column that is numeric except for a few stray strings (list it in `TEXT_COLUMNS` if it really is "
                "categorical). A refusal is one message, not a traceback, and the stage first removes this notebook's earlier "
                "exports from `outputs/`, so a refused input never leaves an older result looking current. Rows whose label is "
                "missing are **dropped and counted** in Section 5's input manifest.\n\n"
                "**BYOD privacy boundary.** Uploaded CSV bytes are read inside the current notebook runtime and are not sent by this "
                "notebook to an external inference or training service; the network requests on the default path are the pinned "
                "checkpoint download of Section 3 and the pinned MAGIC archive."
            ),
            "code": (
                "DATA_SOURCE = 'Sample: MAGIC Gamma Telescope'  # @param [\"Sample: MAGIC Gamma Telescope\", \"Sample: Breast Cancer\", \"Sample: Wine\", \"Sample: Palmer Penguins\", \"Upload CSV\", \"Upload pre-split train/val/test\"]\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "TARGET_COLUMN = 'target'  # @param {{type:\"string\"}}\n"
                "BYOD_CSV_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_TRAIN_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_VAL_PATH = ''  # @param {{type:\"string\"}}\n"
                "BYOD_TEST_PATH = ''  # @param {{type:\"string\"}}\n"
                "TEXT_COLUMNS = []  # @param {{type:\"raw\"}}\n\n"
                "def refuse(name, message):\n"
                "    # DAT20: one message, no traceback; the earlier run is invalidated so nothing stale reads as current.\n"
                "    (ROOT / 'state' / 'data.json').unlink(missing_ok=True)\n"
                "    for stale in OUTPUTS.glob('{stem}_*'):\n"
                "        if stale.is_file():\n"
                "            stale.unlink()\n"
                "    raise Refused(name, message)\n\n"
                "def upload_files(what):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        refuse('(no file)', f'No path was given for {{what}}, and the upload dialog exists only in Google Colab: set the BYOD path field(s), then run this cell again.')\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        refuse('(no file)', f'The upload was cancelled or empty: no file was received for {{what}}. Run this cell again, or set the path field(s).')\n"
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
                "            refuse(', '.join(sorted(saved)), f'Upload exactly one CSV; got {{len(saved)}} files. Run this cell again.')\n"
                "        paths = {{'csv': next(iter(saved.values()))}}\n"
                "elif USE_BYOD and DATA_SOURCE == 'Upload pre-split train/val/test':\n"
                "    if BYOD_TRAIN_PATH or BYOD_VAL_PATH:\n"
                "        paths = {{'train': BYOD_TRAIN_PATH, 'val': BYOD_VAL_PATH, 'test': BYOD_TEST_PATH}}\n"
                "    else:\n"
                "        saved = upload_files('train.csv and val.csv (test.csv optional)')\n"
                "        if not {{'train.csv', 'val.csv'}} <= set(saved):\n"
                "            refuse(', '.join(sorted(saved)), 'Upload train.csv and val.csv (test.csv optional). Run this cell again.')\n"
                "        paths = {{'train': saved['train.csv'], 'val': saved['val.csv'], 'test': saved.get('test.csv', '')}}\n"
                "run_stage('data', data_source=DATA_SOURCE, use_byod=USE_BYOD, target_column=TARGET_COLUMN, paths=paths, text_columns=TEXT_COLUMNS)"
            ),
        },
        {
            "md": (
                "**What to notice:** the `fresh_test_mask` line (printed before anything is fetched); `sample_kind: 'sample'`, "
                "3,000 support and 1,000 validation (`holdout`) rows, `test_rows: 0` because the fresh test is not loaded here, and "
                "the archive and row-id digests. Every later number is measured on these exact partitions."
            ),
        },
        {
            "md": (
                "## 5. Validate the tables → input manifest, then encode · [Evaluation practice]\n\n"
                "`validate_inputs` is the package's public validation stage: it applies exactly the checks "
                "`prepare_classification_table` applies — unique column names, the target present, missing-target rows dropped "
                "with the count recorded, at least `MIN_TRAIN_ROWS` support rows (`MIN_EVAL_ROWS` for a validation partition), at "
                "most `MAX_TRAIN_ROWS` rows and `MAX_FEATURES` columns, at least `MIN_CLASSES` classes — and returns an **input "
                "manifest** naming the schema, the observed structure and the verdict. It is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the stage also validates a probe with too "
                "few rows and records the package's own error message as a finding. The ceilings and the decision rule are printed "
                "before any model runs.\n\n"
                "The validation partition is then aligned to the support schema **and to the support class set** (`align_to_schema` "
                "refuses a class the model never saw), the categorical encoder is fitted on the support split only and applied "
                "everywhere (unseen values counted), and the trivial baseline is computed with `majority_class_baseline`."
            ),
            "code": "run_stage('validate')",
        },
        {
            "md": (
                "**What to notice:** the ceilings; the manifest entry for the support partition (10 numeric features, no "
                "categorical columns, two classes, a majority share near 0.65); the finding `too-few-rows-probe` rejected with the "
                "package's message; `dropped_missing_target_rows` all zero; and the majority-class baseline on validation.\n\n"
                "**Checkpoint:** a BYOD CSV has 7 rows with no label. Where does that number appear, and why must it?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "In the input manifest, as `dropped_missing_target_rows_before_split` on the support entry (and in the printed "
                "`dropped_missing_target_rows`). Those rows cannot be used, but dropping them silently would change the data you "
                "think you evaluated. Recording the count makes the change visible and auditable.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Condition TabICLv2, then run the pre-registered fine-tuning grid (GPU) · [Concept]\n\n"
                "`fit` does not train anything; it registers the 3,000 support rows as the model's context. "
                "`classification_metrics` scores validation: **accuracy**, **balanced accuracy** and **weighted F1** are "
                "discrete correctness under the implicit `argmax` rule; **log loss** scores the class probabilities and is the "
                "**primary metric** here, because it registers changes that accuracy cannot; **ROC-AUC** scores ranking quality. "
                "Probabilities are renormalised row-wise (TabICL's reduced-precision ensemble average misses 1 by rounding only).\n\n"
                "**Fine-tuning a strong in-context model often does not help.** TabICLv2 already reads the support rows at "
                "prediction time; gradient fine-tuning can only add something the context does not already give it, and with a few "
                "thousand rows **gradient fine-tuning** easily overfits. This section measures that honestly instead of assuming a gain. Three configs "
                "were **pre-registered** before anything was run (`docs/magic-finetune-preregistration.md`): **C1** "
                "production-like (learning rate 1e-5, up to 30 epochs, patience 8, the `dimer-pipeline.json` defaults), **C2** "
                "larger steps (5e-5) and **C3** smaller steps (1e-6), each up to 10 epochs with patience 3. All of them take "
                "**real optimizer steps**: `max_data_size=500` splits the 3,000 support rows into 6 chunks, so each epoch is 6 "
                "AdamW steps, and a constant learning rate with no warmup means no step runs at learning rate 0. (The earlier "
                "recipe of this notebook fitted the whole support set in one chunk, so an epoch was one step, and its first step "
                "ran at learning rate 0.)\n\n"
                "**How tabicl keeps weights.** It measures the validation log loss of the pretrained weights first (the "
                "*baseline*), then after every epoch. An epoch's weights are kept only if they beat the best value so far by "
                "`min_delta` (1e-4); after `patience` epochs without improvement it stops and **restores the best state**, which is "
                "the pretrained weights if no epoch improved, and writes it as `best.ckpt` (a pickle, which the stage converts at once "
                "to `best.safetensors`; see below). The stage prints each config's "
                "validation curve, its optimizer steps, and a **parameter delta** (the L2 norm of the weight change): 0 means "
                "tabicl gave back the pretrained weights. It also reads out the **last epoch's** weights on validation, as a "
                "disclosed exploratory readout, so you can see what the gradient steps did.\n\n"
                "TabICL writes its checkpoints itself with `torch.save` (`best.ckpt` plus one per epoch). Right after each config "
                "trains, the stage reads `best.ckpt` and the last epoch's checkpoint once with `torch.load(weights_only=True)`, keeps "
                "only their `config` and `state_dict` (the optimizer state is dropped) as `best.safetensors`, and deletes every "
                "`.ckpt`; every later step, the export included, reads the safetensors file only. It keeps `best.safetensors` of "
                "the selected config only.\n\n"
                "**Selection (pre-registered, validation only).** The pretrained model is kept unless a config beats its "
                "validation log loss by more than **0.005**; the fresh test plays no part.\n\n"
                "**Reproducibility boundary.** The stages set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic`, no `cudnn.benchmark` "
                "and `torch.use_deterministic_algorithms(True)` with fixed seeds; fp16 autocast runs with float32 weights. An "
                "operation without a deterministic CUDA kernel stops the stage and names itself. Attention runs on PyTorch's **MATH** "
                "kernel with tabicl's FlashAttention-3 path off, in every stage that runs the model (pre-registration amendment 2: "
                "two masked T4 runs that let PyTorch choose its memory-efficient kernel diverged at the first backward step), and "
                "any warning about a non-deterministic kernel is printed as `DETERMINISM WARNING`. The printed **grid hash** is "
                "informational: on a T4 the fp16 grid has given at least two different hashes for the same notebook revision "
                "(issue #28; hosted runs at commits `d4172f5` and `edbe94d`, same notebook and stage runner), while the selection "
                "and every fresh-test value matched exactly. So the grid hash is printed and reported, never compared to decide "
                "a pass.\n\n"
                "**Form fields.** Leave them at their defaults for the pre-registered run. Changing any of them (a smaller "
                "`GRID_EPOCHS_CAP` to save time, a `GRID_LEARNING_RATE` override, another `RANDOM_STATE`), unticking "
                "`RUN_FINE_TUNING`, BYOD, another data source or a non-T4 runtime makes the run **exploratory**: validation only, "
                "and the fresh test is never read. On a CPU runtime the grid is skipped with a printed note.\n\n"
                "**Predict before running:** will any config beat the pretrained model on validation log loss by more than 0.005? "
                "What will the parameter delta of `best.safetensors` be if none does?"
            ),
            "code": (
                "RUN_FINE_TUNING = True  # @param {{type:\"boolean\"}}\n"
                "GRID_EPOCHS_CAP = 0  # @param {{type:\"integer\"}}\n"
                "GRID_LEARNING_RATE = 0.0  # @param {{type:\"number\"}}\n"
                "RANDOM_STATE = 42  # @param {{type:\"integer\"}}\n"
                "run_stage('condition', run_fine_tuning=RUN_FINE_TUNING, grid_epochs_cap=GRID_EPOCHS_CAP, grid_learning_rate=GRID_LEARNING_RATE, random_state=RANDOM_STATE)"
            ),
        },
        {
            "md": (
                "**What to notice:** `preregistered` and `settings_differing` at the top; the pretrained validation metrics; for "
                "each config its `val_log_loss_per_epoch` against `baseline_val_log_loss`, the optimizer steps it attempted, took and "
                "skipped (`optimizer_steps_attempted` / `optimizer_steps_taken` / `amp_skipped_steps`; fp16 on the GPU can skip a step "
                "whose gradients overflow), `weights_changed` and "
                "the two parameter deltas; `recommended_for_export` with its `selection_basis`; the `exploratory_readout`; and the "
                "grid hash. On a CPU runtime: `fine_tune.skipped` with the reason.\n\n"
                "For orientation only — the **CPU development run** of this exact grid (2026-10-19, CPU, validation only, fresh "
                "test masked, grid hash `11cb0310…`; recorded in `docs/release-verification.md`, not a run of this notebook on a "
                "T4): the pretrained model scored validation log loss 0.2634 and accuracy 0.903; against tabicl's baseline of 0.2701 "
                "(measured with 2 ensemble members), C1 ran 8 epochs at 0.2712–0.2762, C2 ran 3 epochs at 0.2864 / 0.3340 / 0.3027, "
                "and C3 ran 3 epochs at 0.2702 / 0.2701 / 0.2700 — within ±0.0001 of the baseline. **No epoch beat the baseline by "
                "more than tabicl's `min_delta` (1e-4)**, so tabicl kept the pretrained weights in every config, and no config "
                "beat the pretrained model by the pre-registered 0.005 tolerance.\n\n"
                "The **hosted run of record** (2026-10-20, Tesla T4, commit `9ea0888`, grid hash `4b1bf536…`; recorded in "
                "`docs/release-verification.md`) differed from the CPU run in one config: on the T4, C3 (learning rate 1e-6) ran "
                "all 10 epochs (60 optimizer steps taken, 0 skipped), its weights changed (`weights_changed: True`, parameter delta "
                "0.059), and its validation log loss ended at 0.2628 against the pretrained 0.2634 — an improvement of about 0.0006, "
                "far below the 0.005 tolerance. No config beat the pretrained model by more than 0.005, so the pretrained model was "
                "kept. A T4 run of this notebook prints a grid hash that need not equal the record's, even at the same revision "
                "(issue #28); its selection and fresh-test results should match the record. Read your own run's numbers.\n\n"
                "**Checkpoint:** suppose a config reports `weights_changed: False` while its last-epoch parameter delta is not 0. "
                "What happened? And what would it mean if a config instead reported `weights_changed: True`?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "`weights_changed: False` with a non-zero last-epoch delta: the gradient steps did move the weights, but no epoch's "
                "weights beat the validation baseline by more than `min_delta` — some were worse, some (like C3 in the CPU "
                "development run) were within a ten-thousandth of it — so tabicl's early stopping restored the pretrained state and "
                "wrote it as `best.ckpt` (converted to `best.safetensors`). That is the procedure working as designed: selection on held-out data keeps an adaptation "
                "only when it measurably helps. A restored config can never be selected as \"fine-tuned\".\n\n"
                "`weights_changed: True`: some epoch did beat the baseline by more than `min_delta`, and tabicl kept those weights "
                "(C3 on the hosted T4 run of record is this case). "
                "That is still not enough to be selected: the config must also beat the pretrained model's validation log loss by "
                "more than 0.005. Either way the validation partition chose, and the fresh test, untouched by that choice, says how "
                "the chosen model does on new data. A strong in-context model often leaves little for fine-tuning to add on a few "
                "thousand rows; a null result is a finding, reported as one.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Classical baselines, blending and the evaluation report · [Evaluation practice]\n\n"
                "LightGBM and a random forest are fitted on the exact same encoded support rows and scored on the exact same "
                "validation partition, so the comparison is fair. Their probability columns are mapped onto the TabICLv2 class "
                "order with `align_probabilities` **before** any blending. A convex blend of TabICLv2 and LightGBM probabilities is "
                "chosen by maximising validation accuracy over a 101-point grid (ties broken toward the foundation model); it is "
                "evaluated in memory only, its tied weight range is printed, and the exported bundle in Section 9 is the unchanged "
                "selected model. Each model is reported with its validation error count beside the one-row resolution ($1/N$). "
                "Latencies are medians of five warmed runs on this runtime.\n\n"
                "`evaluation_report` carries the selected model's validation metrics, the majority-class baseline, the classical "
                "baselines, the error counts, the blend, the whole fine-tuning grid and its hash, with the verdict "
                "`sample-sanity`. It is written to `outputs/{stem}_evaluation_report.json`. The MAGIC fresh test is **not** read "
                "here (Section 7b). For the exploratory samples their test partition is printed with its label (for Breast Cancer: "
                "*exposed development test*).\n\n"
                "**Predict before running:** on 1,000 validation rows one row is 0.001 of accuracy. Can a blend chosen on those "
                "rows be trusted to beat TabICLv2 by a few rows?"
            ),
            "code": "run_stage('report')",
        },
        {
            "md": (
                "**What to notice:** `one_row_resolution_pct: 0.1`; the TabICLv2, LightGBM and random-forest lines on validation "
                "with their error counts; the blend's tied weight range; and the report's `verdict: sample-sanity`.\n\n"
                "**Checkpoint:** the blend scores a few rows better than TabICLv2 on validation. What do you report?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "That the blend's validation gain is selection-biased: its weight was chosen to maximise exactly that number. Only "
                "data untouched by the choice can say whether it helps, and the fresh test is reserved for the pre-registered "
                "model comparison, not for tuning a blend. Report the blend as an in-memory demonstration and export the selected "
                "model.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7b. The fresh test: scored only under the pre-registered settings · [Evaluation practice]\n\n"
                "The 1,500 fresh-test rows answer one pre-registered question: **how does the selected model compare with the "
                "pretrained model on rows that no choice has seen?** The primary endpoint is the fresh-test **log loss difference "
                "Δ = selected − pretrained**, with a paired bootstrap 95% interval over rows (seed 20261019, 10,000 resamples). "
                "Verdict: *improved* if the interval's upper bound is below 0 and Δ ≤ −0.005; *worse* if its lower bound is above 0; "
                "otherwise *no detectable difference*. **If the pretrained model was selected, the verdict is \"same model "
                "(Δ = 0)\"**: no difference was measured, because none can exist. Accuracy (Wilson interval) and ROC-AUC are "
                "secondary, and a sensitivity readout repeats Δ without the 4 fresh-test rows that have an exact feature duplicate "
                "in the support or validation rows (3 and 1). Power, from the dataset research's planning estimates: about ±0.017 "
                "on a single accuracy, about ±0.011 on a paired accuracy difference, about ±0.017 on a paired log-loss difference.\n\n"
                "**Score-once, fail-closed.** Before reading the test the stage recomputes, from the recorded settings, that the "
                "run is `preregistered` (every fixed setting, BYOD off, a Tesla T4), that the grid is complete and its hash matches, "
                "that the selected checkpoint's digest is the one Section 6 recorded, and that the archive digest matches. If any "
                "setting differs it writes `outputs/fresh-test/{stem}_fresh_test_exploratory_<n>.json` (`test_look: not scored "
                "(exploratory)`, with the differing settings) and reads nothing. The evaluation of record already exists — the "
                "first pre-registered scoring, on a hosted T4 on 2026-10-20 (commit `9ea0888`), labelled **\"pre-registered "
                "evaluation (this runtime)\"** — and this notebook carries it, so every pre-registered scoring now, the first in a "
                "new runtime included, writes `_reproduction_<n>`, is labelled **\"reproduction\"** and prints `matches_record` "
                "with the fields that `differs` from the record. The selection, the device and every fresh-test value must match "
                "exactly. The grid hash is printed as `grid_hash` with the record's value, this run's value and whether they are "
                "`equal`, labelled **informational**: it never decides `matches_record`, whatever the stage runner (issue #28), "
                "and neither does the printed `stage_runner_sha256`. `docs/release-verification.md` lists the record and its "
                "reproductions. Set the environment variable "
                "`DIMER_MASK_FRESH_TEST=1` (for example `%env DIMER_MASK_FRESH_TEST=1`) to skip the fresh test entirely; it is "
                "re-read on every stage call. The mask stops the fresh test from being read, not from being downloaded: the pinned "
                "archive holds all three CSVs, so the data stage still fetches and caches it (digest-checked), but `test.csv` is "
                "never extracted, parsed or scored under the mask.\n\n"
                "**Predict before running:** if Section 6 kept the pretrained model, what will Δ and its interval be?"
            ),
            "code": "run_stage('fresh_test')",
        },
        {
            "md": (
                "**What to notice:** the mask line; `test_look`; `selected`; the primary endpoint with its interval and verdict; "
                "the secondary accuracy and ROC-AUC; and the sensitivity readout. On an exploratory run: the list of "
                "`settings_differing` and no test numbers at all.\n\n"
                "**Checkpoint:** why does the notebook refuse to score the fresh test when you change only `GRID_EPOCHS_CAP`?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Because a test you can re-score under any setting stops being a test: try enough settings, keep the best number, "
                "and the result is chosen on the test. Fixing every setting in advance and scoring only that combination keeps the "
                "fresh-test number an honest estimate. Exploration belongs on validation.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. New-data inference with class probabilities · [Engineering]\n\n"
                "By default the `predict` stage scores the first eight validation rows — already scored in Section 7, so this "
                "demonstrates the output contract, not a new evaluation — and writes their unlabelled copy, with their original "
                "`row_id`, to `outputs/{stem}_new_rows.csv`. Tick `RUN_NEW_DATA_INFERENCE` and set `NEW_DATA_PATH` (or, in Colab, "
                "leave it empty to upload) to score your own unlabelled CSV: `read_inference_csv` and `validate_inputs(..., "
                "target_column=None, feature_columns=...)` require a unique header, no pre-existing `prediction`/`probability_*` "
                "columns and every feature present; extra columns are kept in the output and not passed to the model. The output "
                "adds `prediction` (the `argmax` label) and one `probability_<class>` column per class **in the fitted class "
                "order**; the probabilities are uncalibrated and no threshold is shipped. `outputs/{stem}_result.json` records the "
                "predictions, metrics, baselines, the fine-tuning record, the evaluation report, the input manifest, the sample "
                "digest, the notebook's source, the model identity, revision and licence, and the runtime identity."
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
                "summing to 1 per row (they are renormalised after the deviation is checked)."
            ),
        },
        {
            "md": (
                "## 9. Export a DIMER-style serving bundle, then prove a fresh reload · [Engineering]\n\n"
                "The ZIP carries the minimum serving contract the DIMER pipeline expects (bundle format `tabicl-dimer-classifier-v2`): "
                "`checkpoints/best.safetensors` (the converted fine-tune, or for `pretrained` the converted base whose SHA-256 is pinned), "
                "`training_context.parquet`, and `artifact.json`. The training context is *required*: TabICL is still an in-context "
                "learner at serve time, so whoever loads the bundle must hand the model the same rows you evaluated with, and the "
                "context inherits the source data's licensing obligations (MAGIC: CC BY 4.0, attribution required). `artifact.json` "
                "records the feature and target columns, the class labels, the base-model identity and digest, the selected mode "
                "and its basis, the metrics, the inference settings, per-file sizes and SHA-256 digests, the payload allowlist and "
                "the producer runtime. The `export` stage prints the ZIP's SHA-256 — the **trusted digest** to hand over with the ZIP, "
                "which whoever loads it checks before opening it (Section 10 shows that check on an artifact made elsewhere).\n\n"
                "The `reload` stage then runs in a **fresh process**: it extracts the ZIP with `safe_extract_zip`, verifies the "
                "allowlist, sizes and digests with `verify_artifact_bundle`, rebuilds the classifier from the bundle alone, and "
                "checks that its labels equal and its probabilities agree with the exporting process's on eight validation rows "
                "(`rtol=1e-5`, `atol=1e-7`).\n\n"
                "**Predict before running:** if Section 6 kept the pretrained model, which checkpoint is in the bundle, and what "
                "does Section 10 check about such a checkpoint before it loads one?"
            ),
            "code": "run_stage('export')\nrun_stage('reload')",
        },
        {
            "md": (
                "**What to notice:** the ZIP path, its `zip_sha256`, the `mode` and the checkpoint size; then "
                "`labels_identical: True` and the `PASS` line.\n\n"
                "**Checkpoint:** the bundle's checkpoint is code-capable (a PyTorch/Lightning checkpoint). What protects the person "
                "who loads it?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "For a `pretrained` bundle, the external-artifact section (Section 10) requires the checkpoint's SHA-256 to equal the pinned base checkpoint's, so "
                "it is byte for byte the file the model's publisher released and Section 3 verified. For a `fine-tuned` bundle no "
                "such reference exists: the only protection is that the ZIP's digest matches a trusted digest received from the "
                "producer through a separate channel (`EXPECTED_ZIP_SHA256`). So: load fine-tuned bundles only from producers you "
                "trust, with their printed digest.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
            "## Interpretation and limits\n\n"
            "`prediction` is the `argmax` of uncalibrated class probabilities over the fitted class set; the package ships no "
            "threshold, and any deployment cut-off must be chosen on the caller's own labelled, domain-representative data. The "
            "evaluation report's `sample-sanity` verdict names what it is: one seeded split of a public sample with no dispersion "
            "estimate. MAGIC is a simulated physics table in the TALENT benchmark, which the TabICLv2 paper evaluated on [1]; "
            "TabICLv2 is pretrained on synthetic tables only, so that is evaluation exposure, not training leakage. It is not in "
            "TabArena or OpenML-CC18. The fine-tuning selection and the blend weight are chosen on validation and are "
            "selection-biased; only the fresh test, scored under the pre-registered settings, is an unbiased estimate, and its "
            "recorded evaluation and reproductions are listed in `docs/release-verification.md`. The evaluation of record (2026-10-20, "
            "hosted T4) kept the pretrained model, so its fresh test reads \"same model (Δ = 0)\": accuracy 0.8993 (Wilson 95 % "
            "0.8831–0.9136) and ROC AUC 0.95 on 1,500 rows. A null fine-tuning result here is "
            "evidence about this model on this table, not a general rule.\n\n"
            "**Earlier-run reference (Breast Cancer, previous recipe).** In the Colab T4 run of blob `051c81e9` (2026-10-10, recorded "
            "in `docs/release-verification.md`), the earlier default — the Breast Cancer table with a 114-row holdout and a 114-row "
            "test — gave the pretrained model holdout accuracy 0.9825 (112/114); the old three-epoch fine-tune took one non-zero "
            "step and was discarded by tabicl, so the candidate equalled the pretrained model; LightGBM scored 0.9649 and a random "
            "forest 0.9561 on that holdout. That test was scored in recorded runs and is an exposed development test; no setting of "
            "this notebook was chosen on it. These numbers are history, not results of this revision.\n\n"
            "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire "
            "and digest-verify the pinned checkpoint and data, validate the demonstrated tables with class coverage, condition on "
            "support rows, run a pre-registered fine-tuning grid with validation-based selection on a GPU, score the fresh test "
            "only under the pre-registered settings, compute sample metrics against trivial and classical baselines, score rows "
            "with class probabilities under an explicit `argmax` rule, export a DIMER-style serving bundle and rebuild an "
            "equivalent classifier from that bundle alone in a fresh process, and validate, rebuild and serve a pinned artifact "
            "that an earlier recorded run produced — without the repository being reachable. It does **not** establish benchmark superiority, "
            "domain generalisation, fairness, robustness, probability calibration, production safety, or deployment fitness.\n\n"
            "## Conclusion · [Evaluation practice]\n\n"
            "Write three to five sentences, using the numbers your run printed:\n\n"
            "1. **Result:** the selected model's validation accuracy and log loss beside the majority baseline, LightGBM and the "
            "random forest.\n"
            "2. **Adaptation:** what each grid config did on validation (curve, steps, parameter delta), and which model was "
            "selected and why.\n"
            "3. **Fresh test:** the primary endpoint with its interval and verdict, and its `test_look`.\n"
            "4. **Reuse and limits:** what the Section 9 reload proved, what Section 10 (next) adds with an artifact made elsewhere, and the one "
            "limitation you would fix first.\n\n"
            "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
            "A sample of the reasoning, written for the case the hosted run of record showed (2026-10-20, T4); "
            "replace it with your own numbers: on MAGIC's 1,000 validation rows the pretrained TabICLv2 model already beat LightGBM "
            "and the random forest on log loss (0.2634 against 0.3412 and 0.3234). Fine-tuning moved the weights in one config — "
            "C3 at learning rate 1e-6 lowered validation log loss by about 0.0006 — but no config beat the pretrained model by "
            "the 0.005 tolerance, so the pretrained model was kept: the fresh test then reports \"same model (Δ = 0)\" "
            "(accuracy 0.8993, Wilson 95 % 0.8831–0.9136) — nothing to compare, which is itself the finding that fine-tuning "
            "this in-context model on 3,000 rows did not measurably help.If instead a config was "
            "selected, report the fresh-test Δ with its interval and verdict, and say whether the interval excludes 0. "
            "The bundle rebuilt an identical classifier in a fresh process. Before deployment I would evaluate on a "
            "domain-representative table and calibrate the probabilities.\n\n"
            "</details>\n\n"
            ),
        },
        {
            "md": (
                "## 10. Using an artifact made elsewhere · [Engineering]\n\n"
                "So far this notebook has scored data with the model it adapted in this run, and the fresh reload showed that this "
                "run's export survives serialization. This section does something different: the reload shows that this run's "
                "export survives serialization, and this section shows that an artifact made elsewhere can be validated and served "
                "here. It downloads a pinned artifact that an earlier, recorded run of this notebook produced, verifies its SHA-256 "
                "before opening it, shows its provenance and format, rebuilds serving state from the files alone, and scores sample "
                "rows that follow the same data contract as the rest of this notebook. **This run's own export (Section 9) is never "
                "used here.**\n\n"
                "**This section needs only the setup cells (Sections 1–3).** It reads no partition, encoder, model or file that "
                "Sections 4–9 created, so if you already hold a compatible artifact you can run Sections 1–3 and then this section "
                "alone. Its outputs go to `outputs/external_artifact/`, apart from the main path's files.\n\n"
                "**The pinned sample artifact.** By default the `external_artifact` stage downloads `tabiclv2_classifier_artifact.zip` "
                "from this repository's GitHub release `sample-artifact-magic-v1` (no account or token). It is the deterministic "
                "export of a recorded hosted T4 run of this notebook's default path (the MAGIC sample) at commit `383ca04`; the "
                "release notes name the run record, and the asset is never replaced (a changed artifact gets a new tag). The stage "
                "prints the URL, the pinned SHA-256, the producer and the format (this pinned asset is a `tabicl-dimer-classifier-v1` "
                "bundle; its re-issue in the v2 layout is staged as `sample-artifact-magic-v2`) **before** it "
                "downloads anything, then refuses the file unless its SHA-256 equals the pinned value. Only then does "
                "`safe_extract_zip` copy the members one by one (no absolute paths, `..`, symlinks or oversize archives); "
                "`validate_artifact_runtime` checks the format and the `tabicl`/`torch` versions; the manifest must name the pinned "
                "base checkpoint; `verify_artifact_bundle` checks the member allowlist, sizes and digests; and the manifest's "
                "features, target and class labels must equal the MAGIC default path's. Any failed check stops the run, and no "
                "prediction file is written.\n\n"
                "**Trust boundary.** A v2 artifact's `checkpoints/best.safetensors` holds tensors and a JSON config only and is "
                "loaded without `torch.load`. A v1 artifact (such as this pinned asset) carries a pickle `checkpoints/best.ckpt`; that "
                "layout is deprecated and is read only with `torch.load(weights_only=True)`, with a `DeprecationWarning`. The "
                "pinned SHA-256 binds this download to the file the maintainer published; for a `pretrained` artifact the checkpoint "
                "must also be, byte for byte, the pinned base (v2: its safetensors conversion; v1: the source checkpoint Section 3 "
                "verified). Nothing is deserialised before these checks pass. For a "
                "`fine-tuned` artifact of your own, the only binding is a digest you received from its producer through a separate "
                "channel (`EXPECTED_ZIP_SHA256`).\n\n"
                "**Optional: your own artifact (off by default).** Set `ARTIFACT_SOURCE` to `My own artifact ZIP` and give "
                "`ARTIFACT_ZIP_PATH` (or leave it empty in Colab to upload) and the producer's `EXPECTED_ZIP_SHA256`. Without a "
                "digest the checks establish internal consistency only, and the stage says so."
            ),
            "code": (
                "ARTIFACT_SOURCE = 'Pinned sample artifact'  # @param [\"Pinned sample artifact\", \"My own artifact ZIP\"]\n"
                "ARTIFACT_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "EXPECTED_ZIP_SHA256 = ''  # @param {{type:\"string\"}}\n\n\n"
                "def external_upload(what):\n"
                "    \"\"\"Colab upload for the optional branches of this section; elsewhere, set the path field.\"\"\"\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError as exc:\n"
                "        raise Refused('(no file)', f'No path was given for {{what}}, and the upload dialog exists only in Google Colab: set the path field, then run this cell again.') from exc\n"
                "    uploaded = files.upload()\n"
                "    if len(uploaded) != 1:\n"
                "        raise Refused(', '.join(sorted(uploaded)) or '(no file)', f'Expected exactly one file for {{what}}; received {{len(uploaded)}}. Run this cell again.')\n"
                "    ((upload_name, upload_payload),) = uploaded.items()\n"
                "    upload_path = ROOT / 'inputs' / Path(upload_name).name\n"
                "    upload_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    upload_path.write_bytes(upload_payload)\n"
                "    return str(upload_path)\n\n\n"
                "if ARTIFACT_SOURCE == 'Pinned sample artifact':\n"
                "    run_stage('external_artifact', source='pinned')\n"
                "else:\n"
                "    run_stage('external_artifact', source='own', zip_path=ARTIFACT_ZIP_PATH or external_upload('your artifact ZIP'), expected_zip_sha256=EXPECTED_ZIP_SHA256)"
            ),
        },
        {
            "md": (
                "**What to notice:** the provenance line is printed before the download; then `trusted_digest: verified (pinned "
                "SAMPLE_ARTIFACT_SHA256)`, the `mode`, the `checkpoint_binding` and the `data_contract`; the producer's runtime and "
                "notebook source; and the artifact's 10 MAGIC features, its `target` column and the class labels `g` and `h`."
            ),
        },
        {
            "md": (
                "**Reconstruct serving state locally.** The `external_reconstruct` stage rebuilds the classifier from the "
                "artifact's files alone, in a new process: its checkpoint (with `allow_auto_download=False`), its "
                "`training_context.parquet` — the fitted support rows TabICL conditions on — and the categorical encoders recorded in "
                "`artifact.json`. Nothing is refitted from the rows you are about to score."
            ),
            "code": "run_stage('external_reconstruct')",
        },
        {
            "md": (
                "**Score sample rows, or your own.** By default the `external_predict` stage takes the first eight rows of the pinned "
                "MAGIC validation file, removes their label, and scores them: they follow the artifact's data contract and are not "
                "among its support rows (the 1,500-row fresh test is never read). The rows pass the same `read_inference_csv` / "
                "`validate_inputs` checks as Section 8, a probe with a missing feature must be refused, and non-numeric values in a "
                "numeric feature are refused naming the column. The predictions, the input manifest and the artifact's provenance "
                "go to `outputs/external_artifact/{stem}_external_predictions.csv` and "
                "`outputs/external_artifact/{stem}_external_result.json`.\n\n"
                "**Optional: your own rows (BYOD, off by default).** Tick `USE_OWN_ROWS` and set `EXTERNAL_ROWS_PATH` (or leave it "
                "empty in Colab to upload) to an unlabelled CSV with the artifact's feature columns. With your own artifact, your own "
                "rows are required: the MAGIC rows follow only the pinned artifact's contract."
            ),
            "code": (
                "USE_OWN_ROWS = False  # @param {{type:\"boolean\"}}\n"
                "EXTERNAL_ROWS_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "external_rows = ''\n"
                "if USE_OWN_ROWS:\n"
                "    external_rows = EXTERNAL_ROWS_PATH or external_upload('one unlabelled CSV')\n"
                "run_stage('external_predict', rows_path=external_rows)"
            ),
        },
        {
            "md": (
                "**What to notice:** eight rows with `row_id`, `prediction` and `probability_g` / `probability_h` summing to 1; the "
                "two output paths under `outputs/external_artifact/`; and the artifact's `zip_sha256`, equal to the pinned digest.\n\n"
                "**Checkpoint:** if you set `ARTIFACT_SOURCE` to this run's own `outputs/{stem}_artifact.zip`, would that be "
                "external-artifact evidence?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "No. An artifact this run created and then reloaded is a fresh-boundary test, which Section 9 already performed. "
                "External evidence needs an artifact made outside this run with documented provenance, which is why the default is a "
                "pinned release asset from an earlier recorded run, checked by its SHA-256 before it is opened.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 11. Optional activity: a three-class table with 36 holdout rows · [Concept]\n\n"
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
                "**What to notice (if you ran it):** `one_row_resolution` 0.0278 against the canonical 0.001; the majority baseline "
                "near 0.39; TabICLv2's accuracy and ROC-AUC on 36 rows. `canonical_outputs_unchanged: True`.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "With 36 rows one mistake moves accuracy by 2.8 points, so models that differ by one row are indistinguishable. "
                "ROC-AUC uses the full probability ranking, so it can separate models whose `argmax` labels agree, but on a table "
                "this easy it saturates near 1 too. Compare MAGIC's 1,000 validation and 1,500 fresh-test rows: that is why the "
                "default table changed.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle, or a Linux x86_64 Jupyter kernel. |\n"
        "| `No CUDA GPU detected`; later `Fine-tuning skipped: no CUDA device` | a CPU runtime | Everything else runs, as an exploratory run. For the documented runtime choose *Runtime → Change runtime type → T4 GPU*. |\n"
        "| `Not enough free disk` | the isolated environment needs about 8 GB | Start a fresh runtime; an environment built from the same lock is reused. |\n"
        "| `Carried file integrity failure` | a carried file was edited in the notebook | Open a fresh copy from the repository. |\n"
        "| `uv … mismatch`, `URLError`, or `CalledProcessError` from `uv` | network or a transient PyPI error | Re-run the Section 2 install cell. Never remove a pin or a hash. |\n"
        "| `The run directory … has no carried files, or the isolated environment is gone` | Section 1 run with `NEW_RUN_DIRECTORY` ticked | Run Sections 1–3 again, or *Run all*. |\n"
        "| `RuntimeError: Stage '…' failed (exit 1): …` | the stage's own error follows the colon | Find it below; fix the cause and re-run from that cell. |\n"
        "| `Refused: <file name>: …` | an input was refused (one line, no traceback; your data, rows or artifact ZIP) | Fix the named file or field and run that cell again (for Section 4 data, re-run the Section 4 cell); nothing from the refused input was kept. |\n"
        "| A Hub download error in Section 3, or `… sha256 … != manifest` | a transient failure or a corrupted download | Re-run Section 3; delete the partial file under `weights/` if the digest fails. Never edit the manifest. |\n"
        "| `MAGIC sample archive digest mismatch` / `Palmer Penguins sample archive digest mismatch` | the pinned archive could not be fetched intact | Retry; never change the pinned digest. |\n"
        "| `Trusted digest mismatch` in Section 10 | the downloaded or supplied ZIP is not the pinned (or expected) artifact | Re-run the Section 10 cell to download again; never change the pinned digest. With your own ZIP, ask its producer for the digest. |\n"
        "| `does not follow this notebook's default data contract (SART10)` | the artifact was built from another dataset or an earlier default | Use an artifact exported from this notebook's current default path. |\n"
        "| `external_artifact.json is missing` | a later Section 10 cell was run before the first one | Run the Section 10 cells in order (setup Sections 1–3 first). |\n"
        "| `… does not have a deterministic implementation …` in Section 6 | an operation without a deterministic CUDA kernel | Report it with the log; do not switch the determinism flags off for a run of record. |\n"
        "| `CUDA out of memory` in Section 6 | a large BYOD table | Untick `RUN_FINE_TUNING` (the run becomes exploratory), or restart on a fresh T4. |\n"
        "| `fresh test not scored: pre-scoring check(s) failed` | Section 6 was not run after Section 4, or a file changed | Re-run from Section 4; a failed check is not a look. |\n"
        "| `test_look: not scored (exploratory)` | a form field, the data source, BYOD or the device differs from the pre-registration | Expected: exploratory runs are compared on validation only. |\n"
        "| `the target column … is not in the header` | your label column has another name | Set `TARGET_COLUMN`. |\n"
        "| `column … is numeric except for N value(s)` | stray text in a numeric column | Fix the values, or list the column in `TEXT_COLUMNS`. |\n"
        "| `need at least 50 labelled rows` / `Every class needs >= 2 rows` | a table below the floors | Supply more rows or merge rare classes. |\n"
        "| `target classes unseen in training` | a validation/test class missing from the support rows | Re-split so every class is in the support partition. |\n"
        "| `No path was given …` outside Colab, or `The upload was cancelled or empty` | no file supplied | Set the path field(s), or run the cell again and choose files. |\n\n"
        "**Next experiments (all exploratory, validation only):** set `GRID_LEARNING_RATE` or `GRID_EPOCHS_CAP` and compare the "
        "curves; switch `DATA_SOURCE` to `Sample: Palmer Penguins` (categorical features); upload your own pre-split "
        "partitions; in a separate session, set `ARTIFACT_SOURCE` to your own ZIP in Section 10 and give it this run's "
        "`outputs/{stem}_artifact.zip` with its printed digest (an exercise of the optional branch, not external-artifact "
        "evidence).\n\n"
        "## References\n\n"
        "[1] J. Qu et al., \"TabICLv2,\" arXiv:2602.11139, 2026. [Online]. Available: https://arxiv.org/abs/2602.11139\n\n"
        "[2] J. Qu et al., \"TabICL: A tabular foundation model for in-context learning on large data,\" arXiv:2502.05564, 2025. [Online]. Available: https://arxiv.org/abs/2502.05564\n\n"
        "[3] R. Bock, \"MAGIC Gamma Telescope,\" UCI Machine Learning Repository, 2004, doi: 10.24432/C52C8B. [Online]. Available: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope\n\n"
        "[4] soda-inria, \"tabicl,\" software, version 2.1.1. [Online]. Available: https://github.com/soda-inria/tabicl\n\n"
        "[5] jingang, \"TabICL model checkpoints,\" Hugging Face. [Online]. Available: https://huggingface.co/{MODEL_ID}\n\n"
        f"Repository documents:\n\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Fine-tuning pre-registration: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/magic-finetune-preregistration.md\n"
        f"- Release verification: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/release-verification.md\n\n"
        f"[6] K. Valcorza, \"tabicl-classifier-pipeline: README, model card, weight provenance, sample dataset card, and the MAGIC pre-registration,\" GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/{REPO} (see `README.md`, `MODEL_CARD.md`, `docs/WEIGHTS.md`, `examples/sample-data/DATASET_CARD.md`, `docs/magic-finetune-preregistration.md`)"
    ),
}
