from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/tabiclv2_classifier_colab.ipynb")
README = Path("tutorials/README.md")
SOURCE_SAMPLE_REVISION = "6306684aa53afc5b0a6b3ce7e58df44339d32a03"

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cell = next(
    c
    for c in nb["cells"]
    if c.get("cell_type") == "code"
    and 'DATA_SOURCE = "Sample: Breast Cancer"' in "".join(c.get("source", []))
)
src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

src = src.replace(
    "import math\n\nimport numpy as np",
    "import math\nimport urllib.request\nimport zipfile\n\nimport numpy as np",
    1,
)
src = src.replace(
    "from sklearn.datasets import load_breast_cancer",
    "from sklearn.datasets import load_breast_cancer, load_wine",
    1,
)
old_selector = (
    'DATA_SOURCE = "Sample: Breast Cancer"  # @param '
    '["Sample: Breast Cancer", "Upload CSV", "Upload pre-split train/val/test"]'
)
new_selector = (
    'DATA_SOURCE = "Sample: Breast Cancer (binary sanity)"  # @param '
    '["Sample: Breast Cancer (binary sanity)", "Sample: Wine (3-class multiclass)", '
    '"Sample: Palmer Penguins (multiclass + categoricals)", "Sample: Breast Cancer", '
    '"Upload CSV", "Upload pre-split train/val/test"]'
)
if old_selector not in src:
    raise SystemExit("classifier DATA_SOURCE selector anchor not found")
src = src.replace(old_selector, new_selector, 1)

start_marker = 'test_data = None\nif DATA_SOURCE == "Sample: Breast Cancer":\n'
end_marker = 'elif DATA_SOURCE == "Upload CSV":\n'
start = src.find(start_marker)
end = src.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit("classifier sample branch anchors not found")

sample_block = f'''test_data = None
if DATA_SOURCE in {{"Sample: Breast Cancer", "Sample: Breast Cancer (binary sanity)"}}:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.rename(columns={{dataset.target.name: TARGET_COLUMN}})
    train_data, remainder = train_test_split(
        frame, test_size=0.4, random_state=RANDOM_SEED, stratify=frame[TARGET_COLUMN]
    )
    holdout_data, test_data = train_test_split(
        remainder,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=remainder[TARGET_COLUMN],
    )
    print("✓ Using Breast Cancer Wisconsin sample (binary sanity).")
elif DATA_SOURCE == "Sample: Wine (3-class multiclass)":
    dataset = load_wine(as_frame=True)
    frame = dataset.frame.rename(columns={{dataset.target.name: TARGET_COLUMN}})
    train_data, remainder = train_test_split(
        frame, test_size=0.4, random_state=RANDOM_SEED, stratify=frame[TARGET_COLUMN]
    )
    holdout_data, test_data = train_test_split(
        remainder,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=remainder[TARGET_COLUMN],
    )
    print("✓ Using Wine Recognition sample (3-class multiclass).")
elif DATA_SOURCE == "Sample: Palmer Penguins (multiclass + categoricals)":
    TARGET_COLUMN = "species"
    sample_url = (
        "https://raw.githubusercontent.com/kurtvalcorza/tabicl-classifier-pipeline/"
        "{SOURCE_SAMPLE_REVISION}/examples/sample-data/palmer-penguins.zip"
    )
    with urllib.request.urlopen(sample_url, timeout=30) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        by_name = {{Path(info.filename).name.lower(): info for info in archive.infolist() if not info.is_dir()}}
        required = {{"train.csv", "val.csv", "test.csv"}}
        missing = required - set(by_name)
        if missing:
            raise RuntimeError(f"Palmer Penguins sample ZIP missing: {{sorted(missing)}}")
        train_data = read_csv_payload(archive.read(by_name["train.csv"]), "train.csv")
        holdout_data = read_csv_payload(archive.read(by_name["val.csv"]), "val.csv")
        test_data = read_csv_payload(archive.read(by_name["test.csv"]), "test.csv")
    print("✓ Using Palmer Penguins sample (multiclass + categoricals).")
'''
src = src[:start] + sample_block + src[end:]
cell["source"] = src

# Enrich only the sample description while preserving the Notebook Spec BYOD/privacy text.
markdown = next(
    c
    for c in nb["cells"]
    if c.get("cell_type") == "markdown"
    and "## 3. Load sample or BYOD data" in "".join(c.get("source", []))
)
md = "".join(markdown["source"]) if isinstance(markdown["source"], list) else markdown["source"]
old_intro = (
    "**Sample: Breast Cancer** is scikit-learn's Wisconsin diagnostic set (569 rows, 30 numeric "
    "features, malignant/benign). It is split 60 / 20 / 20, stratified, into train / holdout / "
    "independent test. It is a *sanity* dataset: the pretrained model already scores in the high "
    "0.9s on it, which makes it a poor place to see fine-tuning help, and a good place to see the "
    "plumbing work."
)
new_intro = (
    "The default **Breast Cancer Wisconsin** sample remains the binary sanity path. The selector "
    "also exposes **Wine Recognition** (3-class numeric multiclass) and the bundled **Palmer "
    "Penguins** archive (3-class mixed numeric/categorical features). All sample splits remain "
    "tutorial/sanity evidence rather than benchmark evidence; provenance and license details are "
    "recorded in `examples/sample-data/DATASET_CARD.md`."
)
if old_intro not in md:
    raise SystemExit("classifier Step 3 markdown anchor not found")
markdown["source"] = md.replace(old_intro, new_intro, 1)

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

text = README.read_text(encoding="utf-8")
if "## Sample portfolio\n" not in text:
    text = text.rstrip() + """

## Sample portfolio

The E2E notebook retains Breast Cancer Wisconsin as its default binary sanity case and adds the sample portfolio from PR #15: scikit-learn Wine Recognition for numeric multiclass classification and Palmer Penguins for mixed numeric/categorical multiclass classification. `examples/sample-data/DATASET_CARD.md` records provenance and licensing; sample results remain tutorial evidence rather than model-ranking claims.
"""
    README.write_text(text, encoding="utf-8")
