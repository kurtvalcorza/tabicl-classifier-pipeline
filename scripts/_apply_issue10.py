#!/usr/bin/env python3
import json
from pathlib import Path

MAIN = Path('tutorials/tabiclv2_classifier_colab.ipynb')
INFERENCE = Path('tutorials/tabiclv2_classifier_artifact_inference_colab.ipynb')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, notebook):
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def source_text(cell):
    source = cell.get('source', '')
    return ''.join(source) if isinstance(source, list) else str(source)


def set_source(cell, text):
    cell['source'] = text


def find_cell(notebook, marker):
    matches = [cell for cell in notebook['cells'] if marker in source_text(cell)]
    if len(matches) != 1:
        raise SystemExit(f'Expected one cell containing {marker!r}; found {len(matches)}')
    return matches[0]


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected one occurrence, found {count}')
    return text.replace(old, new, 1)


encoder_old = '''def apply_encoder(frame, encoders):
    out = frame.copy()
    for col, cats in encoders.items():
        lookup, unknown = {c:i for i,c in enumerate(cats)}, len(cats)
        out[col] = [unknown if pd.isna(v) else lookup.get(str(v), unknown) for v in out[col]]
    return out
'''
encoder_new = '''def apply_encoder(frame, encoders, label="data"):
    out = frame.copy()
    for col, cats in encoders.items():
        lookup, unknown = {c:i for i,c in enumerate(cats)}, len(cats)
        encoded, unseen = [], 0
        for value in out[col]:
            if pd.isna(value):
                encoded.append(unknown)
                continue
            key = str(value)
            if key not in lookup:
                unseen += 1
            encoded.append(lookup.get(key, unknown))
        out[col] = encoded
        if unseen:
            print(f"⚠ {label}: {unseen} unseen categorical value(s) in {col!r} encoded as unknown.")
    return out
'''

main = load(MAIN)
data_cell = find_cell(main, 'def apply_encoder(frame, encoders):')
text = replace_once(source_text(data_cell), encoder_old, encoder_new, 'main encoder')
for old, new in (
    ('train_encoded = apply_encoder(train_data, CATEGORICAL_ENCODERS)', 'train_encoded = apply_encoder(train_data, CATEGORICAL_ENCODERS, "training data")'),
    ('holdout_encoded = apply_encoder(holdout_data, CATEGORICAL_ENCODERS)', 'holdout_encoded = apply_encoder(holdout_data, CATEGORICAL_ENCODERS, "holdout")'),
    ('test_encoded = apply_encoder(test_data, CATEGORICAL_ENCODERS) if test_data is not None else None', 'test_encoded = apply_encoder(test_data, CATEGORICAL_ENCODERS, "independent test") if test_data is not None else None'),
):
    text = replace_once(text, old, new, 'main encoder call')
set_source(data_cell, text)

fine_markdown = find_cell(main, '## 4. Evaluate pretrained TabICLv2')
text = source_text(fine_markdown)
if 'non-best epoch checkpoints' not in text:
    text += '\n**Fine-tuning disk usage:** TabICL writes epoch checkpoints while tuning. Their size scales with `FINE_TUNE_EPOCHS`; after the best checkpoint is loaded and evaluated, this notebook deletes non-best epoch checkpoints and retains `best.ckpt` only.\n'
set_source(fine_markdown, text)

fine_cell = find_cell(main, 'FINE_TUNE_EPOCHS, FINE_TUNE_TIME_LIMIT, FINE_TUNE_PATIENCE')
text = source_text(fine_cell)
prune_after = '    if candidate_test_metrics: show("Fine-tuned test",candidate_test_metrics)\n'
prune_block = prune_after + '''    transient_checkpoints = [
        checkpoint for checkpoint in ft_dir.rglob("*.ckpt")
        if checkpoint.resolve() != candidate_checkpoint.resolve()
    ]
    for checkpoint in transient_checkpoints:
        checkpoint.unlink()
    if transient_checkpoints:
        print(f"✓ Pruned {len(transient_checkpoints)} non-best fine-tuning checkpoint(s); retained best.ckpt.")
'''
text = replace_once(text, prune_after, prune_block, 'checkpoint pruning')
set_source(fine_cell, text)

manifest_cell = find_cell(main, '"artifactFormat":"tabicl-dimer-classifier-v1"')
text = source_text(manifest_cell)
manifest_old = '    "tabiclVersion":TABICL_VERSION,"mode":ACTIVE_MODE,"selectionBasis":SELECTION_BASIS,\n'
manifest_new = '''    "tabiclVersion":TABICL_VERSION,"mode":ACTIVE_MODE,"selectionBasis":SELECTION_BASIS,
    "checkpointSource":CHECKPOINT_SOURCE,
    "metrics":{"selectionMetric":EVAL_METRIC,
               "pretrainedHoldout":baseline_metrics,"fineTunedHoldout":candidate_metrics,
               "pretrainedIndependentTest":baseline_test_metrics,"fineTunedIndependentTest":candidate_test_metrics},
'''
text = replace_once(text, manifest_old, manifest_new, 'artifact provenance')
set_source(manifest_cell, text)
save(MAIN, main)

inference = load(INFERENCE)
inference_cell = find_cell(inference, 'def apply_encoder(frame,encoders):')
text = source_text(inference_cell)
inference_old = '''def apply_encoder(frame,encoders):
    out=frame.copy()
    for col,cats in encoders.items():
        lookup,unknown={c:i for i,c in enumerate(cats)},len(cats)
        out[col]=[unknown if pd.isna(v) else lookup.get(str(v),unknown) for v in out[col]]
    return out
'''
inference_new = '''def apply_encoder(frame,encoders,label="data"):
    out=frame.copy()
    for col,cats in encoders.items():
        lookup,unknown={c:i for i,c in enumerate(cats)},len(cats)
        encoded=[]; unseen=0
        for value in out[col]:
            if pd.isna(value):
                encoded.append(unknown); continue
            key=str(value)
            if key not in lookup: unseen+=1
            encoded.append(lookup.get(key,unknown))
        out[col]=encoded
        if unseen: print(f"⚠ {label}: {unseen} unseen categorical value(s) in {col!r} encoded as unknown.")
    return out
'''
text = replace_once(text, inference_old, inference_new, 'inference encoder')
text = replace_once(
    text,
    'X=apply_encoder(rows[FEATURE_COLUMNS],inference.get("categoricalEncoders",{}))',
    'X=apply_encoder(rows[FEATURE_COLUMNS],inference.get("categoricalEncoders",{}),"inference CSV")',
    'inference encoder call',
)
set_source(inference_cell, text)
save(INFERENCE, inference)
