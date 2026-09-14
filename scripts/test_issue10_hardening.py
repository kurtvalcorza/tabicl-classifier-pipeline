"""Classification checks for the tutorial refinements tracked in issue #10.

The notebooks are generated (NOTEBOOK_SPEC 1.1 standalone carrier), so source markers are asserted on
the generated notebooks and the encoder behaviour on the carried package function
(`apply_categorical_encoder`, which returns the unseen-value counts the notebooks print instead of
printing itself).
"""

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tabicl_classifier_pipeline import apply_categorical_encoder  # noqa: E402

MAIN = ROOT / "tutorials" / "tabiclv2_classifier_colab.ipynb"
INFERENCE = ROOT / "tutorials" / "tabiclv2_classifier_artifact_inference_colab.ipynb"


def text(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        parts.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(parts)


class Issue10HardeningTests(unittest.TestCase):
    def test_manifest_records_selection_metrics(self):
        main = text(MAIN)
        self.assertIn("'metrics': {'selectionMetric': EVAL_METRIC", main)
        self.assertIn("'pretrainedHoldout': pretrained_metrics", main)
        self.assertIn("'fineTunedHoldout': candidate_metrics", main)
        self.assertIn("'mode': ACTIVE_MODE, 'selectionBasis': SELECTION_BASIS", main)

    def test_non_best_finetune_checkpoints_are_pruned(self):
        main = text(MAIN)
        self.assertIn("candidate_checkpoint.resolve()", main)
        self.assertIn("checkpoint.unlink()", main)
        self.assertIn("deletes the others and keeps `best.ckpt` only", main)

    def test_finetuned_checkpoint_size_is_documented_without_mutating_checkpoint_format(self):
        main = text(MAIN)
        self.assertIn("fine-tuned `best.ckpt` may remain larger", main)
        self.assertIn("checkpoint compatibility", main)

    def test_unseen_categories_are_counted_for_main_and_companion(self):
        frame = pd.DataFrame({"kind": ["known", "new", None]})
        encoded, unseen = apply_categorical_encoder(frame, {"kind": ["known"]})
        self.assertEqual(encoded["kind"].tolist(), [0, 1, 1])
        self.assertEqual(unseen, {"kind": 1})
        for path in (MAIN, INFERENCE):
            with self.subTest(notebook=path.name):
                self.assertIn("apply_categorical_encoder(", text(path))
                self.assertIn("unseen", text(path))


if __name__ == "__main__":
    unittest.main()
