"""Classification checks for the tutorial refinements tracked in issue #10.

The notebooks are generated (NOTEBOOK_SPEC 2.2 standalone carrier): their code runs as a carried stage
runner (`tutorial_stages.py`, text in the carrier cell) in an isolated environment. Source markers are
asserted on the notebook text plus the carried runner, and the encoder behaviour on the carried package
function (`apply_categorical_encoder`, which returns the unseen-value counts the notebooks print instead of
printing itself).
"""

import ast
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


def text(path):
    """Every cell's text, with the carrier cell replaced by the stage runner it carries."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        if cell.get("metadata", {}).get("dimer", {}).get("embedded_sources"):
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CARRIED_FILES":
                    source = ast.literal_eval(node.value)["tutorial_stages.py"]
        parts.append(source)
    return "\n".join(parts)


class Issue10HardeningTests(unittest.TestCase):
    def test_manifest_records_selection_metrics(self):
        main = text(MAIN)
        self.assertIn('metrics = {"selectionMetric": cond["eval_metric"]', main)
        self.assertIn('"pretrainedHoldout": cond["pretrained_holdout"]', main)
        self.assertIn('"fineTunedHoldout": cond["candidate_holdout"]', main)
        self.assertIn('"mode": mode,', main)
        self.assertIn('"selectionBasis": selection,', main)

    def test_non_best_finetune_checkpoints_are_pruned(self):
        main = text(MAIN)
        self.assertIn("for checkpoint in epoch_ckpts[:-1]:", main)
        self.assertIn("checkpoint.unlink()", main)
        self.assertIn("It keeps `best.safetensors` of", main)

    def test_finetuned_checkpoints_are_converted_to_safetensors_and_no_pickle_survives(self):
        main = text(MAIN)
        self.assertIn("best_facts = P.convert_finetuned_checkpoint(best_pickle)", main)
        self.assertIn("deletes every", main)
        self.assertIn("`torch.load(weights_only=True)`", main)

    def test_unseen_categories_are_counted_in_the_notebook(self):
        frame = pd.DataFrame({"kind": ["known", "new", None]})
        encoded, unseen = apply_categorical_encoder(frame, {"kind": ["known"]})
        self.assertEqual(encoded["kind"].tolist(), [0, 1, 1])
        self.assertEqual(unseen, {"kind": 1})
        for path in (MAIN,):
            with self.subTest(notebook=path.name):
                self.assertIn("apply_categorical_encoder(", text(path))
                self.assertIn("unseen", text(path))


if __name__ == "__main__":
    unittest.main()
