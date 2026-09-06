"""Regression checks for the tutorial refinements tracked in issue #10."""

import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials" / "tabiclv2_classifier_colab.ipynb"
INFERENCE = ROOT / "tutorials" / "tabiclv2_classifier_artifact_inference_colab.ipynb"


def sources(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        yield "".join(source) if isinstance(source, list) else str(source)


def text(path):
    return "\n".join(sources(path))


def encoder(path):
    for source in sources(path):
        if "def apply_encoder" not in source:
            continue
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "apply_encoder"
        )
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {"pd": pd}
        exec(compile(ast.fix_missing_locations(module), path.name, "exec"), namespace)
        return namespace["apply_encoder"]
    raise AssertionError(f"Could not locate apply_encoder in {path.name}")


class Issue10HardeningTests(unittest.TestCase):
    def test_manifest_records_acquisition_source_and_selection_metrics(self):
        main = text(MAIN)
        self.assertIn('"checkpointSource":CHECKPOINT_SOURCE', main)
        self.assertIn('"metrics":{', main)
        self.assertIn('"selectionMetric":EVAL_METRIC', main)
        self.assertIn('"pretrainedHoldout":baseline_metrics', main)
        self.assertIn('"fineTunedHoldout":candidate_metrics', main)

    def test_non_best_finetune_checkpoints_are_pruned(self):
        main = text(MAIN)
        self.assertIn("transient_checkpoints", main)
        self.assertIn("candidate_checkpoint.resolve()", main)
        self.assertIn("checkpoint.unlink()", main)
        self.assertIn("non-best fine-tuning checkpoint", main)

    def test_unseen_categories_emit_warning_in_main_and_companion(self):
        frame = pd.DataFrame({"kind": ["known", "new", None]})
        for path in (MAIN, INFERENCE):
            with self.subTest(notebook=path.name):
                fn = encoder(path)
                with patch("builtins.print") as printer:
                    encoded = fn(frame, {"kind": ["known"]}, "inference CSV")
                self.assertEqual(encoded["kind"].tolist(), [0, 1, 1])
                messages = " ".join(" ".join(map(str, call.args)) for call in printer.call_args_list)
                self.assertIn("1 unseen categorical value", messages)
                self.assertIn("kind", messages)


if __name__ == "__main__":
    unittest.main()
