from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'tutorials/tabiclv2_classifier_colab.ipynb'
INFERENCE = ROOT / 'tutorials/tabiclv2_classifier_artifact_inference_colab.ipynb'

def load_code(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        s = "".join(src) if isinstance(src, list) else str(src)
        s = "\n".join(line for line in s.splitlines() if not line.lstrip().startswith("%"))
        chunks.append(s)
    return "\n".join(chunks)

def extract_functions(source: str, names: set[str]):
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    mod = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"csv": __import__("csv"), "io": __import__("io"), "pd": pd}
    exec(compile(mod, "<extracted>", "exec"), ns)
    return ns

main_ns = extract_functions(load_code(MAIN), {"raw_header", "duplicate_names", "read_csv_payload"})
inf_ns = extract_functions(load_code(INFERENCE), {"raw_header", "read_inference_csv"})

class CsvHeaderTests(unittest.TestCase):
    def test_training_duplicate_headers_rejected(self):
        payloads = [
            b"a,a,target\n1,2,x\n",
            b'"sale,amount","sale,amount",target\n1,2,x\n',
            b"\xef\xbb\xbfa,a,target\r\n1,2,x\r\n",
            b"\n\na,a,target\n1,2,x\n",
        ]
        for payload in payloads:
            with self.assertRaisesRegex(ValueError, "duplicate column names"):
                main_ns["read_csv_payload"](payload, "train.csv")

    def test_valid_training_headers_keep_order(self):
        frame = main_ns["read_csv_payload"](b'amount,"sale,amount",target\n1,2,x\n', "train.csv")
        self.assertEqual(list(frame.columns), ["amount", "sale,amount", "target"])

    def test_inference_duplicate_headers_rejected_and_order_kept(self):
        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            inf_ns["read_inference_csv"](b"a,a\n1,2\n", ["a"])
        frame = inf_ns["read_inference_csv"](b'amount,"sale,amount"\n1,2\n', ["amount", "sale,amount"])
        self.assertEqual(list(frame.columns), ["amount", "sale,amount"])

if __name__ == "__main__":
    unittest.main()
