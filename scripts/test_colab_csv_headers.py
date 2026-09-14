"""CSV-header and artifact-path safety classifications.

These behaviours used to live inline in the tutorial notebooks and were tested by extracting the
functions from the notebook source; they now live in the public reference API
(`tabicl_classifier_pipeline.api`), which the standalone notebooks carry verbatim, so the tests
exercise the package functions directly.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tabicl_classifier_pipeline import (  # noqa: E402
    manifest_member_path,
    read_csv_payload,
    read_inference_csv,
)


class ColabSafetyTests(unittest.TestCase):
    def test_training_duplicate_headers_rejected(self):
        payloads = [
            b"a,a,target\n1,2,x\n",
            b'"sale,amount","sale,amount",target\n1,2,x\n',
            b"\xef\xbb\xbfa,a,target\r\n1,2,x\r\n",
            b"\n\na,a,target\n1,2,x\n",
        ]
        for payload in payloads:
            with self.assertRaisesRegex(ValueError, "duplicate column names"):
                read_csv_payload(payload, "train.csv")

    def test_valid_training_headers_keep_order(self):
        frame = read_csv_payload(b'amount,"sale,amount",target\n1,2,x\n', "train.csv")
        self.assertEqual(list(frame.columns), ["amount", "sale,amount", "target"])

    def test_inference_duplicate_headers_rejected_and_order_kept(self):
        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            read_inference_csv(b"a,a\n1,2\n", ["a"])
        frame = read_inference_csv(b'amount,"sale,amount"\n1,2\n', ["amount", "sale,amount"])
        self.assertEqual(list(frame.columns), ["amount", "sale,amount"])

    def test_manifest_member_path_is_confined_to_artifact_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = (root / "checkpoints" / "best.ckpt").resolve()
            self.assertEqual(manifest_member_path(root, "checkpoints/best.ckpt", "checkpoint"), expected)
            self.assertEqual(manifest_member_path(root, "./checkpoints/best.ckpt", "checkpoint"), expected)

            absolute_escape = str((root.parent / "escape.ckpt").resolve())
            for value in (absolute_escape, "../escape.ckpt", "checkpoints/../../escape.ckpt"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    manifest_member_path(root, value, "checkpoint")


if __name__ == "__main__":
    unittest.main()
