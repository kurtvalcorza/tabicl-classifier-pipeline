"""Print the SHA-256 of the canonical safetensors shim that the fine-tune worker vendors byte for byte.

kurtvalcorza/tabicl-classifier-finetuner carries ``tabicl_safetensors_shim.py`` as an exact copy of
``src/tabicl_classifier_pipeline/safetensors_shim.py`` and pins its SHA-256 in
``tests/test_vendored_shim.py``. After any change to the shim here: run this script, copy the file over
the worker's copy, and set the worker's pinned digest to the printed value. CI prints it on every run
(no third-party imports), so the value to re-sync to is always in the CI log.
"""

import hashlib
from pathlib import Path

RELATIVE = "src/tabicl_classifier_pipeline/safetensors_shim.py"
SHIM = Path(__file__).resolve().parents[1] / RELATIVE

if __name__ == "__main__":
    data = SHIM.read_bytes()
    if b"\r\n" in data:
        raise SystemExit(f"{SHIM}: CRLF line endings; the vendored copy is pinned on the LF bytes")
    print(f"{hashlib.sha256(data).hexdigest()}  {RELATIVE} ({len(data)} bytes)")
