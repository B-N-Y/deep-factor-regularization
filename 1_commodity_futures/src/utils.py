"""
utils.py
--------
Run metadata and safe output handling shared by the run_* scripts.
"""
import glob
import hashlib
import json
import os
import platform
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hash_sources() -> dict:
    files = sorted(glob.glob(os.path.join(ROOT, 'src', '*.py')) + glob.glob(os.path.join(ROOT, 'run_*.py')))
    out = {}
    for f in files:
        with open(f, 'rb') as fh:
            out[os.path.relpath(f, ROOT)] = hashlib.md5(fh.read()).hexdigest()
    return out


# Taken when a run script imports this module, i.e. from the code that is actually loaded
# (files replaced on disk while a job runs do not change what the job reports).
_FINGERPRINT_AT_START = _hash_sources()


def code_fingerprint() -> dict:
    """md5 of every source file as loaded when the run started."""
    return dict(_FINGERPRINT_AT_START)


# Versions of these source files that give identical results (checked by running both
# versions on the same experiments), so that runs made with either can be combined.
# src/utils.py only holds metadata/IO helpers and is not part of any computation.
COMPATIBLE_VERSIONS = {
    'src/config.py': {'fb17b9bf9cf96697c6276f6b636b31e0', 'e7613a54f8396f67b52bc38396e66df8'},
    'src/models.py': {'8230a66da5efe12b6d415c4354b2c11f', '0411d666df05c452b1d26e2633cb46ed'},
}
IGNORED_FOR_COMPARISON = {'src/utils.py'}


def code_mismatch(fingerprints: list):
    """Return the first src/ file whose versions differ beyond COMPATIBLE_VERSIONS, else None."""
    files = set().union(*[{k for k in f if k.startswith('src/')} for f in fingerprints]) - IGNORED_FOR_COMPARISON
    for name in sorted(files):
        seen = {f.get(name) for f in fingerprints}
        if len(seen) > 1 and not seen <= COMPATIBLE_VERSIONS.get(name, set()):
            return name
    return None


def versions() -> dict:
    import numpy, pandas, scipy, sklearn, torch, optuna
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'optuna': optuna.__version__,
        'numpy': numpy.__version__,
        'pandas': pandas.__version__,
        'scipy': scipy.__version__,
        'sklearn': sklearn.__version__,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def prepare_out_dir(path: str, overwrite: bool) -> str:
    """Create a results folder; refuse to mix with an existing run unless overwrite."""
    if os.path.isdir(path) and os.listdir(path) and not overwrite:
        sys.exit(f"❌ {path} already contains results. Use a new --out-root or pass --overwrite.")
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def same_run_data(p1: dict, p2: dict) -> bool:
    """Same data split and benchmarks: labels and dates exactly, numbers up to 1e-9 (platform rounding)."""
    if p1['meta'].keys() != p2['meta'].keys() or p1['benchmarks'].keys() != p2['benchmarks'].keys():
        return False
    for k, v in p1['meta'].items():
        w = p2['meta'][k]
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            if not np.allclose(v, w, rtol=1e-9, atol=1e-10):
                return False
        elif v != w:
            return False
    return all(np.allclose(p1['benchmarks'][k], p2['benchmarks'][k], rtol=1e-9, atol=1e-10) for k in p1['benchmarks'])
