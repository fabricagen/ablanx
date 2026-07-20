"""Numerical agreement: the JAX AbRep forward reproduces reference AbLang2 embeddings.

Loads the AbLang2 AbRep weights (207-tensor npz; export_weights.py, or the release asset once tagged) and the
committed golden panel (golden_ablang2_panel.npz: reference rescoding embeddings for 30 antibody Fvs -- 25
paired VH+VL and 5 VH-only, drawn from public PDB structures and produced by reference PyTorch AbLang2), runs
the JAX forward on each record's token ids, and asserts every Fv matches. Point ABLANG_WEIGHTS at the npz;
the test skips if the weights are absent (test_shapes.py always runs).

    ABLANG_WEIGHTS=/path/to/ablang2_weights.npz python -m pytest test_agreement.py
    ABLANG_WEIGHTS=/path/to/ablang2_weights.npz python test_agreement.py
"""
import os

import numpy as np
import jax.numpy as jnp

from ablang_jax import Ablanx, load_ablanx_params

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("ABLANG_WEIGHTS", os.path.join(HERE, "ablang2_weights.npz"))
PANEL = os.path.join(HERE, "golden_ablang2_panel.npz")
MAX_ABS_TOL = 1e-4
MIN_COS_TOL = 0.9999


def _agree(tree, ids, ref):
    ids = np.asarray(ids)[None].astype(np.int32)
    ref = np.asarray(ref).astype(np.float32)
    mask = np.ones_like(ids, np.float32)
    hid, _ = Ablanx().apply({"params": tree}, jnp.asarray(ids), jnp.asarray(mask), return_attn=False)
    jx = np.asarray(hid[0], np.float32)
    max_abs = float(np.abs(jx - ref).max())
    cos = (jx * ref).sum(-1) / (np.linalg.norm(jx, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-9)
    return max_abs, float(cos.min())


def _panel(tree):
    p = np.load(PANEL, allow_pickle=True)
    ids, emb, names = p["ids"], p["emb"], p["names"]
    return [(str(names[k]), *_agree(tree, ids[k], emb[k])) for k in range(len(names))]


def test_matches_reference_embeddings():
    import pytest
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"set ABLANG_WEIGHTS to the AbRep npz (see export_weights.py); not found at {WEIGHTS}")
    tree = load_ablanx_params(dict(np.load(WEIGHTS)))
    for name, max_abs, min_cos in _panel(tree):
        assert max_abs < MAX_ABS_TOL, f"{name}: max_abs {max_abs:.3e} >= {MAX_ABS_TOL}"
        assert min_cos > MIN_COS_TOL, f"{name}: min_cos {min_cos:.6f} <= {MIN_COS_TOL}"


if __name__ == "__main__":
    if not os.path.exists(WEIGHTS):
        raise SystemExit(f"set ABLANG_WEIGHTS to the AbRep npz (see export_weights.py); not found at {WEIGHTS}")
    tree = load_ablanx_params(dict(np.load(WEIGHTS)))
    rows = _panel(tree)
    for name, ma, mc in rows:
        print(f"{name:24s} max_abs={ma:.3e}  min_cos={mc:.6f}")
    worst = max(rows, key=lambda r: r[1])
    print(f"PANEL n={len(rows)}  worst max_abs={worst[1]:.3e} ({worst[0]})")
    assert all(ma < MAX_ABS_TOL and mc > MIN_COS_TOL for _, ma, mc in rows)
    print("AGREEMENT OK")
