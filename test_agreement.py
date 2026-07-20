"""Numerical agreement: the JAX AbRep forward reproduces reference AbLang2 embeddings.

Loads the AbLang2 AbRep weights (207-tensor npz; get them with export_weights.py or the release asset) and
the committed golden fixture golden_ablang2_pert_vh.npz (reference rescoding embeddings for one Fv, produced
by reference PyTorch AbLang2), runs the JAX forward on the same token ids, and asserts they match.

Point ABLANG_WEIGHTS at the npz. The test skips if the weights are not present (test_shapes.py always runs).

    ABLANG_WEIGHTS=/path/to/ablang2_weights.npz python -m pytest test_agreement.py
    ABLANG_WEIGHTS=/path/to/ablang2_weights.npz python test_agreement.py
"""
import os

import numpy as np
import jax.numpy as jnp

from ablang_jax import Ablanx, load_ablanx_params

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("ABLANG_WEIGHTS", os.path.join(HERE, "ablang2_weights.npz"))
GOLDEN = os.path.join(HERE, "golden_ablang2_pert_vh.npz")
MAX_ABS_TOL = 1e-4
MIN_COS_TOL = 0.9999


def _run():
    tree = load_ablanx_params(dict(np.load(WEIGHTS)))
    g = np.load(GOLDEN)
    ids = g["ids"][None].astype(np.int32)
    ref = g["emb"].astype(np.float32)
    mask = np.ones_like(ids, np.float32)
    hid, _ = Ablanx().apply({"params": tree}, jnp.asarray(ids), jnp.asarray(mask), return_attn=False)
    jx = np.asarray(hid[0], np.float32)
    max_abs = float(np.abs(jx - ref).max())
    cos = (jx * ref).sum(-1) / (np.linalg.norm(jx, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-9)
    return max_abs, float(cos.min())


def test_matches_reference_embeddings():
    import pytest
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"set ABLANG_WEIGHTS to the AbRep npz (see export_weights.py); not found at {WEIGHTS}")
    max_abs, min_cos = _run()
    assert max_abs < MAX_ABS_TOL, f"max_abs {max_abs:.3e} >= {MAX_ABS_TOL}"
    assert min_cos > MIN_COS_TOL, f"min_cos {min_cos:.6f} <= {MIN_COS_TOL}"


if __name__ == "__main__":
    if not os.path.exists(WEIGHTS):
        raise SystemExit(f"set ABLANG_WEIGHTS to the AbRep npz (see export_weights.py); not found at {WEIGHTS}")
    ma, mc = _run()
    print(f"max_abs={ma:.3e}  min_cos={mc:.6f}")
    assert ma < MAX_ABS_TOL and mc > MIN_COS_TOL
    print("AGREEMENT OK")
