"""Shape, weight-mapping, and masking smoke tests for the AbRep encoder port.

These do not check numerical agreement with reference AbLang2 (that needs the ported weights and is
published with them). They check that the Flax forward runs, that `load_ablanx_params` covers the AbLang2
AbRep state_dict keys with shape-compatible tensors, and that key masking works.

    python -m pytest test_shapes.py
    # or: python test_shapes.py
"""
import numpy as np
import jax
import jax.numpy as jnp

from ablang_jax import Ablanx, load_ablanx_params, VOCAB, HID, N_HEAD, N_BLOCK, HEAD_DIM

L = 24


def _synthetic_state_dict(seed=0):
    """An AbLang2 AbRep state_dict of the right keys and shapes, filled with small random values."""
    rng = np.random.default_rng(seed)
    r = lambda *s: (rng.standard_normal(s) * 0.02).astype(np.float32)
    sd = {"AbRep.aa_embed_layer.weight": r(VOCAB, HID),
          "AbRep.layer_norm_after_encoder_blocks.weight": r(HID),
          "AbRep.layer_norm_after_encoder_blocks.bias": r(HID)}
    for i in range(N_BLOCK):
        b = f"AbRep.encoder_blocks.{i}"
        sd[f"{b}.pre_attn_layer_norm.weight"] = r(HID)
        sd[f"{b}.pre_attn_layer_norm.bias"] = r(HID)
        for pj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            sd[f"{b}.multihead_attention.{pj}.weight"] = r(HID, HID)   # PyTorch Linear [out, in]
            sd[f"{b}.multihead_attention.{pj}.bias"] = r(HID)
        sd[f"{b}.intermediate_layer.0.weight"] = r(3840, HID)
        sd[f"{b}.intermediate_layer.0.bias"] = r(3840)
        sd[f"{b}.intermediate_layer.2.weight"] = r(HID, 1920)
        sd[f"{b}.intermediate_layer.2.bias"] = r(HID)
        sd[f"{b}.final_layer_norm.weight"] = r(HID)
        sd[f"{b}.final_layer_norm.bias"] = r(HID)
    return sd


def _tokens(L=L):
    return jnp.array([[i % VOCAB for i in range(L)]], dtype=jnp.int32)


def test_config():
    assert HEAD_DIM == HID // N_HEAD == 24
    assert (HID, N_HEAD, N_BLOCK, VOCAB) == (480, 20, 12, 26)


def test_random_init_forward():
    model, toks = Ablanx(), _tokens()
    mask = jnp.ones((1, L), jnp.float32)
    params = model.init(jax.random.PRNGKey(0), toks, mask)
    hidden, attn = model.apply(params, toks, mask, return_attn=True)
    assert hidden.shape == (1, L, HID)
    assert attn.shape == (N_BLOCK, 1, N_HEAD, L, L)


def test_weight_mapping_loads_and_runs():
    model, toks = Ablanx(), _tokens()
    mask = jnp.ones((1, L), jnp.float32)
    tree = load_ablanx_params(_synthetic_state_dict())
    hidden, attn = model.apply({"params": tree}, toks, mask, return_attn=True)
    assert hidden.shape == (1, L, HID)
    assert attn.shape == (N_BLOCK, 1, N_HEAD, L, L)
    assert bool(jnp.all(jnp.isfinite(hidden)))


def test_key_masking_zeroes_padded_keys():
    model, toks = Ablanx(), _tokens()
    tree = load_ablanx_params(_synthetic_state_dict())
    mask = jnp.array([[1.0] * (L - 4) + [0.0] * 4], jnp.float32)
    _, attn = model.apply({"params": tree}, toks, mask, return_attn=True)
    assert float(jnp.max(attn[..., -4:])) < 1e-6   # no attention to padded keys


if __name__ == "__main__":
    test_config()
    test_random_init_forward()
    test_weight_mapping_loads_and_runs()
    test_key_masking_zeroes_padded_keys()
    print("ALL SHAPE/MAPPING/MASK CHECKS PASSED")
