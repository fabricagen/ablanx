"""Flax reimplementation of AbLang2's AbRep encoder.

A 12-block pre-norm transformer (hidden 480, 20 heads, SwiGLU feed-forward, rotary positions,
vocab 26) that reproduces AbLang2's AbRep so the original BSD-3-Clause PyTorch weights load
unchanged. A single forward returns per-residue hidden states and per-block attention maps.

PyTorch Linear weights [out, in] map to Flax Dense kernels [in, out] by transpose; see
load_ablanx_params for the full state_dict key mapping. Outputs match reference PyTorch AbLang2 to under
4e-6 across a 30-Fv panel (test_agreement.py).

AbLang2 is the work of the Oxford Protein Informatics Group; this is a port. See ATTRIBUTION.md and
https://github.com/oxpig/AbLang2.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

VOCAB = 26
HID = 480
N_HEAD = 20
N_BLOCK = 12
HEAD_DIM = HID // N_HEAD          # 24
LN_EPS = 1e-12


def _rope_freqs(head_dim, base=10000.0):
    # inverse frequencies over half the head dim (matches AbLang2's rotary_emb.freqs)
    half = head_dim // 2
    return 1.0 / (base ** (jnp.arange(0, half, dtype=jnp.float32) * 2.0 / head_dim))


def _apply_rope(x, freqs):
    """Rotary embedding on x [B, L, H, D], interleaved convention (as AbLang2 uses)."""
    L = x.shape[1]
    pos = jnp.arange(L, dtype=jnp.float32)
    ang = pos[:, None] * freqs[None, :]
    ang = jnp.repeat(ang, 2, axis=-1)
    cos = jnp.cos(ang)[None, :, None, :]
    sin = jnp.sin(ang)[None, :, None, :]
    x_even = x[..., 0::2]; x_odd = x[..., 1::2]
    rot = jnp.stack([-x_odd, x_even], axis=-1).reshape(x.shape)
    return x * cos + rot * sin


class Block(nn.Module):
    @nn.compact
    def __call__(self, x, mask, freqs, return_attn=False):
        B, L, _ = x.shape
        h = nn.LayerNorm(epsilon=LN_EPS, name="pre_attn_layer_norm")(x)
        q = nn.Dense(HID, name="q_proj")(h).reshape(B, L, N_HEAD, HEAD_DIM)
        k = nn.Dense(HID, name="k_proj")(h).reshape(B, L, N_HEAD, HEAD_DIM)
        v = nn.Dense(HID, name="v_proj")(h).reshape(B, L, N_HEAD, HEAD_DIM)
        q = _apply_rope(q, freqs); k = _apply_rope(k, freqs)
        # AbLang2's net attention scale is 1/head_dim (q scaled by head_dim^-0.5, then divided by sqrt(head_dim))
        logits = jnp.einsum("blhd,bmhd->bhlm", q, k) / HEAD_DIM
        logits = logits + (1.0 - mask[:, None, None, :]) * -1e9
        attn = jax.nn.softmax(logits, axis=-1)
        o = jnp.einsum("bhlm,bmhd->blhd", attn, v).reshape(B, L, HID)
        o = nn.Dense(HID, name="out_proj")(o)
        x = x + o
        # SwiGLU: split the feed-forward projection into value and gate, gate with silu
        h2 = nn.LayerNorm(epsilon=LN_EPS, name="final_layer_norm")(x)
        up = nn.Dense(3840, name="ffn_in")(h2)
        val, gate = jnp.split(up, 2, axis=-1)
        x = x + nn.Dense(HID, name="ffn_out")(jax.nn.silu(gate) * val)
        return (x, attn) if return_attn else (x, None)


class Ablanx(nn.Module):
    """AbLang2 AbRep encoder. Returns hidden [B, L, 480] and, optionally, attention [N_BLOCK, B, H, L, L]."""
    @nn.compact
    def __call__(self, tokens, mask, return_attn=True):
        freqs = _rope_freqs(HEAD_DIM)
        x = nn.Embed(VOCAB, HID, name="aa_embed_layer")(tokens.astype("int32"))
        attns = []
        for i in range(N_BLOCK):
            x, a = Block(name=f"encoder_blocks_{i}")(x, mask, freqs, return_attn=return_attn)
            if return_attn:
                attns.append(a)
        # final layer norm after all blocks; this is what rescoding returns
        x = nn.LayerNorm(epsilon=LN_EPS, name="layer_norm_after_encoder_blocks")(x)
        return x, (jnp.stack(attns) if return_attn else None)


def load_ablanx_params(w):
    """Map AbLang2 AbRep PyTorch state_dict (dict of numpy arrays, keys 'AbRep....') -> flax param tree.
    PyTorch Linear weight [out,in] -> flax Dense kernel [in,out] (transpose). LayerNorm weight->scale."""
    import numpy as np
    def T(k): return np.asarray(w[k]).T
    def V(k): return np.asarray(w[k])
    p = {"aa_embed_layer": {"embedding": V("AbRep.aa_embed_layer.weight")},
         "layer_norm_after_encoder_blocks": {"scale": V("AbRep.layer_norm_after_encoder_blocks.weight"),
                                             "bias": V("AbRep.layer_norm_after_encoder_blocks.bias")}}
    for i in range(N_BLOCK):
        b = f"AbRep.encoder_blocks.{i}"
        p[f"encoder_blocks_{i}"] = {
            "pre_attn_layer_norm": {"scale": V(f"{b}.pre_attn_layer_norm.weight"),
                                    "bias": V(f"{b}.pre_attn_layer_norm.bias")},
            "q_proj": {"kernel": T(f"{b}.multihead_attention.q_proj.weight"), "bias": V(f"{b}.multihead_attention.q_proj.bias")},
            "k_proj": {"kernel": T(f"{b}.multihead_attention.k_proj.weight"), "bias": V(f"{b}.multihead_attention.k_proj.bias")},
            "v_proj": {"kernel": T(f"{b}.multihead_attention.v_proj.weight"), "bias": V(f"{b}.multihead_attention.v_proj.bias")},
            "out_proj": {"kernel": T(f"{b}.multihead_attention.out_proj.weight"), "bias": V(f"{b}.multihead_attention.out_proj.bias")},
            "ffn_in": {"kernel": T(f"{b}.intermediate_layer.0.weight"), "bias": V(f"{b}.intermediate_layer.0.bias")},
            "ffn_out": {"kernel": T(f"{b}.intermediate_layer.2.weight"), "bias": V(f"{b}.intermediate_layer.2.bias")},
            "final_layer_norm": {"scale": V(f"{b}.final_layer_norm.weight"), "bias": V(f"{b}.final_layer_norm.bias")},
        }
    return p
