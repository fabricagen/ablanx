"""ablang_jax - a JAX/Flax port of AbLang2 (OPIG).

This package re-implements AbLang2's AbRep encoder in Flax and provides the state_dict key
mapping needed to load the original BSD-3-Clause PyTorch weights unchanged. The AbLang2 model
and weights are the work of the AbLang2 authors (OPIG); this is a port. See ATTRIBUTION.md and
https://github.com/oxpig/AbLang2.
"""
from .model import (
    HEAD_DIM,
    HID,
    LN_EPS,
    N_BLOCK,
    N_HEAD,
    VOCAB,
    Ablanx,
    Block,
    load_ablanx_params,
)

__all__ = [
    "Ablanx",
    "Block",
    "load_ablanx_params",
    "VOCAB",
    "HID",
    "N_HEAD",
    "N_BLOCK",
    "HEAD_DIM",
    "LN_EPS",
]

__version__ = "0.1.0"
