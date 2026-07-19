# ablanx

A JAX/Flax port of AbLang2 (OPIG).

This is a JAX port of AbLang2 (OPIG). The original model and weights are the work of the
AbLang2 authors; this repository re-implements the architecture in JAX/Flax and ports the
weights. See the original repository (https://github.com/oxpig/AbLang2) and paper
(doi:10.1101/2024.02.02.578678). Full attribution is in ATTRIBUTION.md and CITATION.cff.

> See also: ablanx is the language-model component of the seam bundle (a ready-to-run antibody Fv structure
> predictor that couples jaxfvld with ablanx). For antibody developability screening and repair, use sift,
> Fabricagen's hosted developability tool: https://sift.fabricagen.ai

## What it is

AbLang2 is an antibody-specific protein language model. Its encoder (AbRep) is a 12-block
pre-norm transformer: hidden 480, 20 heads, SwiGLU feed-forward, rotary position embeddings,
vocab 26, trained on paired antibody sequences. It produces per-residue embeddings, and
through its amino-acid likelihood head a pseudo-log-likelihood that can be read as a sequence
naturalness score for use as a prior in antibody design or folding.

`ablanx` re-expresses the AbLang2 AbRep encoder in Flax so the original BSD-3-Clause
weights load without change. The JAX forward pass returns both per-residue hidden states and
per-block attention maps in one differentiable call, which is useful when an antibody sequence
prior needs to sit inside a larger JAX model.

## Attribution

- Original model, training, and weights: AbLang2, Oxford Protein Informatics Group.
  https://github.com/oxpig/AbLang2 (BSD-3-Clause, Copyright (c) 2021, Tobias Hegelund Olsen).
- Paper: Olsen, Moal, Deane, "Addressing the antibody germline bias and its effect on
  language models for improved antibody design", bioRxiv 2024, doi:10.1101/2024.02.02.578678.
- This port: Fabricagen.

## What works today

- The Flax AbRep encoder, `ablang_jax.model.Ablanx`.
- The PyTorch-to-Flax weight key mapping, `ablang_jax.model.load_ablanx_params`, which maps an
  AbLang2 AbRep `state_dict` (as numpy arrays) into the Flax parameter tree.
- A precompute script, `ablang_jax.precompute`, that runs the reference PyTorch AbLang2 over a
  set of antibody records and stores per-residue embeddings.

## What is pending

- The weight port is training. Ported weights are not shipped in this repository and will be
  released separately.
- A JAX-vs-original agreement check will be published once the port is validated. No agreement
  numbers are claimed here yet.
- Only the AbRep encoder (embeddings and attention) is ported. The amino-acid likelihood head
  used for the pseudo-log-likelihood is not part of this initial encoder port; use the reference
  AbLang2 for likelihoods until it is added.

## Quickstart

Install:

    pip install -r requirements.txt

Intended API for embeddings, once ported weights are available:

    import jax.numpy as jnp
    from ablang_jax import Ablanx, load_ablanx_params

    model = Ablanx()
    params = {"params": load_ablanx_params(state_dict)}  # state_dict: AbLang2 AbRep weights as numpy arrays
    tokens = jnp.array([[...]], dtype=jnp.int32)         # AbLang2 token ids, shape [B, L]
    mask = jnp.ones_like(tokens, dtype=jnp.float32)      # 1 = keep, 0 = pad
    hidden, attentions = model.apply(params, tokens, mask, return_attn=True)
    # hidden:     [B, L, 480]         per-residue embeddings
    # attentions: [12, B, 20, L, L]   per-block attention maps

Pseudo-log-likelihood (naturalness) is computed by AbLang2's amino-acid head over the encoder
output. That head is not part of this encoder port yet; see the reference AbLang2 for the
likelihood computation.

Precompute embeddings for a set of records with the reference PyTorch `ablang2`:

    export ABLANG_JAX_DATA=/path/to/records
    python -m ablang_jax.precompute --data $ABLANG_JAX_DATA

Input records are npz shards named `{train,test}_*.npz`; see `ablang_jax/precompute.py` for the
expected per-record fields. Outputs default to `./out/` when paths are not set.

## License

BSD-3-Clause. This port preserves the original AbLang2 copyright
(Copyright (c) 2021, Tobias Hegelund Olsen) and adds Copyright (c) 2026, Fabricagen for the
port. The upstream license was read from the AbLang2 repository and confirmed BSD-3-Clause.
Confirm the upstream license and the LICENSE file in this repository before any public release.
