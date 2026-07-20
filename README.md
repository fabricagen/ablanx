# ablanx

A JAX/Flax port of AbLang2 (OPIG). The original model and weights are the work of the AbLang2
authors; this repository re-implements the AbRep encoder in JAX/Flax and loads the original weights
unchanged. See the upstream repository (https://github.com/oxpig/AbLang2) and paper
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
- A weight exporter, `export_weights.py`, that writes the AbRep weight npz from the reference model.

## Validation

The JAX forward reproduces reference PyTorch AbLang2 to float32 precision. On VH-only and paired VH+VL
Fvs, per-residue embeddings match the reference with maximum absolute difference 2.5e-6 and cosine
1.000000, using the same weights. The weights are byte-identical to the reference AbRep state_dict (207
tensors, exact match). Reproduce with `test_agreement.py`.

## Weights

The weights are the original AbLang2 AbRep weights, unmodified (207 tensors). Obtain them either way:

- Export from the reference model, no download:
  `pip install ablang2 torch && python export_weights.py` writes `ablang2_weights.npz`, verified
  byte-identical to the reference.
- Or download the released `ablang2_weights.npz` asset (torch-free path).

Point `ABLANG_WEIGHTS` at the npz, or pass `dict(np.load(...))` to `load_ablanx_params`.

## Not included

- Only the AbRep encoder (embeddings and attention) is ported. The amino-acid likelihood head used for the
  pseudo-log-likelihood is not part of this encoder port; use the reference AbLang2 for likelihoods.

## Quickstart

Install:

    pip install -r requirements.txt

Embeddings:

    import numpy as np, jax.numpy as jnp
    from ablang_jax import Ablanx, load_ablanx_params

    model = Ablanx()
    params = {"params": load_ablanx_params(dict(np.load("ablang2_weights.npz")))}  # AbRep weights
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

## Tests

    python test_shapes.py                                          # forward, weight-key mapping, masking
    ABLANG_WEIGHTS=ablang2_weights.npz python test_agreement.py    # agreement vs reference (needs weights)

`test_shapes.py` needs no weights. `test_agreement.py` loads the committed golden fixture
(`golden_ablang2_pert_vh.npz`, reference embeddings for one Fv) and checks the JAX forward matches it; it
skips under pytest if `ABLANG_WEIGHTS` is unset.

## License

BSD-3-Clause. This port preserves the original AbLang2 copyright
(Copyright (c) 2021, Tobias Hegelund Olsen) and adds Copyright (c) 2026, Fabricagen for the
port. The upstream AbLang2 license was confirmed BSD-3-Clause. See `ATTRIBUTION.md`.
