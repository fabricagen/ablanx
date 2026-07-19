# Attribution

`ablang-jax` is a JAX/Flax port of **AbLang2**, the antibody-specific protein language
model from the Oxford Protein Informatics Group (OPIG).

## Original work

- **AbLang2** (model, training, and weights).
  - Repository: https://github.com/oxpig/AbLang2
  - License: BSD-3-Clause, Copyright (c) 2021, Tobias Hegelund Olsen
  - Paper: Tobias H. Olsen, Iain H. Moal, Charlotte M. Deane.
    "Addressing the antibody germline bias and its effect on language models for
    improved antibody design." bioRxiv (2024). doi:10.1101/2024.02.02.578678

The AbLang2 architecture and all pretrained weights are the work of the AbLang2 authors.
This repository does not claim authorship of the model or its weights.

## This port

- **ablang-jax** (a re-implementation of AbLang2's AbRep encoder in JAX/Flax, plus the
  state_dict key mapping needed to load the original PyTorch weights unchanged).
  - Copyright (c) 2026, Fabricagen
  - License: BSD-3-Clause (same as the original)

The port re-expresses the published architecture. It does not modify or retrain the model.
Ported weights are released separately once the port is validated against the reference
PyTorch AbLang2.

## License verification

The upstream license was read from the AbLang2 repository LICENSE file
(https://github.com/oxpig/AbLang2) and confirmed to be BSD-3-Clause with copyright
"Copyright (c) 2021, Tobias Hegelund Olsen". Confirm this against the upstream repository
before any public release.
