#!/usr/bin/env python3
"""Export AbLang2's AbRep weights to the npz that `ablang_jax.load_ablanx_params` consumes.

Runs the reference PyTorch AbLang2 (`pip install ablang2 torch`), pulls the AbRep encoder state_dict, and
writes each tensor as a numpy array under its original `AbRep.*` key. This is the provenance path: the
resulting npz is the reference AbRep weights unmodified (207 tensors). Verified byte-equal to the released
weights (max abs diff 0.0). Use this if you would rather regenerate the weights from the original AbLang2
than download the release asset.

    pip install ablang2 torch
    python export_weights.py --out ablang2_weights.npz

Then load them:

    import numpy as np
    from ablang_jax import Ablanx, load_ablanx_params
    tree = load_ablanx_params(dict(np.load("ablang2_weights.npz")))
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="ablang2_weights.npz")
    args = ap.parse_args()
    import ablang2

    model = ablang2.pretrained()
    sd = model.AbLang.state_dict()
    out = {k: v.detach().cpu().numpy() for k, v in sd.items() if k.startswith("AbRep.")}
    if not out:
        raise SystemExit("no AbRep.* tensors found in the reference state_dict; check the ablang2 version")
    np.savez(args.out, **out)
    print(f"wrote {args.out}: {len(out)} AbRep tensors")


if __name__ == "__main__":
    main()
