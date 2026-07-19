#!/usr/bin/env python3
"""ablang_jax.precompute - precompute AbLang2 per-residue embeddings for a set of antibody records.

Runs the reference PyTorch AbLang2 (`pip install ablang2`) over antibody Fv sequences stored as
sharded .npz files and writes a per-record `seq_emb` field ([L, 480], VH then VL, residue positions
only). These embeddings can be used as a fixed antibody sequence prior.

This script uses the original PyTorch AbLang2 package, not the JAX port in this repository. The port
exists to run the same encoder differentiably in JAX once the ported weights are available; this
precompute path is provided so embeddings can be materialised today with the reference model.

Expected input: one or more npz shards named `{split}_*.npz` (split in {train, test}) under the data
directory, each an object-array store with per-record fields:
    pdb_id      [N]        record id
    aa          [N][Li]    integer amino-acid indices in the order "ARNDCQEGHILKMFPSTWYV"
    chain_id    [N][Li]    0 = heavy (VH), 1 = light (VL)
    res_present [N][Li]    bool mask of resolved residues
Each shard is rewritten in place with an added `seq_emb` object field.

    python -m ablang_jax.precompute --data /path/to/records
    # or set ABLANG_JAX_DATA (default: ./out/records)
"""
from __future__ import annotations

import argparse
import glob
import os

import ablang2
import numpy as np

_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"   # AlphaFold-style restype index order

_DEFAULT_OUT = os.environ.get("ABLANG_JAX_OUT", "./out")
_DEFAULT_DATA = os.environ.get("ABLANG_JAX_DATA", os.path.join(_DEFAULT_OUT, "records"))


def embed_fv(model, vh_seq, vl_seq):
    """[L,480] per-residue AbLang2 embedding, residue positions only (ids 1..20), VH then VL."""
    pair = [[vh_seq, vl_seq]] if vl_seq else [[vh_seq, ""]]
    emb = np.asarray(model(pair, mode="rescoding")[0], np.float32)          # [T_tok, 480]
    ids = np.asarray(model.tokenizer(pair, pad=True))[0]                    # [T_tok]
    keep = (ids >= 1) & (ids <= 20)                                         # 20 standard AAs
    return emb[: len(ids)][keep[: emb.shape[0]]]                            # [n_residues, 480]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=_DEFAULT_DATA,
                    help="directory of {train,test}_*.npz shards (env: ABLANG_JAX_DATA)")
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()
    model = ablang2.pretrained()

    for split in ("train", "test"):
        shards = sorted(glob.glob(os.path.join(args.data, f"{split}_*.npz")))
        if args.limit_shards:
            shards = shards[: args.limit_shards]
        for si, shard in enumerate(shards):
            z = np.load(shard, allow_pickle=True); arrs = {k: z[k] for k in z.files}
            n = len(arrs["pdb_id"])
            embs = []
            for i in range(n):
                aa = np.asarray(arrs["aa"][i]); cid = np.asarray(arrs["chain_id"][i])
                pres = np.asarray(arrs["res_present"][i], bool)
                vh = "".join(_AA_ORDER[a] for a in aa[(cid == 0) & pres])
                vl = "".join(_AA_ORDER[a] for a in aa[(cid == 1) & pres])
                e = embed_fv(model, vh, vl)                                 # [len(vh)+len(vl), 480]
                exp = len(vh) + len(vl)
                if e.shape[0] != exp:                                        # alignment guard
                    e = e[:exp] if e.shape[0] > exp else np.pad(e, [(0, exp - e.shape[0]), (0, 0)])
                embs.append(e.astype(np.float32))
            arrs["seq_emb"] = np.array(embs, dtype=object)
            np.savez_compressed(shard, **arrs)
            if si % 5 == 0:
                print(f"{split} shard {si+1}/{len(shards)} emb[0]={embs[0].shape}", flush=True)
        print(f"{split}: DONE ({len(shards)} shards)", flush=True)


if __name__ == "__main__":
    main()
