#!/usr/bin/env python3
"""
W-5 Stage 3 inertness assertion (plan §3).

Fingerprints build_model(arch) for arch in {resnet, resnet_se, resnet_nores}:
post-construction torch RNG state, concatenated initial parameter bytes, and
parameter count. Adding optional `widths` / `blocks_per_stage` to build_model
and the two --flags must leave all three byte-identical for these archs when the
new flags are absent. Same contract as w3_inertness.py / w4_inertness.py.

  python w5_inertness.py --save  w5_fp_before.json   # on pre-change code
  python w5_inertness.py --check w5_fp_before.json   # after the edit
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
import numpy as np
import torch
from cnn_architectures import build_model, count_params

ARCHS = ["resnet", "resnet_se", "resnet_nores"]


def fingerprint(arch: str, seed: int = 42) -> dict:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = build_model(arch, 9, 4)
    rng_bytes = torch.get_rng_state().numpy().tobytes()
    param_bytes = b"".join(p.detach().cpu().numpy().tobytes() for p in m.parameters())
    return {
        "rng_sha256": hashlib.sha256(rng_bytes).hexdigest(),
        "params_sha256": hashlib.sha256(param_bytes).hexdigest(),
        "n_params": int(sum(p.numel() for p in m.parameters())),
        "count_params": int(count_params(m)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save")
    g.add_argument("--check")
    args = ap.parse_args()

    fp = {a: fingerprint(a) for a in ARCHS}

    if args.save:
        json.dump(fp, open(args.save, "w"), indent=2)
        print(f"saved -> {args.save}")
        for a in ARCHS:
            print(f"  {a:14} n_params={fp[a]['n_params']:>8}  rng={fp[a]['rng_sha256'][:12]}  "
                  f"params={fp[a]['params_sha256'][:12]}")
        return 0

    before = json.load(open(args.check))
    ok = True
    for a in ARCHS:
        for key in ("rng_sha256", "params_sha256", "n_params"):
            same = before[a][key] == fp[a][key]
            ok &= same
            if not same:
                print(f"  MISMATCH {a}.{key}: before={before[a][key]} after={fp[a][key]}")
        print(f"  {a:14} n_params={fp[a]['n_params']:>8}  "
              f"rng {'OK' if before[a]['rng_sha256']==fp[a]['rng_sha256'] else 'DIFF'}  "
              f"params {'OK' if before[a]['params_sha256']==fp[a]['params_sha256'] else 'DIFF'}")
    print("\nINERTNESS: PASS" if ok else "\nINERTNESS: FAIL -- stop, initialization moved")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
