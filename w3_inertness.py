#!/usr/bin/env python3
"""
W-3 Stage R0 inertness assertion (plan §2.1).

Fingerprints build_model(arch) for arch in {resnet, resnet_se}: the post-
construction torch RNG state AND the concatenated initial parameter bytes AND
the parameter count. The use_residual edit must leave all three byte-identical
for these two archs; reproducing a number is not enough because run-to-run SD
is 0.47 pp.

  python w3_inertness.py --save  w3_fp_before.json   # on pre-change code
  python w3_inertness.py --check w3_fp_before.json   # after the edit
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
import numpy as np
import torch
from cnn_architectures import build_model, count_params

ARCHS = ["resnet", "resnet_se"]


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
    # also record resnet_nores count if it exists now (post-change only)
    try:
        fp["resnet_nores"] = fingerprint("resnet_nores")
    except Exception:
        pass

    if args.save:
        json.dump(fp, open(args.save, "w"), indent=2)
        print(f"saved fingerprint -> {args.save}")
        for a in ARCHS:
            print(f"  {a:12} n_params={fp[a]['n_params']:>8}  rng={fp[a]['rng_sha256'][:12]}  "
                  f"params={fp[a]['params_sha256'][:12]}")
        return 0

    before = json.load(open(args.check))
    ok = True
    for a in ARCHS:
        for key in ("rng_sha256", "params_sha256", "n_params"):
            b, n = before[a][key], fp[a][key]
            same = (b == n)
            ok &= same
            if not same:
                print(f"  MISMATCH {a}.{key}: before={b}  after={n}")
    for a in ARCHS:
        print(f"  {a:12} n_params={fp[a]['n_params']:>8}  rng {'OK' if before[a]['rng_sha256']==fp[a]['rng_sha256'] else 'DIFF'}  "
              f"params {'OK' if before[a]['params_sha256']==fp[a]['params_sha256'] else 'DIFF'}")
    if "resnet_nores" in fp:
        print(f"  resnet_nores n_params={fp['resnet_nores']['n_params']:>8}  count_params={fp['resnet_nores']['count_params']}")
    print("\nINERTNESS: PASS" if ok else "\nINERTNESS: FAIL -- the edit reordered or consumed random draws; fix before running")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
