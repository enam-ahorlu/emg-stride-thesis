#!/usr/bin/env python3
"""
P-5 / P-6 inertness assertion (plan sections 5A.2 and 5B.6).

Adding the `subset` mode (P-5) and the `mpchandrop` mean-preserving mode (P-6)
to augment_batch must leave every EXISTING mode byte-for-byte unchanged. This
follows the w4_inertness.py / w3_inertness.py precedent: a reproduced number is
not enough (run-to-run SD 0.47 pp), so we fingerprint

  1. augment_batch output bytes AND post-call torch RNG state, for all six
     existing modes, imported from BOTH train_cnn_loso and run_cnn_arch_loso;
  2. build_model post-construction RNG state, concatenated initial-parameter
     bytes and parameter count, for arch in {resnet, resnet_se}.

  python p5p6_inertness.py --save  p5p6_fp_before.json   # on pre-change code
  python p5p6_inertness.py --check p5p6_fp_before.json   # after the edits
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
import numpy as np
import torch

EXISTING_MODES = ["none", "gaussian", "chandrop", "timemask", "combined", "gainjitter"]
ARCHS = ["resnet", "resnet_se"]


def _aug_sources():
    from train_cnn_loso import augment_batch as ab_train
    from run_cnn_arch_loso import augment_batch as ab_run
    return {"train_cnn_loso": ab_train, "run_cnn_arch_loso": ab_run}


def aug_fingerprint() -> dict:
    fp = {}
    for src, ab in _aug_sources().items():
        fp[src] = {}
        for mode in EXISTING_MODES:
            torch.manual_seed(42)
            X = torch.randn(8, 9, 500)
            torch.manual_seed(7)
            out = ab(X.clone(), mode=mode, sigma=0.1, chandrop_p=0.2, mask_frac=0.15)
            state = torch.get_rng_state().numpy().tobytes()
            fp[src][mode] = {
                "out_sha256": hashlib.sha256(out.detach().cpu().numpy().tobytes()).hexdigest(),
                "rng_sha256": hashlib.sha256(state).hexdigest(),
            }
    return fp


def model_fingerprint(arch: str, seed: int = 42) -> dict:
    from cnn_architectures import build_model, count_params
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


def fingerprint() -> dict:
    return {"aug": aug_fingerprint(), "model": {a: model_fingerprint(a) for a in ARCHS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save")
    g.add_argument("--check")
    args = ap.parse_args()

    fp = fingerprint()
    if args.save:
        json.dump(fp, open(args.save, "w"), indent=2)
        print(f"saved -> {args.save}")
        for src in fp["aug"]:
            for mode in EXISTING_MODES:
                e = fp["aug"][src][mode]
                print(f"  {src:18} {mode:10} out={e['out_sha256'][:12]} rng={e['rng_sha256'][:12]}")
        for a in ARCHS:
            print(f"  {a:12} n_params={fp['model'][a]['n_params']:>8} "
                  f"params={fp['model'][a]['params_sha256'][:12]}")
        return 0

    before = json.load(open(args.check))
    ok = True
    for src in before["aug"]:
        for mode in EXISTING_MODES:
            b, n = before["aug"][src][mode], fp["aug"][src][mode]
            for key in ("out_sha256", "rng_sha256"):
                same = b[key] == n[key]
                ok &= same
                if not same:
                    print(f"  MISMATCH {src} {mode} {key}: before={b[key][:16]} after={n[key][:16]}")
            print(f"  {src:18} {mode:10} out {'OK' if b['out_sha256']==n['out_sha256'] else 'DIFF'}  "
                  f"rng {'OK' if b['rng_sha256']==n['rng_sha256'] else 'DIFF'}")
    for a in ARCHS:
        for key in ("rng_sha256", "params_sha256", "n_params"):
            b, n = before["model"][a][key], fp["model"][a][key]
            same = (b == n)
            ok &= same
            if not same:
                print(f"  MISMATCH model {a}.{key}: before={b} after={n}")
        print(f"  {a:12} params {'OK' if before['model'][a]['params_sha256']==fp['model'][a]['params_sha256'] else 'DIFF'}  "
              f"n_params {'OK' if before['model'][a]['n_params']==fp['model'][a]['n_params'] else 'DIFF'}")
    print("\nINERTNESS: PASS" if ok else
          "\nINERTNESS: FAIL -- an existing mode changed; fix before running")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
