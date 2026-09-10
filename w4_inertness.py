#!/usr/bin/env python3
"""
W-4 Stage 2.2 inertness assertion (plan §2.2, master doc assertion 1).

For all five existing augmentation modes, imported from BOTH train_cnn_loso and
run_cnn_arch_loso: the output tensor bytes AND the post-call torch RNG state
must be byte-identical before and after adding the `gainjitter` branch. A
reproduced number is not acceptable (run-to-run SD 0.47 pp).

  python w4_inertness.py --save  w4_fp_before.json    # on pre-change code
  python w4_inertness.py --check w4_fp_before.json    # after the edit
"""
from __future__ import annotations
import argparse, hashlib, json, sys
import torch

MODES = ["none", "gaussian", "chandrop", "timemask", "combined"]


def _sources():
    from train_cnn_loso import augment_batch as ab_train
    from run_cnn_arch_loso import augment_batch as ab_run
    return {"train_cnn_loso": ab_train, "run_cnn_arch_loso": ab_run}


def fingerprint() -> dict:
    fp = {}
    for src, ab in _sources().items():
        fp[src] = {}
        for mode in MODES:
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
        for src in fp:
            for mode in MODES:
                print(f"  {src:18} {mode:9} out={fp[src][mode]['out_sha256'][:12]} "
                      f"rng={fp[src][mode]['rng_sha256'][:12]}")
        return 0

    before = json.load(open(args.check))
    ok = True
    for src in before:
        for mode in MODES:
            b, n = before[src][mode], fp[src][mode]
            for key in ("out_sha256", "rng_sha256"):
                same = b[key] == n[key]
                ok &= same
                mark = "OK" if same else "MISMATCH"
                if not same:
                    print(f"  {src} {mode} {key}: {mark}  before={b[key][:16]} after={n[key][:16]}")
            print(f"  {src:18} {mode:9} out {'OK' if b['out_sha256']==n['out_sha256'] else 'DIFF'}  "
                  f"rng {'OK' if b['rng_sha256']==n['rng_sha256'] else 'DIFF'}")
    print("\nINERTNESS: PASS" if ok else "\nINERTNESS: FAIL -- the gainjitter branch consumes randomness it should not; fix before running")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
