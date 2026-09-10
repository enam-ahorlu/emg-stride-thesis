#!/usr/bin/env python3
"""
P-6 multiplier gate (plan section 5B.2).

The mean-preserving channel-dropout arm multiplies by Bernoulli(1 - p') / (1 - p').
p' MUST be derived in code from the requested multiplicative SD, not hard-coded:

    SD(multiplier) = sqrt(p' / (1 - p'))  ==  requested_sd
    =>  p' = requested_sd**2 / (1 + requested_sd**2)

This script calls the exact `mpchandrop` branch of augment_batch on a large
batch and asserts the realized multiplier has mean ~ 1 and SD ~ requested_sd.
It also confirms the derived p' matches the analytic value and reports the
p = 0.2 chandrop multiplier moments for reference (mean 0.8, SD 0.4), which is
the mean that W-4 left unmatched.

  python p6_multiplier_gate.py            # requested SD = 0.40 (core P-6 arm)
  python p6_multiplier_gate.py 0.30 0.50  # extension arms
"""
from __future__ import annotations
import sys
import torch

from train_cnn_loso import augment_batch


def check(requested_sd: float, tol_mean: float = 5e-3, tol_sd: float = 5e-3) -> bool:
    p_prime_analytic = requested_sd ** 2 / (1.0 + requested_sd ** 2)

    # recover the multiplier by dividing the augmented ones-tensor by the input
    torch.manual_seed(42)
    N, C, T = 4096, 9, 64
    X = torch.ones(N, C, T)
    out = augment_batch(X.clone(), mode="mpchandrop", sigma=0.1, chandrop_p=0.2,
                        mask_frac=0.15, gain_sd=requested_sd)
    mult = out[:, :, 0]                         # (N, C); constant over T
    realized_mean = mult.mean().item()
    realized_sd = mult.std(unbiased=False).item()
    zero_frac = (mult == 0).float().mean().item()

    ok_mean = abs(realized_mean - 1.0) < tol_mean
    ok_sd = abs(realized_sd - requested_sd) < tol_sd
    ok_pprime = abs(zero_frac - p_prime_analytic) < 0.01

    print(f"--- mpchandrop, requested SD = {requested_sd:.4f} ---")
    print(f"  derived p' = requested_sd^2 / (1 + requested_sd^2) = {p_prime_analytic:.6f}")
    print(f"  realized zero fraction (empirical p')             = {zero_frac:.6f}  "
          f"{'OK' if ok_pprime else 'FAIL'}")
    print(f"  realized multiplier mean = {realized_mean:.5f}  (target 1.00000)  "
          f"{'OK' if ok_mean else 'FAIL'}")
    print(f"  realized multiplier SD   = {realized_sd:.5f}  (target {requested_sd:.5f})  "
          f"{'OK' if ok_sd else 'FAIL'}")

    # reference: the p = 0.2 Bernoulli(1-p) multiplier W-4 used for chandrop
    torch.manual_seed(42)
    cd = augment_batch(torch.ones(N, C, T).clone(), mode="chandrop", sigma=0.1,
                       chandrop_p=0.2, mask_frac=0.15)[:, :, 0]
    print(f"  [ref] chandrop p=0.2 multiplier mean = {cd.mean().item():.4f} "
          f"SD = {cd.std(unbiased=False).item():.4f}  (mean W-4 left unmatched)")
    return ok_mean and ok_sd and ok_pprime


def main() -> int:
    sds = [float(a) for a in sys.argv[1:]] or [0.40]
    all_ok = all(check(sd) for sd in sds)
    print("\nP-6 MULTIPLIER GATE: PASS" if all_ok else
          "\nP-6 MULTIPLIER GATE: FAIL -- do not launch the run")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
