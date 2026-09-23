#!/usr/bin/env python3
"""
kc23_d0_inertness.py
=====================
EXPERIMENT_PLAN_KC23_DEEP.md KC-D0.4 -- the two mandatory inertness
assertions, comparing the before/after captures written by
kc23_d0_capture_before.py.

  1. For every EXISTING augmentation mode in both scripts: identical output
     bytes and identical post-call torch RNG state, before and after Change 1
     (the new chanoffset/globalgain modes).
  2. The smoke run after Change 2 (instrumentation) reproduces the
     before-capture's occlusion.csv, attenuation.csv and se_gates.csv byte for
     byte, the same final F1, and the same post-run torch RNG state.

GPU note. `cudnn.benchmark = True` (train_cnn_loso.py module level) makes
convolution-algorithm SELECTION non-deterministic ACROSS PROCESSES on this
machine: two separate `python ... --device cuda` invocations of the IDENTICAL
pre-D0 code gave different F1 (0.5449 vs 0.5454) and different
occlusion/attenuation/se_gates hashes, while two trainings inside the SAME
process gave identical F1. This is exactly the failure mode the plan
anticipated ("If cuDNN nondeterminism makes even the before-capture
unrepeatable, run the smoke on CPU for this assertion and say so"). The smoke
assertion below is therefore run on CPU (`before_cpu` / `after_cpu`), which
gives byte-identical results.

Numpy RNG caveat. augment_batch makes zero numpy RNG calls (only
torch.randn/torch.rand/torch.randint), so the numpy component of the
"post-call RNG state" is not something augment_batch could perturb either
way; it differs between process runs purely because kc23_d0_capture_before.py
never seeds the global numpy RNG before the capture loop (OS-entropy seeded
at interpreter start, per process). Only the torch CPU RNG state is a
meaningful inertness signal for augment_batch; it is checked explicitly below
and the numpy component is reported but not gated on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(d):
    d = Path(d)
    return {
        "augment": json.loads((d / "augment_modes_capture.json").read_text()),
        "smoke": json.loads((d / "smoke_result.json").read_text()) if (d / "smoke_result.json").exists() else None,
        "occlusion": (d / "instrument" / "occlusion.csv").read_bytes() if (d / "instrument" / "occlusion.csv").exists() else None,
        "attenuation": (d / "instrument" / "attenuation.csv").read_bytes() if (d / "instrument" / "attenuation.csv").exists() else None,
        "se_gates": (d / "instrument" / "se_gates.csv").read_bytes() if (d / "instrument" / "se_gates.csv").exists() else None,
    }


def check_augment_modes(before, after):
    """Assertion 1: every EXISTING mode, both entry points, identical output
    bytes and identical torch-CPU RNG state."""
    existing_keys = [k for k in before if not any(k.endswith(f"::{m}") for m in ("chanoffset", "globalgain"))]
    fails = []
    for k in existing_keys:
        if k not in after:
            fails.append((k, "missing from after-capture"))
            continue
        if before[k]["output_sha256"] != after[k]["output_sha256"]:
            fails.append((k, "output tensor bytes differ"))
        if before[k]["rng_after"]["torch_cpu"] != after[k]["rng_after"]["torch_cpu"]:
            fails.append((k, "torch CPU RNG state differs"))
    ok = len(fails) == 0
    print(f"[assertion 1] {len(existing_keys)} (entry, mode) pairs checked "
          f"(existing modes only, {len(before) - len(existing_keys)} new modes excluded): "
          f"{'PASS' if ok else 'FAIL'}")
    for k, why in fails:
        print(f"    FAIL: {k}: {why}")
    return ok


def check_smoke(before, after):
    """Assertion 2: the CPU smoke run reproduces occlusion/attenuation/
    se_gates byte for byte, the same F1, and the same torch CPU RNG state."""
    fails = []
    for name in ("occlusion", "attenuation", "se_gates"):
        if before[name] is None or after[name] is None:
            fails.append((name, "missing"))
        elif before[name] != after[name]:
            fails.append((name, "bytes differ"))
    if before["smoke"]["f1_macro"] != after["smoke"]["f1_macro"]:
        fails.append(("f1_macro", f"{before['smoke']['f1_macro']} != {after['smoke']['f1_macro']}"))
    if before["smoke"]["rng_after"]["torch_cpu"] != after["smoke"]["rng_after"]["torch_cpu"]:
        fails.append(("rng_after.torch_cpu", "differs"))
    ok = len(fails) == 0
    print(f"[assertion 2] CPU smoke fold reproduction: {'PASS' if ok else 'FAIL'}")
    for name, why in fails:
        print(f"    FAIL: {name}: {why}")
    if ok:
        print(f"    f1_macro (both) = {before['smoke']['f1_macro']:.6f}")
    return ok


def main():
    before_dir = ROOT / "results_kc23_d0_capture" / "before_cpu"
    after_dir = ROOT / "results_kc23_d0_capture" / "after_cpu"
    if not before_dir.exists() or not after_dir.exists():
        sys.exit(f"[kc23-d0-inertness] missing captures: run kc23_d0_capture_before.py "
                 f"--out {before_dir} --device cpu (before the D0 code changes) and "
                 f"--out {after_dir} --device cpu (after) first.")

    before = load(before_dir)
    after = load(after_dir)

    ok1 = check_augment_modes(before["augment"], after["augment"])
    ok2 = check_smoke(before, after)

    overall = ok1 and ok2
    print(f"\n[KC-D0.4 VERDICT] {'PASS -- both assertions hold, D0.2/D0.3 are inert on every existing path' if overall else 'FAIL -- STOP, per the plan'}")
    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
