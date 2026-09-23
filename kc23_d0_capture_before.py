#!/usr/bin/env python3
"""
kc23_d0_capture_before.py
==========================
EXPERIMENT_PLAN_KC23_DEEP.md KC-D0.1 -- capture the "before" state, before any
D0 code change is made, so kc23_d0_inertness.py can prove the changes are
byte-identical on every existing path.

Two captures:

1. For every EXISTING augmentation mode (none, gaussian, chandrop, timemask,
   combined, gainjitter, subset, mpchandrop), through BOTH import paths
   (train_cnn_loso.augment_batch and run_cnn_arch_loso.augment_batch -- the
   same function object, since run_cnn_arch_loso.py imports it from
   train_cnn_loso.py rather than redefining it, but both entry points are
   captured separately as the plan specifies): the output tensor bytes on a
   fixed seeded batch, and the post-call torch RNG state.

2. One instrumented smoke run (--heldout 1 --epochs 2, arch resnet_se,
   chandrop): occlusion.csv, attenuation.csv, se_gates.csv, final F1, and the
   post-run RNG state (torch CPU, CUDA if present, numpy).

   This duplicates ~30 lines of run_cnn_arch_loso.py's main() fold loop
   rather than importing main() itself, because main() is a monolithic
   function with no return value and no RNG-state hook -- reimplementing the
   one-fold path here, calling the SAME underlying functions
   (build_model, train_fold, evaluate_with_proba, instrument_fold) that
   main() calls, lets this script capture RNG state at the exact point D0.4
   needs without touching run_cnn_arch_loso.py itself (which stays untouched
   until the real D0.2/D0.3 changes land).

Usage:
  python kc23_d0_capture_before.py --out results_kc23_d0_capture/before
  (then, after the D0 code changes land)
  python kc23_d0_capture_before.py --out results_kc23_d0_capture/after
  python kc23_d0_inertness.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent

AUG_MODES = ["none", "gaussian", "chandrop", "timemask", "combined",
             "gainjitter", "subset", "mpchandrop"]
AUG_SEED = 20230923          # fixed seed applied immediately before each augment_batch call
BATCH_SHAPE = (16, 9, 500)   # (N, C, T): representative, independent of any real dataset
DATA_SEED = 999              # fixed seed for constructing the synthetic input batch

DEFAULT_NPZ = ROOT / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz"
DEFAULT_META = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"


def sha256_of(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def rng_state_hash():
    t = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    c = None
    if torch.cuda.is_available():
        c = hashlib.sha256(
            b"".join(s.cpu().numpy().tobytes() for s in torch.cuda.get_rng_state_all())
        ).hexdigest()
    n_state = np.random.get_state()
    n = hashlib.sha256(n_state[1].tobytes() + str(n_state[2]).encode()).hexdigest()
    return {"torch_cpu": t, "torch_cuda": c, "numpy": n}


def capture_augment_modes(out_dir: Path, device: torch.device):
    from train_cnn_loso import augment_batch as augment_batch_via_train
    from run_cnn_arch_loso import augment_batch as augment_batch_via_arch
    assert augment_batch_via_train is augment_batch_via_arch, (
        "run_cnn_arch_loso.augment_batch is not the same object as "
        "train_cnn_loso.augment_batch -- the shared-function assumption in "
        "the D0 plan no longer holds; investigate before trusting one capture "
        "to stand in for both entry points.")

    rng = np.random.default_rng(DATA_SEED)
    Xb_np = rng.standard_normal(BATCH_SHAPE).astype(np.float32)

    results = {}
    for entry_name, fn in [("train_cnn_loso.augment_batch", augment_batch_via_train),
                            ("run_cnn_arch_loso.augment_batch", augment_batch_via_arch)]:
        for mode in AUG_MODES:
            Xb = torch.from_numpy(Xb_np.copy()).to(device)
            torch.manual_seed(AUG_SEED)
            if hasattr(fn, "_mp_gate_printed"):
                del fn._mp_gate_printed
            out = fn(Xb, mode=mode, sigma=0.1, chandrop_p=0.2, mask_frac=0.15, gain_sd=0.4)
            out_np = out.detach().cpu().numpy()
            key = f"{entry_name}::{mode}"
            results[key] = {
                "output_sha256": sha256_of(out_np),
                "output_shape": list(out_np.shape),
                "rng_after": rng_state_hash(),
            }
            np.save(out_dir / f"augbatch__{entry_name.replace('.', '_')}__{mode}.npy", out_np)
    with open(out_dir / "augment_modes_capture.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[capture] {len(results)} (entry, mode) pairs -> {out_dir / 'augment_modes_capture.json'}")
    return results


def run_smoke_fold(npz_path: Path, meta_path: Path, device: torch.device,
                    heldout: int, epochs: int, instr_dir: Path):
    """Reproduces exactly the heldout=1 fold of run_cnn_arch_loso.py's main()
    for arch=resnet_se, augmentation=chandrop, norm-mode=per_subject, the
    model-of-record configuration, calling the SAME functions main() calls."""
    from train_cnn_loso import (
        WindowsDataset, per_subject_zscore_3d, choose_val_subjects,
        normalize_label_to_str, LABELS,
    )
    from cnn_architectures import build_model
    from run_cnn_arch_loso import train_fold, evaluate_with_proba, instrument_fold
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score, balanced_accuracy_score

    LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}
    SEED = 42

    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    meta = pd.read_csv(meta_path); data = np.load(npz_path)
    X = data["X_env"].astype(np.float32)
    y = np.array([LABEL_TO_IDX[s] for s in meta["movement"].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    X = per_subject_zscore_3d(X, subjects)
    in_ch = X.shape[1]

    te = (subjects == heldout); tr = ~te
    Xtr_full, ytr_full, subtr = X[tr], y[tr], subjects[tr]
    tr_subs, va_subs = choose_val_subjects(subtr, 0.15, SEED + heldout)
    m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)

    model = build_model("resnet_se", in_ch, len(LABELS)).to(device)
    model = train_fold(model, Xtr_full[m_tr], ytr_full[m_tr], Xtr_full[m_va], ytr_full[m_va],
                       device, epochs, 512, 1e-3, 7, SEED,
                       aug_mode="chandrop", aug_sigma=0.1, aug_chandrop_p=0.2,
                       aug_timemask_frac=0.15, aug_gain_sd=0.4)

    Xte, yte = X[te], y[te]
    te_dl = DataLoader(WindowsDataset(Xte, yte), batch_size=512, shuffle=False)
    yt, yp, proba = evaluate_with_proba(model, te_dl, device)
    f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    balacc = float(balanced_accuracy_score(yt, yp))

    # KC-D0.3: pass the fold's validation-subject windows too, so a POST-change
    # capture exercises the new permutation/embed_probes instrumentation. A
    # PRE-change capture (instrument_fold without these kwargs, run via `git
    # stash` on the two production files while this harness script itself
    # stays as-is) falls back to the old call shape automatically -- the two
    # captures are compared only on occlusion/attenuation/se_gates/F1/RNG,
    # which do not depend on Xva either way.
    import inspect
    if "seed" in inspect.signature(instrument_fold).parameters:
        instrument_fold(model, Xte, yte, heldout, "resnet_se", instr_dir, device, f1,
                        seed=SEED, Xva=Xtr_full[m_va], yva=ytr_full[m_va], subj_va=subtr[m_va])
    else:
        instrument_fold(model, Xte, yte, heldout, "resnet_se", instr_dir, device, f1)

    rng_after = rng_state_hash()
    return {"f1_macro": f1, "bal_acc": balacc, "rng_after": rng_after}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="results_kc23_d0_capture/{before,after}")
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--heldout", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cuda", "cpu"])
    ap.add_argument("--skip-smoke", action="store_true",
                    help="only capture augment_batch modes, skip the GPU/CPU smoke fold")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    instr_dir = out_dir / "instrument"
    instr_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    print(f"[kc23-d0-capture] out={out_dir} device={device}")

    capture_augment_modes(out_dir, device)

    if not args.skip_smoke:
        smoke = run_smoke_fold(Path(args.npz), Path(args.meta), device,
                               args.heldout, args.epochs, instr_dir)
        with open(out_dir / "smoke_result.json", "w") as f:
            json.dump(smoke, f, indent=2)
        print(f"[capture] smoke fold f1={smoke['f1_macro']:.4f} -> {out_dir / 'smoke_result.json'}")
        for name in ["occlusion.csv", "attenuation.csv", "se_gates.csv"]:
            p = instr_dir / name
            if p.exists():
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                print(f"  {name}: sha256={h[:16]}...")

    print(f"[DONE] capture written to {out_dir}")


if __name__ == "__main__":
    main()
