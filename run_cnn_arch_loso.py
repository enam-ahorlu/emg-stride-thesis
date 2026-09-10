# run_cnn_arch_loso.py
# ---------------------------------------------------------------------------
# Fair-deep-baseline experiment: run a stronger CNN architecture under the
# IDENTICAL LOSO + per-subject-normalisation protocol as the headline
# SimpleEMGCNN, so the "classical vs deep under LOSO" comparison is not confounded
# by architectural under-investment. Only the architecture changes.
#
#   --arch simple     : the original SimpleEMGCNN (reproduces 0.754; sanity check)
#   --arch resnet_se  : compact 1D ResNet + squeeze-excite attention (fairer deep baseline)
#   --arch resnet     : same without SE (ablation on the attention)
#
# Reuses every data/normalisation/training convention from train_cnn_loso.py
# (per_subject_zscore_3d, choose_val_subjects, class weights, early stopping),
# so results are directly comparable to Table 4.2 / Section 4.2.2.
#
# Example (full run, GPU):
#   python run_cnn_arch_loso.py \
#       --npz  windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
#       --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --arch resnet_se --epochs 40 --out results_cnn_loso_resnet_se --resume
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_cnn_loso import (
    WindowsDataset, per_subject_zscore_3d, choose_val_subjects,
    class_weights_from_y, evaluate, normalize_label_to_str, LABELS,
    compute_train_norm, apply_norm, augment_batch,
)
from cnn_architectures import build_model, count_params, SEBlock1d

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


def _cnn_infer(model, X, device, batch=512):
    """Deterministic forward pass, identical to evaluate_with_proba's, returning
    argmax predictions. Used only by the instrumentation."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            Xb = torch.from_numpy(X[i:i + batch]).to(device, non_blocking=True)
            preds.append(model(Xb).argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def instrument_fold(model, Xte, yte, subject, arch, instr_dir, device, row_f1):
    """W-2 Stage G0. After a fold trains and before the model is discarded,
    on the held-out subject's windows only:
      - channel-occlusion sensitivity  -> {instr_dir}/occlusion.csv
      - SE gate activations (SE archs) -> {instr_dir}/se_gates.csv
    RNG state is snapshotted and restored so nothing downstream shifts; the
    driver is bit-identical with the flag absent, and the reported f1_macro is
    already written before this runs.
    """
    from sklearn.metrics import f1_score
    instr_dir = Path(instr_dir); instr_dir.mkdir(parents=True, exist_ok=True)
    t_state = torch.get_rng_state()
    c_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    model.eval()
    hooks = []
    try:
        # ---- SE gate capture (only if the body carries SEBlock1d) ------------
        se_blocks, captured = [], []
        body = getattr(model, "body", None)
        if body is not None:
            for bi, blk in enumerate(body):
                se = getattr(blk, "se", None)
                if isinstance(se, SEBlock1d):
                    idx = len(se_blocks)
                    se_blocks.append((bi, se.fc2.out_features))
                    captured.append([])
                    hooks.append(se.fc2.register_forward_hook(
                        lambda m, inp, out, k=idx: captured[k].append(
                            torch.sigmoid(out).detach().float().cpu())))

        # ---- unoccluded pass (fills the SE captures) ------------------------
        yp_full = _cnn_infer(model, Xte, device)
        f1_full = float(f1_score(yte, yp_full, average="macro", zero_division=0))
        if abs(f1_full - row_f1) > 1e-9:
            print(f"[instrument] WARNING Sub{subject:02d}: recomputed f1 {f1_full:.6f} "
                  f"!= reported {row_f1:.6f}", flush=True)
        for h in hooks:
            h.remove()
        hooks = []

        if se_blocks:
            se_rows = []
            for k, (bi, width) in enumerate(se_blocks):
                g = torch.cat(captured[k], dim=0)          # (n_windows, width)
                gm = g.mean(dim=0).numpy(); gs = g.std(dim=0, unbiased=True).numpy()
                for fc in range(width):
                    se_rows.append({"subject": subject, "block": bi, "block_width": width,
                                    "feature_channel": fc,
                                    "gate_mean": float(gm[fc]), "gate_sd": float(gs[fc])})
            _append_rows(instr_dir / "se_gates.csv", se_rows)

        # ---- channel-occlusion sensitivity --------------------------------
        occ_rows = []
        for ch in range(Xte.shape[1]):
            Xo = Xte.copy()
            Xo[:, ch, :] = 0.0
            yp_o = _cnn_infer(model, Xo, device)
            f1_o = float(f1_score(yte, yp_o, average="macro", zero_division=0))
            occ_rows.append({"subject": subject, "channel": ch, "f1_full": f1_full,
                             "f1_occluded": f1_o, "drop_pp": (f1_full - f1_o) * 100.0})
        _append_rows(instr_dir / "occlusion.csv", occ_rows)

        # ---- P-9: graded channel attenuation -----------------------------
        # Multiply one channel by alpha in {1, 0.75, 0.5, 0.25, 0}. alpha = 0 is
        # exactly the occlusion pass above (Xo[:, ch, :] = 0), so the alpha = 0
        # slice of attenuation.csv reproduces occlusion.csv row for row within
        # the same run. Inference only, inside the same RNG snapshot/restore.
        atten_rows = []
        for ch in range(Xte.shape[1]):
            for alpha in (1.0, 0.75, 0.5, 0.25, 0.0):
                if alpha == 1.0:
                    f1_a = f1_full
                else:
                    Xa = Xte.copy()
                    Xa[:, ch, :] = Xte[:, ch, :] * alpha
                    yp_a = _cnn_infer(model, Xa, device)
                    f1_a = float(f1_score(yte, yp_a, average="macro", zero_division=0))
                atten_rows.append({"subject": subject, "channel": ch, "alpha": alpha,
                                   "f1": f1_a, "drop_pp": (f1_full - f1_a) * 100.0})
        _append_rows(instr_dir / "attenuation.csv", atten_rows)

        print(f"[instrument] Sub{subject:02d}: occlusion 9 ch, attenuation 9x5, "
              f"SE gates {'yes' if se_blocks else 'n/a'}", flush=True)
    finally:
        for h in hooks:
            h.remove()
        torch.set_rng_state(t_state)
        if c_state is not None:
            torch.cuda.set_rng_state_all(c_state)
        np.random.set_state(np_state)


def _append_rows(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def evaluate_with_proba(model, loader, device):
    """Like train_cnn_loso.evaluate() but also returns per-window softmax
    probabilities, in the model's output-index order (== LABELS order, since
    y was encoded via LABEL_TO_IDX over LABELS)."""
    model.eval()
    ys, yhat, probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            logits = model(Xb)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            pred = p.argmax(1)
            ys.append(yb.numpy())
            yhat.append(pred)
            probs.append(p)
    y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(yhat) if yhat else np.array([], dtype=int)
    proba = np.concatenate(probs) if probs else np.zeros((0, len(LABELS)))
    return y_true, y_pred, proba


def train_fold(model, X_tr, y_tr, X_va, y_va, device, epochs, batch, lr, patience, seed,
                aug_mode="none", aug_sigma=0.1, aug_chandrop_p=0.2, aug_timemask_frac=0.15,
                aug_gain_sd=0.4):
    torch.manual_seed(seed)
    w = class_weights_from_y(y_tr, len(LABELS)).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr = DataLoader(WindowsDataset(X_tr, y_tr), batch_size=batch, shuffle=True)
    va = DataLoader(WindowsDataset(X_va, y_va), batch_size=batch, shuffle=False)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for Xb, yb in tr:
            Xb, yb = Xb.to(device), yb.to(device)
            if aug_mode != "none":
                # training batches only (identical to train_cnn_loso.py's SimpleEMGCNN augmentation)
                Xb = augment_batch(Xb, mode=aug_mode, sigma=aug_sigma,
                                    chandrop_p=aug_chandrop_p, mask_frac=aug_timemask_frac,
                                    gain_sd=aug_gain_sd)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xb), yb)
            loss.backward(); opt.step()
        sched.step()
        vloss, _, _ = evaluate(model, va, device)
        if vloss + 1e-6 < best:
            best, best_state, bad = vloss, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser("Stronger-architecture CNN under LOSO (per-subject norm).")
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="resnet_se",
                    choices=["simple", "resnet", "resnet_se", "resnet_nores"])
    ap.add_argument("--widths", default=None,
                    help="W-5: comma list overriding EMGResNet1D stage widths, e.g. 48,96,192. "
                         "Absent = the built-in (32,64,128).")
    ap.add_argument("--blocks-per-stage", type=int, default=None,
                    help="W-5: override EMGResNet1D blocks per stage. Absent = the built-in 2.")
    ap.add_argument("--norm-mode", default="per_subject", choices=["per_subject", "global"],
                    help="per_subject: headline transductive z-score, applied once upfront. "
                         "global: train-fold-only z-score, recomputed per LOSO fold (leak-free).")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--augmentation", "--augment", dest="augmentation", default="none",
                    choices=["none", "gaussian", "chandrop", "timemask", "combined",
                             "gainjitter", "subset", "mpchandrop"],
                    help="Data augmentation applied to training batches only (identical "
                         "transforms/params to train_cnn_loso.py's --augment). subset=P-5 "
                         "fixed 7-of-9 channel-subset vocabulary; mpchandrop=P-6 mean-preserving "
                         "inverted channel dropout with SD from --aug-gain-sd.")
    ap.add_argument("--aug-sigma", type=float, default=0.1,
                    help="Gaussian noise std relative to normalized data scale (default: 0.1)")
    ap.add_argument("--aug-chandrop-p", type=float, default=0.2,
                    help="Per-channel drop probability for chandrop augmentation (default: 0.2)")
    ap.add_argument("--aug-timemask-frac", type=float, default=0.15,
                    help="Fraction of T to zero out for timemask augmentation (default: 0.15)")
    ap.add_argument("--aug-gain-sd", type=float, default=0.4,
                    help="Per-channel multiplicative gain SD for gainjitter augmentation (W-4, default: 0.4)")
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heldout", type=int, default=None, help="Run only this held-out subject (smoke test).")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--out", default="results_cnn_loso_resnet_se")
    ap.add_argument("--save-proba", default=None,
                    help="If set, dir to save per-window softmax probabilities to, as "
                         "{save-proba}/{model-tag}_sub{K:02d}.npz (keys: proba [n,4] in "
                         "LABELS order, y_true [n]). Requires --model-tag.")
    ap.add_argument("--model-tag", default=None,
                    help="Tag used in the saved proba filename (e.g. CNN, RESNET_SE).")
    ap.add_argument("--instrument", default=None,
                    help="W-2 Stage G0. If set, dir for per-fold channel-occlusion "
                         "sensitivity (occlusion.csv) and SE gate activations "
                         "(se_gates.csv), computed on the held-out subject's windows "
                         "after training. RNG-neutral; absent = no behaviour change.")
    args = ap.parse_args()
    if args.save_proba and not args.model_tag:
        ap.error("--save-proba requires --model-tag")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cnn_arch_subjectwise.csv"

    # B1 section 1.5: provenance. Guarded, additive, RNG-inert (asserted inside).
    from run_config_dump import dump_run_config
    dump_run_config(out_dir, args, resolved_paths={"npz": args.npz, "meta": args.meta})

    meta = pd.read_csv(args.meta); data = np.load(args.npz)
    X = data[args.xkey].astype(np.float32)
    y = np.array([LABEL_TO_IDX[s] for s in meta[args.label_col].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    if args.norm_mode == "per_subject":
        X = per_subject_zscore_3d(X, subjects)      # headline normalisation, leak-free (own stats only)
    in_ch = X.shape[1]
    # W-5 geometry overrides: kwargs are None unless the flags were passed, so
    # build_model() is called exactly as before for every existing invocation.
    _geo = dict(widths=[int(w) for w in args.widths.split(",")] if args.widths else None,
                blocks_per_stage=args.blocks_per_stage)
    subjects_u = sorted(np.unique(subjects).tolist())
    if args.heldout is not None:
        subjects_u = [args.heldout]

    print(f"[arch] {args.arch}: {count_params(build_model(args.arch, in_ch, len(LABELS), **_geo)):,} params | device={device}", flush=True)
    print(f"[aug] augmentation={args.augmentation}, sigma={args.aug_sigma}, "
          f"chandrop_p={args.aug_chandrop_p}, timemask_frac={args.aug_timemask_frac}, "
          f"gain_sd={args.aug_gain_sd}", flush=True)
    done = set()
    if args.resume and csv_path.exists():
        done = set(pd.read_csv(csv_path)["subject"].astype(int).tolist())

    if args.save_proba:
        proba_dir = Path(args.save_proba); proba_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import f1_score, balanced_accuracy_score
    for heldout in subjects_u:
        if heldout in done:
            continue
        te = (subjects == heldout); tr = ~te
        if args.norm_mode == "global":
            # train-fold-only stats (leak-free); recomputed per fold since the
            # training-subject set changes each time
            mean, std = compute_train_norm(X[tr])
            Xtr_full = apply_norm(X[tr], mean, std)
            Xte_fold = apply_norm(X[te], mean, std)
        else:
            Xtr_full = X[tr]
            Xte_fold = X[te]
        ytr_full, subtr = y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, args.val_frac, args.seed + heldout)
        m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)
        model = build_model(args.arch, in_ch, len(LABELS), **_geo).to(device)
        model = train_fold(model, Xtr_full[m_tr], ytr_full[m_tr], Xtr_full[m_va], ytr_full[m_va],
                           device, args.epochs, args.batch, args.lr, args.patience, args.seed,
                           aug_mode=args.augmentation, aug_sigma=args.aug_sigma,
                           aug_chandrop_p=args.aug_chandrop_p, aug_timemask_frac=args.aug_timemask_frac,
                           aug_gain_sd=args.aug_gain_sd)
        te_dl = DataLoader(WindowsDataset(Xte_fold, y[te]), batch_size=512, shuffle=False)
        yt, yp, proba = evaluate_with_proba(model, te_dl, device)
        if args.save_proba:
            proba_npz = proba_dir / f"{args.model_tag}_sub{heldout:02d}.npz"
            np.savez(proba_npz, proba=proba.astype(np.float64), y_true=yt.astype(np.int32, copy=False))
            print(f"[save-proba] {args.model_tag} Sub{heldout:02d} -> {proba_npz}", flush=True)
        row = {"subject": int(heldout), "arch": args.arch,
               "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
               "bal_acc": float(balanced_accuracy_score(yt, yp))}
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done.add(heldout)
        print(f"[fold] Sub{heldout:02d} {args.arch} f1={row['f1_macro']:.4f}", flush=True)

        if args.instrument:
            # G0 inertness: the plan's "bit-identical f1 with/without --instrument"
            # check is defeated by cuDNN nondeterminism (R-1), so instead prove the
            # instrumentation is RNG-neutral -- it runs after the row is written and
            # its forward hooks return None, so RNG state is the only way it could
            # perturb a later fold. Assert it is untouched, every fold.
            _rt = torch.get_rng_state()
            _rc = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            _rn = np.random.get_state()
            instrument_fold(model, Xte_fold, y[te], int(heldout), args.arch,
                            args.instrument, device, row["f1_macro"])
            assert torch.equal(torch.get_rng_state(), _rt), "instrument perturbed torch CPU RNG"
            if _rc is not None:
                assert all(torch.equal(a, b) for a, b in
                           zip(torch.cuda.get_rng_state_all(), _rc)), "instrument perturbed CUDA RNG"
            _rn2 = np.random.get_state()
            assert _rn2[0] == _rn[0] and np.array_equal(_rn2[1], _rn[1]) and _rn2[2] == _rn[2], \
                "instrument perturbed numpy RNG"

    df = pd.read_csv(csv_path).drop_duplicates("subject")
    m, s = df["f1_macro"].mean(), df["f1_macro"].std(ddof=1)
    pd.DataFrame([{"arch": args.arch, "f1_macro_mean": round(m, 4), "f1_macro_sd": round(s, 4),
                   "n": len(df)}]).to_csv(out_dir / "cnn_arch_summary.csv", index=False)
    print(f"\n[{args.arch}] LOSO F1 = {m:.4f} ± {s:.4f} (n={len(df)}) | SimpleEMGCNN headline = 0.754")


if __name__ == "__main__":
    main()
