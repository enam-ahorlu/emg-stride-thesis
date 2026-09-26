# run_deep_coral_align_loso.py
# ---------------------------------------------------------------------------
# D2c (19 September 2026): does the Deep CORAL weight actually change invariance?
#
# The D2 sweep (results_deep_coral_lam*) logged per-fold macro-F1 only. F1 was flat
# from lambda 0.1 to 100, but nothing recorded whether the penultimate features became
# more subject-invariant as lambda rose, so the sweep cannot be read as a test of the
# thesis's through-line ("more invariance is not better"). This script re-runs the
# identical Deep CORAL training (same harness, same defaults, same seed, same
# augmentation) and, after each fold is trained, measures alignment on the penultimate
# features. Training numerics are unchanged from run_deep_coral_cnn_loso.py: the loop is
# copied line for line, with logging added and no extra RNG draws inside it.
#
# Per-fold measurements (model in eval mode, after early stopping has restored the best
# state). Source = the fold's training subjects (not its validation subjects); target =
# the held-out subject. Labels of the target are used ONLY for the class probe, after
# training, as a measurement; they never touch training or the reported F1.
#   feat_norm_src / feat_norm_tgt  mean L2 norm of the embedding (detects shrinkage, a
#                                  trivial way to reduce an unnormalised CORAL loss)
#   coral_scaled                   the training objective itself (Sun & Saenko scaling)
#   coral_rel                      ||Cs - Ct||_F / mean(||Cs||_F, ||Ct||_F): scale-free
#   mean_gap_rel                   ||mu_s - mu_t|| / sqrt(trace of pooled covariance)
#   mmd2_rbf                       unbiased MMD^2, RBF kernel, median-heuristic bandwidth,
#                                  on features standardised by the pooled mean and SD
#   domain_probe_bacc              5-fold CV balanced accuracy of a logistic regression
#                                  separating source from target (balanced classes,
#                                  standardised). 0.5 = indistinguishable. PRIMARY.
#   class_probe_tgt_bacc           5-fold CV balanced accuracy of a logistic regression
#                                  predicting movement from the target's embedding. PRIMARY.
#   class_probe_src_bacc           the same on the source subsample, for reference
# Per-epoch training log: mean cross-entropy, mean unweighted CORAL loss, source val loss.
#
# Example (one lambda per output directory; never share a directory between lambdas):
#   python run_deep_coral_align_loso.py \
#       --npz  windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
#       --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --arch resnet_se --augmentation chandrop --coral-lambda 0.1 --epochs 40 \
#       --out results_deep_coral_align_lam0p1 --resume
#
# D2e (20 September 2026): --target-pass {train,eval,none}, default train (bit-identical to
# the behaviour above). D2c Stage B found lambda=0 (target batches still forwarded every step
# in model.train()) 5.8 pp above the no-adaptation global-norm+CD reference; this flag isolates
# whether that lift comes from BatchNorm running stats absorbing the held-out subject rather
# than from the CORAL loss itself.
#   none: no target DataLoader is built, no target batch is forwarded, the CORAL term is not
#         computed. Only legal with --coral-lambda 0. The RNG stream differs from train because
#         the target loader's shuffle no longer draws.
#   eval: the target batch is forwarded with model.eval() (BatchNorm uses, does not update, its
#         running statistics; dropout off), then model.train() is restored before the backward
#         pass. Gradients still flow through the CORAL term (no no_grad).
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_cnn_loso import (
    WindowsDataset, compute_train_norm, apply_norm, choose_val_subjects,
    class_weights_from_y, evaluate, normalize_label_to_str, LABELS, augment_batch,
)
from cnn_architectures import build_model, count_params
from run_deep_coral_cnn_loso import logits_feat, coral_loss

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


def train_deep_coral_logged(model, Xs_tr, ys_tr, Xs_va, ys_va, Xt, device,
                            epochs, batch, lr, patience, lam, seed,
                            aug_mode="none", aug_sigma=0.1, aug_chandrop_p=0.2, aug_timemask_frac=0.15,
                            target_pass="train", coral_normalize="none", grad_clip=None):
    """Identical to run_deep_coral_cnn_loso.train_deep_coral, plus per-epoch logging.
    target_pass="train" (default) is bit-identical to the original: that branch below is the
    original single line, untouched, in its original position. "eval" and "none" are D2e
    additions (see module docstring); they do not alter the "train" code path or its RNG draws.

    coral_normalize="none" (default, KC-D0.5) is bit-identical to the original: the coral_loss
    call is the original single line, untouched. "l2" L2-normalizes fs/ft (to unit norm, per
    row) before computing coral_loss, so the loss cannot be lowered simply by shrinking the
    embedding (the KC-D6 scale-free CORAL / SFC knob). It touches no RNG draws either way --
    a deterministic transform of already-computed tensors."""
    if target_pass == "none" and lam != 0:
        raise ValueError("target_pass='none' is only legal with lam == 0 (--coral-lambda 0)")
    torch.manual_seed(seed)
    w = class_weights_from_y(ys_tr, len(LABELS)).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    src = DataLoader(WindowsDataset(Xs_tr, ys_tr), batch_size=batch, shuffle=True, drop_last=True)
    if target_pass != "none":
        tgt = DataLoader(TensorDataset(torch.from_numpy(Xt.astype(np.float32))),
                         batch_size=batch, shuffle=True, drop_last=True)
    va = DataLoader(WindowsDataset(Xs_va, ys_va), batch_size=batch, shuffle=False)
    best, best_state, bad = float("inf"), None, 0
    log = []
    for ep in range(epochs):
        model.train()
        if target_pass != "none":
            tgt_it = iter(tgt)
        ce_sum, cl_sum, nb = 0.0, 0.0, 0
        nd_max = 0.0          # l2 only: the largest | ||embedding|| - 1 | the CORAL term saw this epoch
        for Xb, yb in src:
            if target_pass == "none":
                # no target DataLoader, no target forward, no CORAL term computed.
                Xb, yb = Xb.to(device), yb.to(device)
                if aug_mode != "none":
                    Xb = augment_batch(Xb, mode=aug_mode, sigma=aug_sigma,
                                        chandrop_p=aug_chandrop_p, mask_frac=aug_timemask_frac)
                opt.zero_grad(set_to_none=True)
                log_s, fs = logits_feat(model, Xb)
                ce = crit(log_s, yb)
                cl = torch.zeros((), device=device)
                loss = ce + lam * cl
                loss.backward(); opt.step()
                ce_sum += float(ce.detach()); cl_sum += float(cl.detach()); nb += 1
                continue
            try:
                (Xtb,) = next(tgt_it)
            except StopIteration:
                tgt_it = iter(tgt); (Xtb,) = next(tgt_it)
            Xb, yb, Xtb = Xb.to(device), yb.to(device), Xtb.to(device)
            if aug_mode != "none":
                Xb = augment_batch(Xb, mode=aug_mode, sigma=aug_sigma,
                                    chandrop_p=aug_chandrop_p, mask_frac=aug_timemask_frac)
            opt.zero_grad(set_to_none=True)
            log_s, fs = logits_feat(model, Xb)
            if target_pass == "eval":
                model.eval()
                _, ft = logits_feat(model, Xtb)
                model.train()
            else:  # target_pass == "train" -- bit-identical to the original: same single line,
                   # same position, no reordering relative to the pre-D2e script.
                _, ft = logits_feat(model, Xtb)
            ce = crit(log_s, yb)
            if coral_normalize == "l2":
                fs_n = fs / (fs.norm(dim=1, keepdim=True) + 1e-8)
                ft_n = ft / (ft.norm(dim=1, keepdim=True) + 1e-8)
                cl = coral_loss(fs_n, ft_n)
                # KC-D6 ruling (26 Sept): 'flat by construction' is a property of the normalised embedding the loss sees
                # and it is CHECKED, not assumed. Measurement only; no RNG draw, no effect on the loss or gradients.
                nd_max = max(nd_max, float((fs_n.detach().norm(dim=1) - 1).abs().max()),
                             float((ft_n.detach().norm(dim=1) - 1).abs().max()))
            else:  # "none" -- bit-identical to the original single line
                cl = coral_loss(fs, ft)
            loss = ce + lam * cl
            loss.backward()
            if grad_clip:      # KC-D6 divergence retry (plan 6.3): only when asked for; absent, the line above is the original
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ce_sum += float(ce.detach()); cl_sum += float(cl.detach()); nb += 1
        sched.step()
        vloss, _, _ = evaluate(model, va, device)
        log.append({"epoch": ep + 1, "train_ce": ce_sum / max(nb, 1),
                    "train_coral": cl_sum / max(nb, 1), "val_loss": float(vloss)})
        if coral_normalize == "l2":
            log[-1]["coral_embed_normdev"] = nd_max
        if vloss + 1e-6 < best:
            best, best_state, bad = vloss, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
            log[-1]["best"] = 1
        else:
            bad += 1
            log[-1]["best"] = 0
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log


@torch.no_grad()
def embed(model, X, device, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs].astype(np.float32)).to(device)
        _, f = logits_feat(model, xb)
        out.append(f.detach().cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def _cov(F):
    F = F - F.mean(0, keepdims=True)
    return F.T @ F / (len(F) - 1)


def mmd2_unbiased_rbf(A, B, rng, n=1500):
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    Z = np.vstack([A, B])
    sq = (Z ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    med = np.median(D[np.triu_indices_from(D, 1)])
    g = 1.0 / med if med > 0 else 1.0
    K = np.exp(-g * D)
    m, k = len(A), len(B)
    Kaa, Kbb, Kab = K[:m, :m], K[m:, m:], K[:m, m:]
    return float((Kaa.sum() - np.trace(Kaa)) / (m * (m - 1))
                 + (Kbb.sum() - np.trace(Kbb)) / (k * (k - 1)) - 2 * Kab.mean())


def cv_bacc(F, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y).astype(int)
    keep_cls = [c for c in np.unique(y) if (y == c).sum() >= 2]   # a class needs 2+ windows to be split
    m = np.isin(y, keep_cls); F, y = F[m], y[m]
    counts = np.bincount(y)
    n_splits = int(min(5, counts[counts > 0].min())) if len(y) else 0
    if n_splits < 2 or len(np.unique(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, F, y, cv=cv, scoring="balanced_accuracy").mean())


def alignment_metrics(Fs, ys, Ft, yt, seed, lam):
    rng = np.random.default_rng(seed)
    Cs, Ct = _cov(Fs), _cov(Ft)
    d = Fs.shape[1]
    fro = np.linalg.norm(Cs - Ct)
    pooled = np.vstack([Fs, Ft])
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-8
    Zs, Zt = (Fs - mu) / sd, (Ft - mu) / sd
    n = min(len(Fs), len(Ft), 3000)
    i_s = rng.choice(len(Fs), n, replace=False); i_t = rng.choice(len(Ft), n, replace=False)
    Fd = np.vstack([Fs[i_s], Ft[i_t]]); yd = np.r_[np.zeros(n, int), np.ones(n, int)]
    return {
        "coral_lambda": lam,
        "feat_dim": d,
        "n_src": len(Fs), "n_tgt": len(Ft),
        "feat_norm_src": float(np.linalg.norm(Fs, axis=1).mean()),
        "feat_norm_tgt": float(np.linalg.norm(Ft, axis=1).mean()),
        "coral_scaled": float(fro ** 2 / (4 * d * d)),
        "coral_rel": float(fro / (0.5 * (np.linalg.norm(Cs) + np.linalg.norm(Ct)) + 1e-12)),
        "mean_gap_rel": float(np.linalg.norm(Fs.mean(0) - Ft.mean(0))
                              / np.sqrt(np.trace(_cov(pooled)) + 1e-12)),
        "mmd2_rbf": mmd2_unbiased_rbf(Zs, Zt, rng),
        "domain_probe_bacc": cv_bacc(Fd, yd, seed),
        "class_probe_tgt_bacc": cv_bacc(Ft, yt, seed),
        "class_probe_src_bacc": cv_bacc(Fs, ys, seed),
    }


def main():
    ap = argparse.ArgumentParser("Deep CORAL under LOSO, with alignment logged on the embedding (D2c).")
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="resnet_se", choices=["simple", "resnet", "resnet_se"])
    ap.add_argument("--coral-lambda", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augmentation", "--augment", dest="augmentation", default="none",
                    choices=["none", "gaussian", "chandrop", "timemask", "combined"])
    ap.add_argument("--aug-sigma", type=float, default=0.1)
    ap.add_argument("--aug-chandrop-p", type=float, default=0.2)
    ap.add_argument("--aug-timemask-frac", type=float, default=0.15)
    ap.add_argument("--n-src-embed", type=int, default=4000,
                    help="Source windows sampled (stratified by class) for the alignment measurements.")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="KC-D6 divergence retry (plan 6.3): clip the gradient norm (5.0 for the retry). Absent (default): no clipping.")
    ap.add_argument("--coral-normalize", default="none", choices=["none", "l2"],
                    help="KC-D0.5/D6 SFC. none (default) is bit-identical to the pre-KC23 "
                         "script: coral_loss(fs, ft) on the raw embeddings. l2 L2-normalizes "
                         "fs/ft to unit norm before coral_loss, removing the shrink-the-"
                         "embedding shortcut that made the unnormalized CORAL loss cheap to "
                         "lower without moving invariance.")
    ap.add_argument("--target-pass", default="train", choices=["train", "eval", "none"],
                    help="D2e: how the target batch is forwarded during training. train (default) is "
                         "bit-identical to the pre-D2e script. eval forwards it under model.eval() "
                         "(BatchNorm running stats unaffected) then restores model.train() before the "
                         "backward pass. none builds no target loader, forwards no target batch, and "
                         "computes no CORAL term; only legal with --coral-lambda 0.")
    ap.add_argument("--heldout", type=int, default=None); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--instrument", default=None,
                    help="KC-D0.3 for KC-D6 SFC (added 26 September 2026): after each fold, write the D0.3 "
                         "measurements (occlusion, attenuation, permutation reliance and the unseen-subject "
                         "embedding probes) into this directory. Absent (default): none of this code runs and "
                         "every output is byte-identical to the pre-change script.")
    ap.add_argument("--probe-cap", type=int, default=100, help="Windows per subject and class for the probes.")
    args = ap.parse_args()

    if args.target_pass == "none" and args.coral_lambda != 0:
        raise SystemExit("--target-pass none is only legal with --coral-lambda 0")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    from run_config_dump import dump_run_config       # plan Section 0 rule 3; a new file only, no existing output changes
    dump_run_config(out_dir, args, resolved_paths={"npz": args.npz, "meta": args.meta})
    csv_path = out_dir / "deep_coral_subjectwise.csv"
    al_path = out_dir / "alignment_subjectwise.csv"
    log_path = out_dir / "training_log.csv"

    meta = pd.read_csv(args.meta); data = np.load(args.npz)
    X = data[args.xkey].astype(np.float32)
    y = np.array([LABEL_TO_IDX[s] for s in meta[args.label_col].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    in_ch = X.shape[1]
    subjects_u = sorted(np.unique(subjects).tolist())
    if args.heldout is not None:
        subjects_u = [args.heldout]
    print(f"[Deep CORAL + alignment] arch={args.arch} ({count_params(build_model(args.arch, in_ch, len(LABELS))):,} params) "
          f"lambda={args.coral_lambda} aug={args.augmentation} target_pass={args.target_pass} device={device}", flush=True)
    if args.target_pass == "none":
        print("[note] --target-pass none: RNG stream differs from train, because the target "
              "loader's shuffle no longer draws.", flush=True)

    done = set()
    if args.resume and csv_path.exists() and al_path.exists():
        done = (set(pd.read_csv(csv_path)["subject"].astype(int))
                & set(pd.read_csv(al_path)["subject"].astype(int)))

    from sklearn.metrics import f1_score, balanced_accuracy_score
    for heldout in subjects_u:
        if heldout in done:
            continue
        te = (subjects == heldout); tr = ~te
        mean, std = compute_train_norm(X[tr])
        Xtr_all = apply_norm(X[tr], mean, std); Xte = apply_norm(X[te], mean, std)
        ytr_all, subtr = y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, args.val_frac, args.seed + heldout)
        m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)
        model = build_model(args.arch, in_ch, len(LABELS)).to(device)
        model, log = train_deep_coral_logged(
            model, Xtr_all[m_tr], ytr_all[m_tr], Xtr_all[m_va], ytr_all[m_va],
            Xte, device, args.epochs, args.batch, args.lr, args.patience,
            args.coral_lambda, args.seed,
            aug_mode=args.augmentation, aug_sigma=args.aug_sigma,
            aug_chandrop_p=args.aug_chandrop_p, aug_timemask_frac=args.aug_timemask_frac,
            target_pass=args.target_pass, coral_normalize=args.coral_normalize, grad_clip=args.grad_clip)

        te_dl = DataLoader(WindowsDataset(Xte, y[te]), batch_size=512, shuffle=False)
        _, yt, yp = evaluate(model, te_dl, device)
        row = {"subject": int(heldout), "arch": args.arch, "coral_lambda": args.coral_lambda,
               "target_pass": args.target_pass, "coral_normalize": args.coral_normalize,
               "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
               "bal_acc": float(balanced_accuracy_score(yt, yp))}

        # ---- measurement only, after training; separate numpy RNG, no torch RNG draws ----
        rng = np.random.default_rng(args.seed + 1000 + heldout)
        Xs_pool, ys_pool = Xtr_all[m_tr], ytr_all[m_tr]
        idx = []
        per = max(1, args.n_src_embed // len(LABELS))
        for c in range(len(LABELS)):
            ic = np.flatnonzero(ys_pool == c)
            if len(ic):
                idx.append(rng.choice(ic, min(per, len(ic)), replace=False))
        idx = np.concatenate(idx)
        Fs = embed(model, Xs_pool[idx], device); Ft = embed(model, Xte, device)
        al = {"subject": int(heldout), "target_pass": args.target_pass, "epochs_run": len(log),
              "best_epoch": int(max([e["epoch"] for e in log if e["best"] == 1] or [0]))}
        al.update(alignment_metrics(Fs, ys_pool[idx], Ft, y[te], args.seed + heldout, args.coral_lambda))
        if args.coral_normalize == "l2":
            al["coral_embed_normdev_max"] = float(max(e["coral_embed_normdev"] for e in log))

        if args.instrument:
            from run_cnn_arch_loso import instrument_fold      # imported only when asked for: the default path is untouched
            instrument_fold(model, Xte, y[te], int(heldout), args.arch, args.instrument, device, row["f1_macro"],
                            seed=args.seed, Xva=Xtr_all[m_va], yva=ytr_all[m_va], subj_va=subtr[m_va],
                            probe_cap=args.probe_cap)

        pd.DataFrame([dict(e, subject=int(heldout), coral_lambda=args.coral_lambda) for e in log]).to_csv(
            log_path, mode="a", header=not log_path.exists(), index=False)
        pd.DataFrame([al]).to_csv(al_path, mode="a", header=not al_path.exists(), index=False)
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done.add(heldout)
        print(f"[fold] Sub{heldout:02d} f1={row['f1_macro']:.4f} domain_probe={al['domain_probe_bacc']:.3f} "
              f"class_probe_tgt={al['class_probe_tgt_bacc']:.3f} coral_rel={al['coral_rel']:.3f}", flush=True)

    df = pd.read_csv(csv_path).drop_duplicates("subject")
    A = pd.read_csv(al_path).drop_duplicates("subject")
    summ = {"method": "DeepCORAL+align", "arch": args.arch, "coral_lambda": args.coral_lambda,
            "f1_macro_mean": df["f1_macro"].mean(), "n": len(df)}
    for c in ["domain_probe_bacc", "class_probe_tgt_bacc", "class_probe_src_bacc", "coral_rel",
              "coral_scaled", "mean_gap_rel", "mmd2_rbf", "feat_norm_src", "feat_norm_tgt"]:
        summ[c + "_mean"] = A[c].mean()
    pd.DataFrame([summ]).to_csv(out_dir / "alignment_summary.csv", index=False)
    print("\n" + "\n".join(f"  {k}: {v}" for k, v in summ.items()))


if __name__ == "__main__":
    main()
