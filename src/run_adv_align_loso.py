#!/usr/bin/env python3
"""
src/run_adv_align_loso.py
=======================
docs/plans/EXPERIMENT_PLAN_KC23_DEEP.md KC-D6 section 6.3 -- a learned alignment axis
whose knob provably moves invariance: subject-adversarial training (DANN,
Ganin et al. 2016), through a gradient-reversal layer, applied per subject
rather than as a single binary source/target discriminator.

New code (KC-D0.5): no inertness assertion applies (nothing existing calls
this script), but it is covered by the KC-D6 sanity gate (Section 6.5): at
lambda_max = 0, F1 must land within +/-1.5 pts of the D2e weight-0
target-pass arm (83.0%).

Built by IMPORTING, not copying, from:
  src/run_deep_coral_align_loso.py -- embed, alignment_metrics, cv_bacc, LABELS
  src/train_cnn_loso.py            -- augment_batch, choose_val_subjects,
                                   class_weights_from_y, compute_train_norm,
                                   apply_norm, WindowsDataset, evaluate,
                                   normalize_label_to_str
  src/cnn_architectures.py         -- build_model, count_params

The training loop is the D2c/D2e Deep CORAL loop's structure (same harness:
resnet_se, channel dropout 0.2, GLOBAL normalization, batch 256, epochs 40,
patience 7, target windows forwarded every step), with the CORAL term
replaced by the adversary's cross-entropy through gradient reversal.

Adversary construction (Section 6.2, family ADV):
  A 40-way-dataset subject adversary: each of the fold's TRAINING subjects
  (the ~33 subjects with labels, i.e. the 39 LOSO training subjects minus the
  ~6 validation subjects -- V3 confirms round(0.15*39)=6) gets its own class
  id, and the held-out (target) subject is treated as one additional,
  unlabeled class -- an (n_train_subjects + 1)-way classification head,
  MLP 128 -> 64 -> n_subjects, through gradient reversal. Warm-up schedule
  (Ganin): lambda(p) = lambda_max * (2 / (1 + exp(-10*p)) - 1), with p the
  fraction of total training steps completed.

  --adv-mode marginal (default, family ADV / ADV-PS): one shared adversary
    head over all windows, source subjects vs. the held-out subject.
  --adv-mode classcond (family ADV-C, mechanism test): one adversary head PER
    MOVEMENT CLASS, each fed only windows of its class. Source uses true
    labels to route windows to the right per-class head; the TARGET uses its
    TRUE labels too (oracle, diagnostic only, never deployable -- requires
    --oracle-target-labels, refused without it) to route target windows to
    the matching per-class head. Every output row carries oracle=True.
  --adv-mode cdan (family ADV-CDAN, optional, deployable): a single adversary
    head on the multilinear map of the embedding and the predicted class
    probabilities (Long et al., 2018): outer product f (x) softmax(logits),
    flattened. No target labels used. At feat_dim=128, n_classes=4 the
    flattened map is 512-d, small enough that the random-feature reduction
    CDAN uses for very high-dimensional maps is not needed here.

Divergence rule (Section 6.3, fixed now): an arm has diverged if the source
validation loss at its best epoch exceeds twice that of the same seed's
lambda_max=0 arm, or any loss is NaN. Checked by the caller (the D6 stats
script), not here -- this script logs every epoch's source validation loss
so that check can be made after the fact. A diverged arm is retried once with
--grad-clip 5.0 by the CALLER (re-invoking this script), not automatically.

Example:
  python src/run_adv_align_loso.py \
      --npz data/windows/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
      --meta data/features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
      --arch resnet_se --augmentation chandrop --adv-lambda 1.0 --adv-mode marginal \
      --epochs 40 --batch 256 --heldout 1 --out results/kc23_d6_adv_l1_s42
"""
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
from run_deep_coral_align_loso import embed, alignment_metrics, cv_bacc

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Gradient reversal (Ganin et al. 2016)
# ---------------------------------------------------------------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float):
    return _GradReverse.apply(x, lambd)


def ganin_lambda(lambd_max: float, p: float) -> float:
    """lambda(p) = lambda_max * (2 / (1 + exp(-10p)) - 1), p in [0, 1]."""
    return float(lambd_max) * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


class SubjectAdversary(nn.Module):
    """MLP 128 -> 64 -> n_subjects, applied to the (gradient-reversed) penultimate embedding."""
    def __init__(self, feat_dim: int, n_subjects: int):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_subjects)

    def forward(self, f):
        x = torch.relu(self.fc1(f))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class CDANAdversary(nn.Module):
    """Adversary on the multilinear map f (x) softmax(logits), flattened."""
    def __init__(self, feat_dim: int, n_classes: int):
        super().__init__()
        in_dim = feat_dim * n_classes
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # source vs target, binary

    def forward(self, f, probs):
        # outer product per sample: (N, feat_dim, n_classes) -> flatten
        m = torch.bmm(f.unsqueeze(2), probs.unsqueeze(1)).flatten(1)
        x = torch.relu(self.fc1(m))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def logits_feat(model, x):
    """Same convention as run_deep_coral_cnn_loso.logits_feat: (logits, penultimate_features)."""
    if hasattr(model, "features"):
        return model(x, return_feat=True)
    n = model.net(x)
    return model.head(n), torch.flatten(n, 1)


def train_adv_logged(model, Xs_tr, ys_tr, subj_tr, Xs_va, ys_va, Xt, yt_oracle, device,
                     epochs, batch, lr, patience, lambd_max, seed, adv_mode,
                     aug_mode="none", aug_sigma=0.1, aug_chandrop_p=0.2, aug_timemask_frac=0.15,
                     grad_clip=None, oracle_target_labels=False):
    """Training loop copied from run_deep_coral_align_loso.train_deep_coral_logged's
    structure (source batch, target batch every step, same optimizer/scheduler/
    early-stopping convention), with the CORAL term replaced by the subject
    adversary's cross-entropy through gradient reversal.

    subj_tr: integer subject id per Xs_tr row, RELABELLED to 0..(n_train_subjects-1)
             contiguous codes by the caller. The target class id is
             n_train_subjects (the last index).
    yt_oracle: target (held-out subject) true movement labels -- used ONLY for
             --adv-mode classcond routing (oracle, diagnostic, never for the
             classifier's own loss or for early stopping).
    """
    torch.manual_seed(seed)
    n_train_subjects = int(subj_tr.max()) + 1
    target_class_id = n_train_subjects
    n_adv_classes = n_train_subjects + 1

    w = class_weights_from_y(ys_tr, len(LABELS)).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    adv_crit = nn.CrossEntropyLoss()

    feat_dim = getattr(model, "feat_dim", 128)
    if adv_mode == "cdan":
        adversaries = {"all": CDANAdversary(feat_dim, len(LABELS)).to(device)}
    elif adv_mode == "classcond":
        adversaries = {c: SubjectAdversary(feat_dim, n_adv_classes).to(device) for c in range(len(LABELS))}
    else:  # marginal
        adversaries = {"all": SubjectAdversary(feat_dim, n_adv_classes).to(device)}

    adv_params = [p for a in adversaries.values() for p in a.parameters()]
    opt = torch.optim.Adam(list(model.parameters()) + adv_params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    src = DataLoader(WindowsDataset(Xs_tr, ys_tr), batch_size=batch, shuffle=True, drop_last=True)
    subj_tr_t = torch.from_numpy(subj_tr.astype(np.int64))
    va = DataLoader(WindowsDataset(Xs_va, ys_va), batch_size=batch, shuffle=False)

    if adv_mode == "classcond":
        # Oracle target routing: the held-out subject's TRUE labels split its
        # windows into one per-class pool, each with its own cycling loader,
        # so a target batch can be routed to the matching per-class adversary
        # head. Diagnostic only (oracle=True on every output row); never used
        # for the classifier's own loss.
        tgt_by_class = {}
        for c in range(len(LABELS)):
            Xtc = Xt[yt_oracle == c]
            if len(Xtc) < 2:
                tgt_by_class[c] = None
                continue
            bs = min(batch, len(Xtc))
            tgt_by_class[c] = DataLoader(TensorDataset(torch.from_numpy(Xtc.astype(np.float32))),
                                         batch_size=bs, shuffle=True, drop_last=(len(Xtc) >= batch))
        tgt_it_by_class = {c: (iter(dl) if dl is not None else None) for c, dl in tgt_by_class.items()}
    else:
        tgt = DataLoader(TensorDataset(torch.from_numpy(Xt.astype(np.float32))),
                         batch_size=batch, shuffle=True, drop_last=True)

    # subject ids aligned to the source loader's shuffled indices are not
    # directly recoverable from WindowsDataset alone, so build a parallel
    # loader over (index) to fetch subj_tr for the same shuffled order.
    idx_loader = DataLoader(torch.arange(len(Xs_tr)), batch_size=batch, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(seed))

    if oracle_target_labels and yt_oracle is None:
        raise ValueError("--oracle-target-labels was set but no target labels were supplied")
    if adv_mode == "classcond" and not oracle_target_labels:
        raise ValueError("--adv-mode classcond requires --oracle-target-labels (refused otherwise)")
    if adv_mode != "classcond" and oracle_target_labels:
        raise ValueError("--oracle-target-labels is only meaningful with --adv-mode classcond")

    best, best_state, bad = float("inf"), None, 0
    log = []
    total_steps = epochs * max(1, len(src))
    step = 0
    for ep in range(epochs):
        model.train()
        for a in adversaries.values():
            a.train()
        if adv_mode != "classcond":
            tgt_it = iter(tgt)
        ce_sum, adv_ce_sum, adv_acc_sum, nb = 0.0, 0.0, 0.0, 0
        # rebuild the index loader each epoch with the SAME seeding convention
        # as `src`'s own shuffle so the (Xb,yb) and subj_tr rows stay aligned:
        # both DataLoaders are built with drop_last=True over the identical
        # source array, and re-iterated together per batch below via zip.
        for (Xb, yb), idx_b in zip(src, idx_loader):
            p = step / max(1, total_steps)
            lambd = ganin_lambda(lambd_max, p)
            Xb, yb = Xb.to(device), yb.to(device)
            subj_b = subj_tr_t[idx_b].to(device)
            if aug_mode != "none":
                Xb = augment_batch(Xb, mode=aug_mode, sigma=aug_sigma,
                                    chandrop_p=aug_chandrop_p, mask_frac=aug_timemask_frac)
            opt.zero_grad(set_to_none=True)
            log_s, fs = logits_feat(model, Xb)
            ce = crit(log_s, yb)

            if adv_mode == "cdan":
                try:
                    (Xtb,) = next(tgt_it)
                except StopIteration:
                    tgt_it = iter(tgt); (Xtb,) = next(tgt_it)
                Xtb = Xtb.to(device)
                log_t, ft = logits_feat(model, Xtb)
                probs_s = torch.softmax(log_s.detach(), dim=1)
                probs_t = torch.softmax(log_t.detach(), dim=1)
                f_rev_s = grad_reverse(fs, lambd)
                f_rev_t = grad_reverse(ft, lambd)
                adv = adversaries["all"]
                out_s = adv(f_rev_s, probs_s)
                out_t = adv(f_rev_t, probs_t)
                dom_s = torch.zeros(out_s.shape[0], dtype=torch.long, device=device)
                dom_t = torch.ones(out_t.shape[0], dtype=torch.long, device=device)
                adv_ce = adv_crit(out_s, dom_s) + adv_crit(out_t, dom_t)
                adv_acc = ((out_s.argmax(1) == dom_s).float().mean()
                          + (out_t.argmax(1) == dom_t).float().mean()).item() / 2.0
            elif adv_mode == "classcond":
                adv_ce = torch.zeros((), device=device)
                correct, total = 0, 0
                for c in range(len(LABELS)):
                    m_s = (yb == c)
                    dl_c = tgt_by_class[c]
                    if dl_c is None:
                        continue
                    try:
                        (Xtc,) = next(tgt_it_by_class[c])
                    except StopIteration:
                        tgt_it_by_class[c] = iter(dl_c); (Xtc,) = next(tgt_it_by_class[c])
                    Xtc = Xtc.to(device)
                    _, ftc = logits_feat(model, Xtc)
                    if m_s.any():
                        f_rev_s = grad_reverse(fs[m_s], lambd)
                        out_s = adversaries[c](f_rev_s)
                        lbl_s = subj_b[m_s]
                        adv_ce = adv_ce + adv_crit(out_s, lbl_s)
                        correct += (out_s.argmax(1) == lbl_s).sum().item(); total += m_s.sum().item()
                    f_rev_t = grad_reverse(ftc, lambd)
                    out_t = adversaries[c](f_rev_t)
                    lbl_t = torch.full((out_t.shape[0],), target_class_id, dtype=torch.long, device=device)
                    adv_ce = adv_ce + adv_crit(out_t, lbl_t)
                    correct += (out_t.argmax(1) == lbl_t).sum().item(); total += out_t.shape[0]
                adv_acc = (correct / total) if total else 0.0
            else:  # marginal
                try:
                    (Xtb,) = next(tgt_it)
                except StopIteration:
                    tgt_it = iter(tgt); (Xtb,) = next(tgt_it)
                Xtb = Xtb.to(device)
                _, ft = logits_feat(model, Xtb)
                f_rev_s = grad_reverse(fs, lambd)
                f_rev_t = grad_reverse(ft, lambd)
                adv = adversaries["all"]
                out_s = adv(f_rev_s)
                out_t = adv(f_rev_t)
                lbl_s = subj_b
                lbl_t = torch.full((out_t.shape[0],), target_class_id, dtype=torch.long, device=device)
                adv_ce = adv_crit(out_s, lbl_s) + adv_crit(out_t, lbl_t)
                acc_s = (out_s.argmax(1) == lbl_s).float().mean()
                acc_t = (out_t.argmax(1) == lbl_t).float().mean()
                adv_acc = ((acc_s + acc_t) / 2.0).item()

            loss = ce + adv_ce
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + adv_params, grad_clip)
            opt.step()

            ce_sum += float(ce.detach()); adv_ce_sum += float(adv_ce.detach() if torch.is_tensor(adv_ce) else adv_ce)
            adv_acc_sum += float(adv_acc); nb += 1
            step += 1

        sched.step()
        vloss, _, _ = evaluate(model, va, device)
        log.append({"epoch": ep + 1, "lambda": ganin_lambda(lambd_max, ep / max(1, epochs)),
                    "train_ce": ce_sum / max(nb, 1), "train_adv_ce": adv_ce_sum / max(nb, 1),
                    "train_adv_acc": adv_acc_sum / max(nb, 1), "val_loss": float(vloss)})
        diverged = not np.isfinite(vloss)
        if diverged:
            log[-1]["diverged"] = 1
            break
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


def main():
    ap = argparse.ArgumentParser("Subject-adversarial alignment under LOSO (KC-D6).")
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="resnet_se", choices=["simple", "resnet", "resnet_se"])
    ap.add_argument("--norm-mode", default="global", choices=["global", "per_subject"],
                    help="global (default) matches the D2c/D6 harness: every learned "
                         "adaptation in the thesis starts from global normalization.")
    ap.add_argument("--adv-lambda", type=float, default=1.0, dest="adv_lambda",
                    help="lambda_max in the Ganin warm-up schedule.")
    ap.add_argument("--adv-mode", default="marginal", choices=["marginal", "classcond", "cdan"])
    ap.add_argument("--oracle-target-labels", action="store_true",
                    help="Required with --adv-mode classcond (refused otherwise). Uses the "
                         "held-out subject's TRUE labels to route target windows to the "
                         "matching per-class adversary head. Oracle, diagnostic only, never "
                         "deployable -- every output row carries oracle=True.")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="Divergence retry (Section 6.3): re-invoke with --grad-clip 5.0.")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augmentation", "--augment", dest="augmentation", default="chandrop",
                    choices=["none", "gaussian", "chandrop", "timemask", "combined"])
    ap.add_argument("--aug-sigma", type=float, default=0.1)
    ap.add_argument("--aug-chandrop-p", type=float, default=0.2)
    ap.add_argument("--aug-timemask-frac", type=float, default=0.15)
    ap.add_argument("--n-src-embed", type=int, default=4000)
    ap.add_argument("--heldout", type=int, default=None); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--instrument", default=None,
                    help="KC-D0.3 for KC-D6 (added 26 September 2026): after each fold, write the D0.3 measurements "
                         "(occlusion, attenuation, permutation reliance and the unseen-subject embedding probes: "
                         "embed_probes.csv) into this directory, taken on the held-out subject and the fold's "
                         "validation subjects, which neither the classifier nor the adversary trains on. Absent "
                         "(default): none of this code runs and every output is unchanged.")
    ap.add_argument("--probe-cap", type=int, default=100, help="Windows per subject and class for the probes.")
    ap.add_argument("--within-class-probe", action="store_true",
                    help="KC-D6 mechanism test: with --instrument, also write the subject probe inside each movement class "
                         "(subject_probe_within_class_bacc) to embed_probes.csv. Absent (default): nothing changes.")
    args = ap.parse_args()

    if args.within_class_probe and not args.instrument:
        raise SystemExit("--within-class-probe needs --instrument")
    if args.adv_mode == "classcond" and not args.oracle_target_labels:
        raise SystemExit("--adv-mode classcond requires --oracle-target-labels")
    if args.oracle_target_labels and args.adv_mode != "classcond":
        raise SystemExit("--oracle-target-labels is only meaningful with --adv-mode classcond")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    from run_config_dump import dump_run_config       # plan Section 0 rule 3: every arm writes run_config.json
    dump_run_config(out_dir, args, resolved_paths={"npz": args.npz, "meta": args.meta})
    csv_path = out_dir / "adv_subjectwise.csv"
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
    print(f"[ADV align] arch={args.arch} ({count_params(build_model(args.arch, in_ch, len(LABELS))):,} params) "
          f"adv_lambda={args.adv_lambda} adv_mode={args.adv_mode} norm={args.norm_mode} "
          f"oracle_target_labels={args.oracle_target_labels} device={device}", flush=True)

    done = set()
    if args.resume and csv_path.exists() and al_path.exists():
        done = (set(pd.read_csv(csv_path)["subject"].astype(int))
                & set(pd.read_csv(al_path)["subject"].astype(int)))

    from sklearn.metrics import f1_score, balanced_accuracy_score
    for heldout in subjects_u:
        if heldout in done:
            continue
        te = (subjects == heldout); tr = ~te
        if args.norm_mode == "global":
            mean, std = compute_train_norm(X[tr]); Xtr_all = apply_norm(X[tr], mean, std); Xte = apply_norm(X[te], mean, std)
        else:
            from train_cnn_loso import per_subject_zscore_3d
            Xn = per_subject_zscore_3d(X, subjects); Xtr_all, Xte = Xn[tr], Xn[te]
        ytr_all, subtr = y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, args.val_frac, args.seed + heldout)
        m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)

        # relabel training-fold subjects to contiguous 0..(n-1) codes for the
        # adversary head; the held-out subject's target class id is n (last).
        train_subj_ids = sorted(np.unique(subtr[m_tr]).tolist())
        subj_code = {s: i for i, s in enumerate(train_subj_ids)}
        subj_tr_codes = np.array([subj_code[s] for s in subtr[m_tr]], dtype=np.int64)

        model = build_model(args.arch, in_ch, len(LABELS)).to(device)
        model, log = train_adv_logged(
            model, Xtr_all[m_tr], ytr_all[m_tr], subj_tr_codes, Xtr_all[m_va], ytr_all[m_va],
            Xte, (y[te] if args.oracle_target_labels else None), device,
            args.epochs, args.batch, args.lr, args.patience, args.adv_lambda, args.seed,
            args.adv_mode, aug_mode=args.augmentation, aug_sigma=args.aug_sigma,
            aug_chandrop_p=args.aug_chandrop_p, aug_timemask_frac=args.aug_timemask_frac,
            grad_clip=args.grad_clip, oracle_target_labels=args.oracle_target_labels)

        te_dl = DataLoader(WindowsDataset(Xte, y[te]), batch_size=512, shuffle=False)
        _, yt, yp = evaluate(model, te_dl, device)
        diverged = any(e.get("diverged") for e in log)
        row = {"subject": int(heldout), "arch": args.arch, "adv_lambda": args.adv_lambda,
               "adv_mode": args.adv_mode, "oracle": bool(args.oracle_target_labels),
               "diverged": bool(diverged),
               "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
               "bal_acc": float(balanced_accuracy_score(yt, yp))}

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
        al = {"subject": int(heldout), "adv_mode": args.adv_mode, "epochs_run": len(log),
              "best_epoch": int(max([e["epoch"] for e in log if e.get("best") == 1] or [0]))}
        al.update(alignment_metrics(Fs, ys_pool[idx], Ft, y[te], args.seed + heldout, args.adv_lambda))

        if args.instrument:
            # Before the fold's rows are written, so a crash here re-runs the whole fold on --resume rather than
            # leaving a fold with rows and no probes. instrument_fold snapshots and restores every RNG it touches.
            from run_cnn_arch_loso import instrument_fold
            instrument_fold(model, Xte, y[te], int(heldout), args.arch, args.instrument, device, row["f1_macro"],
                            seed=args.seed, Xva=Xtr_all[m_va], yva=ytr_all[m_va], subj_va=subtr[m_va],
                            probe_cap=args.probe_cap, within_class_probe=args.within_class_probe)

        pd.DataFrame([dict(e, subject=int(heldout), adv_lambda=args.adv_lambda) for e in log]).to_csv(
            log_path, mode="a", header=not log_path.exists(), index=False)
        pd.DataFrame([al]).to_csv(al_path, mode="a", header=not al_path.exists(), index=False)
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done.add(heldout)
        print(f"[fold] Sub{heldout:02d} f1={row['f1_macro']:.4f} diverged={diverged} "
              f"domain_probe={al['domain_probe_bacc']:.3f} class_probe_tgt={al['class_probe_tgt_bacc']:.3f}",
              flush=True)

    if Path(csv_path).exists():
        df = pd.read_csv(csv_path).drop_duplicates("subject")
        m, s = df["f1_macro"].mean(), df["f1_macro"].std(ddof=1) if len(df) > 1 else float("nan")
        pd.DataFrame([{"method": f"ADV-{args.adv_mode}", "arch": args.arch, "adv_lambda": args.adv_lambda,
                       "f1_macro_mean": round(float(m), 4), "f1_macro_sd": round(float(s), 4) if s == s else None,
                       "n": len(df)}]).to_csv(out_dir / "adv_summary.csv", index=False)
        print(f"\n[ADV-{args.adv_mode}/{args.arch}] LOSO F1 = {m:.4f} (n={len(df)})")


if __name__ == "__main__":
    main()
