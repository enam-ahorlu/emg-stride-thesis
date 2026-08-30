#!/usr/bin/env python3
"""
prepare_deployment_artifacts.py
===============================
Trains the two MyoLens deployment models on 37 of the 40 SIAT-LLMD subjects,
holds 3 out as a deployment validation set, exports both to ONNX, and writes a
provenance manifest plus reference outputs for the separate equivalence check.

This is the first model-persisting script in this codebase. Every training
script in the thesis pipeline is a LOSO *evaluation* driver that fits and
discards a model per fold; none of them contains torch.save or joblib.dump.

WHAT THIS DELIBERATELY REPRODUCES FROM THE THESIS (do not "improve" these):
  * LABELS order is hard-coded ["DNS","STDUP","UPS","WAK"]. The thesis derives
    it from sorted(unique(y)), which happens to agree ONLY because all four
    classes are present. On a subject subset that coincidence can break and the
    SVM's probability columns would silently stop aligning with the ResNet's.
  * Channel dropout has NO 1/(1-p) rescaling. Surviving channels are not scaled
    up, so training-time input energy is ~20% below inference-time. It looks
    like a bug; it is what the 84.0% figure was trained with.
  * Per-subject z-score is transductive: every subject, held-out included, is
    standardised by its own statistics. There is therefore NO fitted normaliser
    to ship. The ONNX graphs expect ALREADY-NORMALISED input, and the serving
    code must z-score each session against that session's own statistics before
    calling them. This is stated on the model card for a reason.

Run from the 06_Code directory with the thesis venv.

    python prepare_deployment_artifacts.py --out ../../Advanced Software Engineering/MyoLens/artifacts

Author: Enam Ahorlu · CSCD602 pre-clock artefact preparation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants --
# Hard-coded, NOT derived. See module docstring.
LABELS = ["DNS", "STDUP", "UPS", "WAK"]
LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}
N_CLASSES = len(LABELS)
N_CHANNELS = 9

# The actual SIAT-LLMD sEMG rate, measured from the recording timestamps:
# meta["fs"] == 1920.0001344 Hz, win_samples == 480, and 480/1920 = 0.250 s exactly.
# The 250 ms window and 125 ms step are therefore correct; the 2000 Hz figure
# quoted in the thesis is not. See the note on FS_FEATURE_CONST below.
FS_TRUE_HZ = 1920.0
WINDOW_SAMPLES = 480          # 250 ms at 1920 Hz
STEP_SAMPLES = 240            # 125 ms, 50% overlap

# The thesis feature extractor hardcodes sampling_rate = 2000.0 (verified in
# features_out/freq_..._features_cfg.json) and uses it for MNF and MDF. Those
# two columns are therefore scaled by 2000/1920 ≈ 1.042 relative to true Hz.
# Per-column z-scoring absorbs a constant scale factor, so this does not affect
# classification — but the serving feature extractor MUST use the same constant,
# because reproducing the model exactly is worth more than being right about Hz.
FS_FEATURE_CONST = 2000.0

CHANNELS = [
    "sEMG: tensor fascia lata",
    "sEMG: rectus femoris",
    "sEMG: vastus medialis",
    "sEMG: semimembranosus",
    "sEMG: upper tibialis anterior",
    "sEMG: lower tibialis anterior",
    "sEMG: lateral gastrocnemius",
    "sEMG: medial gastrocnemius",
    "sEMG: soleus",
]

# Freq-72 is FEATURE-MAJOR: 8 blocks of 9 channels, not 9 blocks of 8 features.
FEATURE_BLOCKS = ["MAV", "RMS", "WL", "ZC", "WAMP", "MNF", "MDF", "logSP"]
FEATURE_NAMES = [f"{b}_ch{c}" for b in FEATURE_BLOCKS for c in range(N_CHANNELS)]
assert len(FEATURE_NAMES) == 72

DEFAULT_FEATURES = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
DEFAULT_META = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
DEFAULT_WINDOWS = "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz"


# ------------------------------------------------------------------ helpers --
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def per_subject_zscore_2d(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Feature-space transductive z-score. Mirrors train_classical_loso.py.

    One (mu, sd) pair per (subject, feature column). Population std, ddof=0.
    """
    out = X.copy().astype(np.float64)
    for sid in np.unique(subjects):
        m = subjects == sid
        block = out[m]
        mu = block.mean(axis=0, keepdims=True)
        sd = block.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        out[m] = (block - mu) / sd
    return out.astype(np.float32)


def per_subject_zscore_3d(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Envelope-space transductive z-score. Verbatim behaviour of
    train_cnn_loso.py:184 — pools over each subject's windows AND time,
    giving one (mu, sd) per (subject, channel). Population std, ddof=0.
    """
    out = X.copy()
    for sid in np.unique(subjects):
        m = subjects == sid
        block = X[m]                                    # (Ns, C, T)
        mu = block.mean(axis=(0, 2), keepdims=True)     # (1, C, 1)
        sd = block.std(axis=(0, 2), keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        out[m] = (block - mu) / sd
    return out


def class_weights_from_y(y: np.ndarray, n_classes: int):
    import torch
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        missing = [LABELS[i] for i, c in enumerate(counts) if c == 0]
        raise SystemExit(
            f"FATAL: class(es) {missing} absent from the training split. "
            "class_weights_from_y would assign weight N/1 with no warning. "
            "Choose different held-out subjects."
        )
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def choose_val_subjects(train_subjects: np.ndarray, frac: float, seed: int) -> np.ndarray:
    n = max(1, int(round(frac * len(train_subjects))))
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(np.sort(train_subjects), size=n, replace=False))


# ------------------------------------------------------------------- loading --
def load_all(root: Path, features: str, meta: str, windows: str):
    fp, mp, wp = root / features, root / meta, root / windows
    for p in (fp, mp, wp):
        if not p.exists():
            raise SystemExit(f"FATAL: missing input {p}")

    X_feat = np.load(fp, allow_pickle=False)["X"]
    if X_feat.ndim != 2 or X_feat.shape[1] != 72:
        raise SystemExit(f"FATAL: expected (N,72) Freq-72 features, got {X_feat.shape}. "
                         "Note Freq-72 lives in *_features_ext.npz, not *_base.npz.")

    md = pd.read_csv(mp)
    if "subject" not in md.columns:
        # subject_int is LEXICOGRAPHICALLY ordered (subject 2 -> 11). Never use it.
        raise SystemExit("FATAL: meta CSV has no 'subject' column. Do not fall back to "
                         "'subject_int' — it is sorted as strings and permutes IDs.")
    label_col = next((c for c in ("movement", "mode_label", "label", "y_str") if c in md.columns), None)
    if label_col is None:
        raise SystemExit(f"FATAL: no label column in {mp.name}; have {list(md.columns)[:12]}")

    subjects = md["subject"].astype(int).to_numpy()
    raw_labels = md[label_col].astype(str).str.strip().to_numpy()
    unknown = sorted(set(raw_labels) - set(LABELS))
    if unknown:
        raise SystemExit(f"FATAL: unexpected label values {unknown}; expected {LABELS}")
    y = np.array([LABEL_TO_IDX[s] for s in raw_labels], dtype=np.int64)

    wz = np.load(wp, allow_pickle=False)
    X_env = wz["X_env"].astype(np.float32)
    if X_env.ndim != 3 or X_env.shape[1] != N_CHANNELS:
        raise SystemExit(f"FATAL: expected (N,9,T) envelopes, got {X_env.shape}")
    if X_env.shape[2] != WINDOW_SAMPLES:
        raise SystemExit(
            f"FATAL: window length {X_env.shape[2]} != expected {WINDOW_SAMPLES}. "
            f"The w250 file should be 480 samples (250 ms at {FS_TRUE_HZ:g} Hz); "
            f"288 would mean you pointed at the w150 file."
        )

    if not (len(X_feat) == len(md) == len(X_env)):
        raise SystemExit(f"FATAL: row-count mismatch — features {len(X_feat)}, "
                         f"meta {len(md)}, windows {len(X_env)}. These must be row-aligned.")

    return X_feat, X_env, y, subjects, {"features": fp, "meta": mp, "windows": wp}


# ------------------------------------------------------------------ training --
def train_svm(Xtr, ytr, C: float, gamma, seed: int):
    from sklearn.svm import SVC
    clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced",
              cache_size=500, probability=True, random_state=seed)
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    return clf, time.perf_counter() - t0


def train_resnet(Xtr, ytr, Xva, yva, *, epochs, batch, lr, patience, seed,
                 chandrop_p, device):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from cnn_architectures import build_model, count_params

    torch.manual_seed(seed)
    model = build_model("resnet_se", in_ch=N_CHANNELS, n_classes=N_CLASSES).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights_from_y(ytr, N_CLASSES).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=batch, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=batch, shuffle=False)

    best, best_state, bad = float("inf"), None, 0
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr:
            Xb, yb = Xb.to(device), yb.to(device)
            # Channel dropout — training batches only, NO 1/(1-p) rescaling.
            if chandrop_p > 0:
                keep = (torch.rand(Xb.shape[0], Xb.shape[1], 1, device=device) >= chandrop_p).float()
                Xb = Xb * keep
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for Xb, yb in va:
                Xb, yb = Xb.to(device), yb.to(device)
                tot += crit(model(Xb), yb).item() * len(yb)
                n += len(yb)
        vloss = tot / max(n, 1)
        print(f"    epoch {ep:>3}/{epochs}  val_loss {vloss:.4f}"
              f"{'  *' if vloss < best else ''}", flush=True)
        if vloss < best:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"    early stop at epoch {ep} (patience {patience})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, count_params(model), time.perf_counter() - t0


# ---------------------------------------------------------------- inference --
def resnet_proba(model, X, device, batch=512):
    import torch
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def soft_vote(p_a, p_b):
    """Unweighted arithmetic mean, exactly as ensemble_v2_combine.soft_vote
    with weights=None. Both inputs must be in LABELS order."""
    return (p_a + p_b) / 2.0


def scores(y_true, proba):
    from sklearn.metrics import f1_score, balanced_accuracy_score
    pred = proba.argmax(1)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "n_windows": int(len(y_true)),
    }


# ------------------------------------------------------------------- export --
def export_svm_onnx(clf, out_path: Path):
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    onx = to_onnx(
        clf,
        initial_types=[("input", FloatTensorType([None, 72]))],
        options={id(clf): {"zipmap": False}},   # plain (N,4) probability tensor
        target_opset=17,
    )
    out_path.write_bytes(onx.SerializeToString())


def export_resnet_onnx(model, out_path: Path, device):
    import torch
    model.eval()
    dummy = torch.zeros(1, N_CHANNELS, WINDOW_SAMPLES, device=device)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description="Build MyoLens deployment artefacts.")
    ap.add_argument("--root", default=".", help="06_Code directory")
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--windows", default=DEFAULT_WINDOWS)
    ap.add_argument("--out", required=True, help="Artefact output directory")
    ap.add_argument("--holdout", default="10,13,22",
                    help="Held-out subjects: one hard, one easy, one middling (thesis 5.4)")
    ap.add_argument("--svm-c", type=float, default=1.0, help="_bestparams.json: all 40 folds agree on C=1")
    ap.add_argument("--svm-gamma", default="scale")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--chandrop-p", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-deep", action="store_true", help="SVM only (fast smoke test)")
    args = ap.parse_args()

    import torch
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))          # for cnn_architectures
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Without these, cuDNN picks convolution algorithms by benchmark and two
        # runs with the same seed differ by a few tenths of a point. Determinism
        # costs almost nothing at this scale and lets the artefact be regenerated
        # byte-identically, which is a claim worth being able to make.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # All INFERENCE runs on CPU, even when training used the GPU. Cloud Run serves
    # on CPU, so CPU is the reference that matters: it makes the reported holdout
    # figures the ones the deployed model actually produces, and it means the
    # ONNX equivalence test measures export fidelity rather than CUDA-vs-CPU
    # kernel differences (which are ~2e-4 on a softmax and would otherwise
    # swamp the 1e-4 tolerance).
    infer_device = torch.device("cpu")

    print(f"[env] python {platform.python_version()} · torch {torch.__version__} · device {device}")
    if device.type == "cuda":
        print(f"[env] gpu: {torch.cuda.get_device_name(0)}")

    # ---- load ----------------------------------------------------------------
    X_feat, X_env, y, subjects, paths = load_all(root, args.features, args.meta, args.windows)
    all_subj = np.unique(subjects)
    print(f"[data] {len(y):,} windows · {len(all_subj)} subjects · "
          f"class counts {dict(zip(LABELS, np.bincount(y, minlength=4).tolist()))}")

    holdout = np.array(sorted(int(s) for s in args.holdout.split(",")))
    missing = set(holdout) - set(all_subj.tolist())
    if missing:
        raise SystemExit(f"FATAL: held-out subjects {sorted(missing)} not in the dataset")
    te_mask = np.isin(subjects, holdout)
    tr_mask = ~te_mask
    train_subj = np.unique(subjects[tr_mask])
    print(f"[split] train {len(train_subj)} subjects · holdout {holdout.tolist()}")

    for name, m in (("train", tr_mask), ("holdout", te_mask)):
        counts = np.bincount(y[m], minlength=N_CLASSES)
        if (counts == 0).any():
            raise SystemExit(f"FATAL: {name} split is missing class(es) "
                             f"{[LABELS[i] for i, c in enumerate(counts) if c == 0]}")
        print(f"[split] {name}: {dict(zip(LABELS, counts.tolist()))}")

    # ---- transductive per-subject normalisation ------------------------------
    # Applied across ALL subjects, each by its own statistics — the thesis's
    # headline condition. Nothing here is persistable; the service recomputes
    # it per session. See module docstring.
    print("[norm] per-subject z-score (transductive) …")
    Xf = per_subject_zscore_2d(X_feat, subjects)
    Xe = per_subject_zscore_3d(X_env, subjects)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "labels": LABELS,
        "channels": CHANNELS,
        "feature_names": FEATURE_NAMES,
        "feature_order": "feature-major: 8 blocks of 9 channels",
        "window": {
            "samples": WINDOW_SAMPLES, "ms": 250,
            "step_samples": STEP_SAMPLES, "step_ms": 125,
            "fs_true_hz": FS_TRUE_HZ,
            "fs_feature_constant": FS_FEATURE_CONST,
            "fs_note": "Recordings are 1920 Hz (meta['fs'] = 1920.0001344). The feature "
                       "extractor hardcodes 2000.0 for MNF/MDF; the serving extractor must "
                       "use the same constant to reproduce the model exactly. Per-column "
                       "z-scoring absorbs the resulting constant scale factor.",
        },
        "preprocessing": {"bandpass_hz": [20, 450], "bandpass_order": 4, "envelope_ms": 50},
        "normalisation": {
            "mode": "per_subject_zscore (transductive)",
            "features": "per (subject, column), ddof=0, sd<1e-8 -> 1.0",
            "envelopes": "per (subject, channel) over axes (0,2), ddof=0, sd<1e-8 -> 1.0",
            "note": "NOT persistable. ONNX graphs expect already-normalised input; "
                    "the service must z-score each session against that session's own statistics.",
        },
        "training_subjects": train_subj.tolist(),
        "holdout_subjects": holdout.tolist(),
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "train_device": str(device),
            "inference_device": "cpu",
            "cudnn_deterministic": bool(torch.cuda.is_available()),
            "note": "Inference and the reference outputs are computed on CPU because "
                    "Cloud Run serves on CPU. Reported holdout figures are therefore "
                    "the figures the deployed model produces.",
        },
        "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in paths.items()},
        "models": {},
    }

    # ---- SVM -----------------------------------------------------------------
    print("[svm] fitting …")
    gamma = args.svm_gamma if args.svm_gamma == "scale" else float(args.svm_gamma)
    svm, svm_secs = train_svm(Xf[tr_mask], y[tr_mask], args.svm_c, gamma, args.seed)
    p_svm = svm.predict_proba(Xf[te_mask]).astype(np.float64)
    if list(svm.classes_) != list(range(N_CLASSES)):
        raise SystemExit(f"FATAL: SVM classes_ = {svm.classes_}, expected 0..3 in LABELS order")
    svm_scores = scores(y[te_mask], p_svm)
    print(f"[svm] fit {svm_secs:.1f}s · holdout macro-F1 {svm_scores['macro_f1']:.4f}")

    svm_onnx = out / "svm_freq72.onnx"
    export_svm_onnx(svm, svm_onnx)
    manifest["models"]["svm"] = {
        "file": svm_onnx.name, "sha256": sha256(svm_onnx),
        "params": {"kernel": "rbf", "C": args.svm_c, "gamma": args.svm_gamma,
                   "class_weight": "balanced", "probability": True},
        "input": {"name": "input", "shape": [None, 72], "dtype": "float32",
                  "expects": "per-session z-scored Freq-72 features"},
        "fit_seconds": round(svm_secs, 1),
        "holdout": svm_scores,
    }

    # ---- ResNet-SE+CD --------------------------------------------------------
    p_res = None
    if not args.skip_deep:
        val_subj = choose_val_subjects(train_subj, args.val_frac, args.seed)
        va_mask = np.isin(subjects, val_subj)
        fit_mask = tr_mask & ~va_mask
        print(f"[deep] val subjects {val_subj.tolist()} · "
              f"fit {int(fit_mask.sum()):,} / val {int(va_mask.sum()):,} windows")

        model, n_params, deep_secs = train_resnet(
            Xe[fit_mask], y[fit_mask], Xe[va_mask], y[va_mask],
            epochs=args.epochs, batch=args.batch, lr=args.lr, patience=args.patience,
            seed=args.seed, chandrop_p=args.chandrop_p, device=device)

        model = model.to(infer_device)      # serve-on-CPU reference, see above
        p_res = resnet_proba(model, Xe[te_mask], infer_device)
        res_scores = scores(y[te_mask], p_res)
        print(f"[deep] {n_params:,} params · fit {deep_secs/60:.1f} min · "
              f"holdout macro-F1 {res_scores['macro_f1']:.4f}")

        res_onnx = out / "resnet_se_cd.onnx"
        export_resnet_onnx(model, res_onnx, infer_device)
        manifest["models"]["resnet_se_cd"] = {
            "file": res_onnx.name, "sha256": sha256(res_onnx),
            "params": {"arch": "EMGResNet1D", "use_se": True, "n_params": n_params,
                       "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                       "weight_decay": 1e-4, "scheduler": "CosineAnnealingLR",
                       "patience": args.patience, "chandrop_p": args.chandrop_p,
                       "chandrop_rescaling": False,
                       "val_subjects": val_subj.tolist()},
            "input": {"name": "input", "shape": [None, 9, 500], "dtype": "float32",
                      "expects": "per-session z-scored linear envelopes"},
            "output": {"name": "logits", "note": "apply softmax in the service"},
            "fit_seconds": round(deep_secs, 1),
            "holdout": res_scores,
        }

    # ---- ensemble ------------------------------------------------------------
    if p_res is not None:
        p_ens = soft_vote(p_svm, p_res)
        ens_scores = scores(y[te_mask], p_ens)
        print(f"[ens] soft vote (unweighted mean) · holdout macro-F1 {ens_scores['macro_f1']:.4f}")
        manifest["ensemble"] = {
            "members": ["svm", "resnet_se_cd"],
            "rule": "unweighted arithmetic mean of probability vectors, then argmax",
            "reference": "ensemble_v2_combine.soft_vote with weights=None",
            "holdout": ens_scores,
        }

    # ---- reference outputs for the equivalence check -------------------------
    # onnxruntime is intentionally NOT imported here: the thesis venv is Python
    # 3.14 and cp314 support is unconfirmed. verify_onnx_equivalence.py runs
    # these against the ONNX graphs in a Python 3.12 venv that matches the
    # serving container.
    ref = out / "reference_outputs.npz"
    payload = {
        "features": Xf[te_mask][:200].astype(np.float32),
        "svm_proba": p_svm[:200].astype(np.float64),
        "y_true": y[te_mask][:200].astype(np.int64),
    }
    if p_res is not None:
        payload["envelopes"] = Xe[te_mask][:200].astype(np.float32)
        payload["resnet_proba"] = p_res[:200].astype(np.float64)
    np.savez_compressed(ref, **payload)
    manifest["reference_outputs"] = {"file": ref.name, "sha256": sha256(ref), "n_windows": 200}

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n[done] artefacts in {out}")
    for f in sorted(out.iterdir()):
        print(f"       {f.name:<28} {f.stat().st_size/1024:>9,.0f} KB")
    print("\nNext: run verify_onnx_equivalence.py in the Python 3.12 venv.")


if __name__ == "__main__":
    main()
