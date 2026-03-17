# compute_per_subject_metrics.py
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score


def _guess_window_and_feat_from_stem(stem: str) -> tuple[int | None, str | None]:
    m_w = re.search(r"_w(150|250)(?:_|$)", stem)
    window_ms = int(m_w.group(1)) if m_w else None

    m_f = re.search(r"_features_(base|ext)(?:_|$)", stem)
    feat_set = m_f.group(1) if m_f else None

    return window_ms, feat_set


def main():
    ap = argparse.ArgumentParser(
        description="Compute per-subject macro-F1 and balanced accuracy from saved CV predictions."
    )
    ap.add_argument("--meta", type=str, required=True, help="Path to *_features_meta.csv used during training.")
    ap.add_argument("--pred-dir", type=str, required=True, help="Directory containing saved prediction .npy files.")
    ap.add_argument("--scheme", type=str, default="subjdep", help="CV scheme tag (default: subjdep).")
    ap.add_argument("--out", type=str, default="per_subject_metrics.csv", help="Output CSV path.")
    args = ap.parse_args()

    meta = pd.read_csv(Path(args.meta))

    subj_col = None
    for c in ["subject", "subject_id", "subject_int", "sid", "subj"]:
        if c in meta.columns:
            subj_col = c
            break
    if subj_col is None:
        raise KeyError(f"No subject column found in meta. Have columns: {list(meta.columns)}")

    subj_ids = meta[subj_col].astype(int).to_numpy()

    pred_dir = Path(args.pred_dir)
    ypred_files = sorted(pred_dir.glob(f"*_{args.scheme}_y_pred.npy"))
    if not ypred_files:
        raise FileNotFoundError(f"No y_pred files found in {pred_dir} with scheme={args.scheme}")

    rows = []
    for ypred_path in ypred_files:
        ytrue_path = pred_dir / ypred_path.name.replace("_y_pred.npy", "_y_true.npy")
        if not ytrue_path.exists():
            raise FileNotFoundError(f"Missing y_true for {ypred_path.name}: expected {ytrue_path.name}")


        base = ypred_path.name.replace("_y_pred.npy", "")
        model = "UNKNOWN"
        stem = base

        # filenames end with "..._<scheme>" after removing "_y_pred.npy"
        if base.endswith(f"_{args.scheme}"):
            before = base[: -(len(args.scheme) + 1)]  # remove "_subjdep" (or "_loso", etc.)


            m = re.match(r"^(?P<stem>.+?_features_(?:base|ext))_(?P<model>.+)$", before)
            if m:
                stem = m.group("stem")
                model = m.group("model")
            else:
                stem = before
                model = "UNKNOWN"
        else:
            stem = base
            model = "UNKNOWN"


        window_ms, feat_set = _guess_window_and_feat_from_stem(stem)


        y_true = np.load(ytrue_path).astype(int)
        y_pred = np.load(ypred_path).astype(int)

        if len(y_true) != len(meta) or len(y_pred) != len(meta):
            raise ValueError(
                f"Length mismatch vs meta rows. meta={len(meta)}, y_true={len(y_true)}, y_pred={len(y_pred)}."
            )

        for subj in np.unique(subj_ids):
            m = subj_ids == subj
            rows.append({
                "source_file": stem,
                "window_ms": window_ms,
                "feat_set": feat_set,
                "model": model,
                "subject": int(subj),
                "n_windows": int(m.sum()),
                "acc": float(accuracy_score(y_true[m], y_pred[m])),
                "f1_macro": float(f1_score(y_true[m], y_pred[m], average="macro")),
                "bal_acc": float(balanced_accuracy_score(y_true[m], y_pred[m])),
            })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[save] per-subject metrics: {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
