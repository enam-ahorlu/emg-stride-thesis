# compare_easy_hard_confusion.py (CNN-friendly)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

LABELS = ["DNS", "STDUP", "UPS", "WAK"]  # movement types (alphabetical = encode order)

def to_label_str(arr):
    """
    Convert y arrays to label strings in LABELS order.
    - If arr is numeric (0..6), map to LABELS.
    - Otherwise cast to stripped strings.
    """
    arr = np.asarray(arr)

    # numeric predictions like 0..6
    if np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(int)
        return np.array([LABELS[i] for i in arr], dtype=object)

    # strings or mixed -> force string
    return np.array([str(x).strip() for x in arr], dtype=object)

def plot_cm(cm, title, outpath):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def load_subject_list(csv_path: Path):
    df = pd.read_csv(csv_path)
    # accept "subject" or "heldout"
    col = "subject" if "subject" in df.columns else ("heldout" if "heldout" in df.columns else None)
    if col is None:
        raise ValueError(f"{csv_path} missing subject column. Columns: {list(df.columns)}")
    return set(int(x) for x in df[col].tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="meta CSV with subject + label col")
    ap.add_argument("--label-col", default="movement", help="label column in meta (default: movement)")
    ap.add_argument("--pred-dir", required=True, help="folder containing per-subject y_pred npy files")
    ap.add_argument("--stem", required=True, help="base stem used in filenames (e.g. windows_..._AorR)")
    ap.add_argument("--model", required=True, help="CNN / SVM / RF (used to locate filename pattern)")
    ap.add_argument("--easy-csv", required=True)
    ap.add_argument("--hard-csv", required=True)
    ap.add_argument("--outdir", default="error_analysis/easy_hard_confusion")
    args = ap.parse_args()

    meta = pd.read_csv(args.meta)
    if args.label_col not in meta.columns:
        raise ValueError(f"meta missing {args.label_col}. Have: {list(meta.columns)}")
    if "subject" not in meta.columns:
        raise ValueError("meta missing 'subject' column")

    subjects = meta["subject"].astype(int).to_numpy()
    y_true_all = meta[args.label_col].to_numpy()

    easy = load_subject_list(Path(args.easy_csv))
    hard = load_subject_list(Path(args.hard_csv))

    pred_dir = Path(args.pred_dir)
    outdir = Path(args.outdir) / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    def subj_pred_path(sub):
        # CNN naming in your run: {stem}_CNN_loso_SubXX_y_pred.npy
        return pred_dir / f"{args.stem}_{args.model}_loso_Sub{sub:02d}_y_pred.npy"

    def group_cm(group_set, tag):
        cm_sum = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)

        for sub in sorted(group_set):
            mask = (subjects == sub)
            y_true = y_true_all[mask]

            p = subj_pred_path(sub)
            if not p.exists():
                raise FileNotFoundError(f"Missing per-subject pred file for Sub{sub:02d}: {p}")

            y_pred = np.load(p)

            if len(y_pred) != mask.sum():
                raise ValueError(
                    f"Length mismatch for Sub{sub:02d}: meta windows={mask.sum()} but y_pred={len(y_pred)}\n"
                    f"File: {p}"
                )

            y_true = to_label_str(y_true)
            y_pred = to_label_str(y_pred)

            cm = confusion_matrix(y_true, y_pred, labels=LABELS)
            cm_sum += cm

        # normalize row-wise
        row_sums = cm_sum.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_sum, np.maximum(row_sums, 1), dtype=float)

        np.savetxt(outdir / f"cm_{tag}_counts.csv", cm_sum, delimiter=",", fmt="%d")
        np.savetxt(outdir / f"cm_{tag}_normalized.csv", cm_norm, delimiter=",", fmt="%.6f")
        plot_cm(cm_norm, f"{args.model} LOSO CM ({tag}) normalized", outdir / f"cm_{tag}_normalized.png")

    group_cm(easy, "easy")
    group_cm(hard, "hard")
    print(f"[done] Easy vs hard confusion outputs saved to: {outdir}")

if __name__ == "__main__":
    main()