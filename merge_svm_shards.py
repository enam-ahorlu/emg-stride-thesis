#!/usr/bin/env python3
"""
merge_svm_shards.py  —  W-1 Stage 2 helper.

The two 150 ms SVM configs were sharded across subjects (--only-heldout) to use
idle CPU cores. This reassembles each config's per-subject rows from:
  - the original partial run's checkpoint   (results_win150_<norm>/checkpoints/*ckpt.csv)
  - every shard's checkpoint                (results_win150_<norm>_sh*/checkpoints/*ckpt.csv)
into the canonical file the driver would have written:
  results_win150_<norm>/<stem>__SVM_nested_loso_subjectwise.csv

Refuses to write unless the result has exactly 40 unique subjects (1..40) and any
subject appearing in more than one source has an identical f1_macro everywhere
(which it must, since --only-heldout is inert per-subject).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
STEM = "freq_windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_ext"


DIR_TOKEN = {"global": "global", "per_subject": "persubj"}


def gather(norm: str) -> pd.DataFrame:
    tok = DIR_TOKEN[norm]
    base = ROOT / f"results_win150_{tok}"
    dirs = [base] + sorted(ROOT.glob(f"results_win150_{tok}_sh*"))
    frames = []
    for d in dirs:
        for ck in sorted(d.glob("checkpoints/*SVM*ckpt.csv")):
            df = pd.read_csv(ck)
            if "heldout_subject" in df.columns and "f1_macro" in df.columns:
                df["_src"] = str(ck.relative_to(ROOT))
                frames.append(df)
    if not frames:
        sys.exit(f"{norm}: no checkpoint rows found")
    return pd.concat(frames, ignore_index=True)


def merge_one(norm: str) -> None:
    allrows = gather(norm)
    # consistency: every subject seen must agree on f1_macro across sources
    bad = []
    for s, g in allrows.groupby("heldout_subject"):
        if g["f1_macro"].round(12).nunique() > 1:
            bad.append((s, g[["_src", "f1_macro"]].to_dict("records")))
    if bad:
        for s, recs in bad:
            print(f"  MISMATCH Sub{s}: {recs}", file=sys.stderr)
        sys.exit(f"{norm}: sharded subjects disagree with the monolithic run — NOT inert, aborting")

    merged = (allrows.drop(columns="_src")
              .drop_duplicates(subset="heldout_subject", keep="first")
              .sort_values("heldout_subject")
              .reset_index(drop=True))
    subs = set(merged["heldout_subject"].astype(int))
    if subs != set(range(1, 41)):
        missing = sorted(set(range(1, 41)) - subs)
        sys.exit(f"{norm}: merged set is not 1..40 (n={len(subs)}, missing {missing})")
    if len(merged) != 40:
        sys.exit(f"{norm}: {len(merged)} rows after dedup, expected 40")

    canon = ROOT / f"results_win150_{DIR_TOKEN[norm]}"
    out_csv = canon / f"{STEM}__SVM_nested_loso_subjectwise.csv"
    merged.to_csv(out_csv, index=False)
    summ = {
        "model": "SVM",
        "f1_macro_mean": float(merged["f1_macro"].mean()),
        "bal_acc_mean": float(merged["bal_acc"].mean()) if "bal_acc" in merged else float("nan"),
        "acc_mean": float(merged["acc"].mean()) if "acc" in merged else float("nan"),
        "f1_macro_sd": float(merged["f1_macro"].std(ddof=1)),
        "bal_acc_sd": float(merged["bal_acc"].std(ddof=1)) if "bal_acc" in merged else float("nan"),
    }
    pd.DataFrame([summ]).to_csv(
        canon / f"{STEM}__SVM_nested_loso_summary.csv", index=False)
    print(f"{norm}: wrote {out_csv.name}  (40 subjects, f1_macro mean {summ['f1_macro_mean']:.4f})")


if __name__ == "__main__":
    for norm in ("global", "per_subject"):
        merge_one(norm)
    print("merge OK")
