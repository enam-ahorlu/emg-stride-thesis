# src/d2e_analysis.py
# ---------------------------------------------------------------------------
# Analysis for D2e (docs/plans/D2e_RUN_INSTRUCTIONS.md). Reads the per-fold CSVs written by
# src/run_deep_coral_align_loso.py and applies the decision rules fixed in the instructions
# BEFORE the runs. Reads *_subjectwise.csv only.
#
#   python src/d2e_analysis.py                # Contrast 1 only (E1 vs lambda-0-train)
#   python src/d2e_analysis.py --with-e2      # also Contrast 2 (E2 vs E1), if E2 ran
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, bootstrap

DIRS = {"lam0_train": "results/deep_coral_align_lam0",
        "e1": "results/deep_coral_align_lam0_notgt",
        "e2": "results/deep_coral_align_lam30_tgteval"}
REF_772 = 0.772
REF_8300 = 0.8300
GATE_PP = 1.5


def load(key):
    d = Path(DIRS[key])
    f = pd.read_csv(d / "deep_coral_subjectwise.csv").drop_duplicates("subject", keep="last")
    a = pd.read_csv(d / "alignment_subjectwise.csv").drop_duplicates("subject", keep="last")
    m = f[["subject", "f1_macro"]].merge(a, on="subject")
    if len(m) != 40:
        raise SystemExit(f"{d}: {len(m)} complete folds, expected 40. Report the arm as unfinished.")
    return m.sort_values("subject").reset_index(drop=True)


def paired(a, b, name, scale=100.0):
    """b minus a, in pp for probe/F1-like metrics (scale=100) or raw units (scale=1)."""
    d = (b - a) * scale
    p = wilcoxon(b, a).pvalue if np.any(d != 0) else 1.0
    ci = bootstrap((d,), np.mean, method="BCa", n_resamples=10000, random_state=42).confidence_interval
    return {"metric": name, "mean_a": a.mean(), "mean_b": b.mean(), "delta": d.mean(),
            "dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
            "bca_lo": ci.low, "bca_hi": ci.high, "n_b_lower": int((d < 0).sum()), "p": p}


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def descriptive_block(lo_key, hi_key, lo_df, hi_df):
    rows = []
    for c in ["domain_probe_bacc", "class_probe_tgt_bacc", "feat_norm_src", "feat_norm_tgt"]:
        rows.append({"contrast": f"{hi_key} minus {lo_key}", "metric": c + " (descriptive)",
                     "mean_a": lo_df[c].mean(), "mean_b": hi_df[c].mean(),
                     "delta": (hi_df[c] - lo_df[c]).mean() * (100.0 if "bacc" in c else 1.0)})
    return pd.DataFrame(rows)


def verdict_contrast1(row, e1_mean):
    sig = row["p_holm"] < 0.05
    e1_lower_ge3 = sig and row["delta"] <= -3.0
    e1_near_772 = abs(e1_mean - REF_772) * 100 <= GATE_PP
    e1_near_8300 = abs(e1_mean - REF_8300) * 100 <= GATE_PP
    if e1_lower_ge3 and e1_near_772:
        return "H CONFIRMED: E1 is >=3.0pp lower than lambda-0-train (Holm-significant) and within 1.5pp of 0.772."
    if e1_near_8300:
        return "H REJECTED: E1 is within 1.5pp of 0.8300 -- the harness differs from the 0.772 run for another reason. Stop, find it, report."
    return "PARTIAL: neither confirmed nor rejected by the pre-registered thresholds. Numbers reported, no framing."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-e2", action="store_true")
    ap.add_argument("--out", default="results/deep_coral_align_d2e")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    lam0_train = load("lam0_train")
    e1 = load("e1")
    assert (lam0_train.subject.values == e1.subject.values).all()

    print("E1 mean F1:", f"{e1.f1_macro.mean():.6f}",
          f"offset from 0.772: {100*(e1.f1_macro.mean()-REF_772):+.3f} pp",
          f"offset from 0.8300: {100*(e1.f1_macro.mean()-REF_8300):+.3f} pp")

    rows = [paired(lam0_train["f1_macro"].values, e1["f1_macro"].values, "f1_macro__E1_minus_lam0train")]
    ps = [rows[0]["p"]]
    contrast2_row = None
    if args.with_e2:
        e2 = load("e2")
        assert (e1.subject.values == e2.subject.values).all()
        contrast2_row = paired(e1["f1_macro"].values, e2["f1_macro"].values, "f1_macro__E2_minus_E1")
        rows.append(contrast2_row)
        ps.append(contrast2_row["p"])

    df = pd.DataFrame(rows)
    df["p_holm"] = holm(np.array(ps))
    df.to_csv(out / "d2e_contrasts.csv", index=False)
    print("\n" + df.to_string(index=False))

    c1 = df.iloc[0]
    print("\nCONTRAST 1 VERDICT:", verdict_contrast1(c1, e1.f1_macro.mean()))

    if args.with_e2:
        c2 = df.iloc[1]
        contributes = c2["p_holm"] < 0.05 and c2["delta"] > 0.5
        print("CONTRAST 2 VERDICT:",
              "the CORAL loss contributes on its own (E2 > E1 by >0.5pp, Holm-significant)."
              if contributes else
              "the CORAL loss does not measurably contribute on its own.")

    desc = [descriptive_block("lam0train", "e1", lam0_train, e1)]
    if args.with_e2:
        desc.append(descriptive_block("lam0train", "e2", lam0_train, e2))
        desc.append(descriptive_block("e1", "e2", e1, e2))
    desc_df = pd.concat(desc, ignore_index=True)
    desc_df.to_csv(out / "d2e_descriptives.csv", index=False)
    print("\n" + desc_df.to_string(index=False))


if __name__ == "__main__":
    main()
