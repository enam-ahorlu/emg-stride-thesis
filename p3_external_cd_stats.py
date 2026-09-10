#!/usr/bin/env python3
"""
p3_external_cd_stats.py
======================
Stage P-3 of the channel-dropout parity programme (EXPERIMENT_PLAN_CD_PARITY.md
section 4). External replication of the channel-dropout augmentation gain on
ENABL3S. No GPU. All four arms already exist.

  results_ext_resnet_se_persubj            ResNet-SE, per-subject norm, no aug
  results_ext_chandrop_resnet_se_persubj   ResNet-SE + channel dropout, per-subject norm
  results_ext_resnet_se_global             ResNet-SE, global norm, no aug
  results_ext_chandrop_resnet_se_global    ResNet-SE + channel dropout, global norm

Primary : channel dropout vs no aug under per-subject normalization, paired /10.
Secondary: same contrast under global normalization.
Holm across the family of two. At n = 10 the effect size is the primary
evidence and the p-value the secondary, the convention section 4.12 uses for the
RF replication. Grid R / W / F / X in section 4.3.

Estimators from window_ablation_stats.py, verified by hand in W-1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from window_ablation_stats import bca_ci, cohens_d_paired  # verified in W-1

OUT = ROOT / "results_parity"
OUT.mkdir(exist_ok=True)

ARMS = {
    "persubj_base": "results_ext_resnet_se_persubj",
    "persubj_cd": "results_ext_chandrop_resnet_se_persubj",
    "global_base": "results_ext_resnet_se_global",
    "global_cd": "results_ext_chandrop_resnet_se_global",
}


def load(dirname: str) -> pd.Series:
    p = ROOT / dirname / "cnn_arch_subjectwise.csv"
    if not p.exists():
        sys.exit(f"missing {p}")
    df = pd.read_csv(p)
    sc = "subject" if "subject" in df.columns else "heldout_subject"
    df = df[[sc, "f1_macro"]].dropna().rename(columns={sc: "subject"})
    if df["subject"].duplicated().any():
        sys.exit(f"{p} has duplicated subjects; a --resume double-appended")
    return df.set_index("subject")["f1_macro"].sort_index().rename(dirname)


def holm2(p_primary: float, p_secondary: float) -> tuple[float, float]:
    ps = sorted([("primary", p_primary), ("secondary", p_secondary)], key=lambda kv: kv[1])
    out, run = {}, 0.0
    for k, (name, p) in enumerate(ps):
        run = max(run, min(1.0, p * (2 - k)))
        out[name] = run
    return out["primary"], out["secondary"]


def contrast(base: pd.Series, cd: pd.Series, label: str) -> dict:
    if not base.index.equals(cd.index):
        sys.exit(f"{label}: subject ids do not match ({list(base.index)} vs {list(cd.index)})")
    d = (cd - base).to_numpy()
    lo, hi = bca_ci(d)
    w = stats.wilcoxon(cd.to_numpy(), base.to_numpy())
    dz = cohens_d_paired(cd.to_numpy(), base.to_numpy())
    print(f"  {label}")
    print(f"    no aug {base.mean():.4f}  ->  channel dropout {cd.mean():.4f}")
    print(f"    paired delta {d.mean()*100:+.2f} pp   95% BCa [{lo*100:+.2f}, {hi*100:+.2f}] pp   "
          f"Wilcoxon p = {w.pvalue:.4g}   Cohen's d = {dz:+.2f}   "
          f"({int((d>0).sum())}/{int((d<0).sum())} +/-)   n = {len(d)}")
    return {"label": label, "mean_base": float(base.mean()), "mean_cd": float(cd.mean()),
            "delta_pp": float(d.mean() * 100), "bca_lo_pp": lo * 100, "bca_hi_pp": hi * 100,
            "p_raw": float(w.pvalue), "cohens_d": dz, "n": len(d),
            "pos": int((d > 0).sum()), "neg": int((d < 0).sum())}


def main() -> int:
    print("=" * 78)
    print("STAGE P-3: external replication of the channel-dropout gain (ENABL3S)")
    print("=" * 78)

    s = {k: load(v) for k, v in ARMS.items()}
    print("\n--- input files ---")
    for k, v in ARMS.items():
        print(f"  {v:<40} n = {len(s[k])}   ids {list(s[k].index)}")

    ids = s["persubj_base"].index
    all_match = all(s[k].index.equals(ids) for k in ARMS)
    print(f"\n  subject id sets identical across all four arms: {all_match}")
    if not all_match:
        print("\nP-3 OUTCOME X: subject ids do not pair across arms. Stop and report.")
        json.dump({"stage": "P-3", "outcome": "X",
                   "outcome_meaning": "subject ids do not pair across arms"},
                  open(OUT / "p3_outcome.json", "w"), indent=2)
        return 0
    n_paired = len(ids)

    print("\n--- contrasts ---")
    primary = contrast(s["persubj_base"], s["persubj_cd"],
                       "PRIMARY: channel dropout vs no aug, per-subject normalization")
    secondary = contrast(s["global_base"], s["global_cd"],
                         "SECONDARY: channel dropout vs no aug, global normalization")

    p_h_primary, p_h_secondary = holm2(primary["p_raw"], secondary["p_raw"])
    primary["p_holm"] = p_h_primary
    secondary["p_holm"] = p_h_secondary
    print(f"\n  Holm across the family of two:")
    print(f"    primary   raw p = {primary['p_raw']:.4g}   Holm p = {p_h_primary:.4g}")
    print(f"    secondary raw p = {secondary['p_raw']:.4g}   Holm p = {p_h_secondary:.4g}")

    # ---- pre-registered grid (section 4.3), driven by the PRIMARY contrast ----
    d = primary["cohens_d"]
    delta = primary["delta_pp"]
    if delta <= 0 or d < 0.2:
        letter = "F"
        meaning = ("Fails to replicate. The primary point estimate is at or below zero, or the "
                   "effect size is below 0.2. The augmentation gain is dataset-specific in a way "
                   "normalization is not; this belongs in section 5.12 as a limitation.")
    elif d >= 0.5 and p_h_primary < 0.05:
        letter = "R"
        meaning = ("Replicates. Primary contrast positive, Holm p < 0.05, d >= 0.5. The "
                   "augmentation gain is not SIAT-specific; goes into section 4.12 beside the "
                   "normalization replication.")
    elif d >= 0.5 and p_h_primary >= 0.05:
        letter = "W"
        meaning = ("Weakly replicates. Positive with d >= 0.5 but Holm p >= 0.05. Consistent in "
                   "direction and size at n = 10 where power is low; report leaning on the effect "
                   "size, as the RF replication is reported.")
    else:
        letter = "W"
        meaning = (f"Positive (delta {delta:+.2f} pp, d {d:+.2f}) but effect size below the 0.5 "
                   "threshold for R/W. Direction consistent; report as a weak replication leaning "
                   "on the effect size, and note d is between 0.2 and 0.5.")
    print("\n" + "=" * 78)
    print(f"P-3 OUTCOME {letter}: {meaning}")
    print("=" * 78)

    rows = pd.DataFrame({
        "subject": ids,
        "persubj_base_f1": s["persubj_base"].values,
        "persubj_cd_f1": s["persubj_cd"].values,
        "persubj_delta_pp": (s["persubj_cd"] - s["persubj_base"]).values * 100,
        "global_base_f1": s["global_base"].values,
        "global_cd_f1": s["global_cd"].values,
        "global_delta_pp": (s["global_cd"] - s["global_base"]).values * 100,
    }).round(5)
    rows.to_csv(OUT / "p3_external_cd_tests.csv", index=False)

    res = {"stage": "P-3", "n_paired": n_paired,
           "primary": primary, "secondary": secondary,
           "outcome": letter, "outcome_meaning": meaning}
    json.dump(res, open(OUT / "p3_outcome.json", "w"), indent=2)

    (OUT / "p3_verdict.md").write_text(
        f"# Stage P-3 verdict: outcome {letter}\n\n{meaning}\n\n"
        f"Subjects paired: {n_paired} (ids {list(ids)}); subject id sets identical across all four arms.\n\n"
        f"## Primary: channel dropout vs no augmentation, per-subject normalization\n\n"
        f"- no aug {primary['mean_base']:.4f} to channel dropout {primary['mean_cd']:.4f}\n"
        f"- paired delta {primary['delta_pp']:+.2f} pp, 95% BCa [{primary['bca_lo_pp']:+.2f}, "
        f"{primary['bca_hi_pp']:+.2f}] pp\n"
        f"- Wilcoxon raw p = {primary['p_raw']:.4g}, Holm p = {primary['p_holm']:.4g}, "
        f"Cohen's d = {primary['cohens_d']:+.2f} ({primary['pos']}/{primary['neg']} +/-)\n\n"
        f"## Secondary: channel dropout vs no augmentation, global normalization\n\n"
        f"- no aug {secondary['mean_base']:.4f} to channel dropout {secondary['mean_cd']:.4f}\n"
        f"- paired delta {secondary['delta_pp']:+.2f} pp, 95% BCa [{secondary['bca_lo_pp']:+.2f}, "
        f"{secondary['bca_hi_pp']:+.2f}] pp\n"
        f"- Wilcoxon raw p = {secondary['p_raw']:.4g}, Holm p = {secondary['p_holm']:.4g}, "
        f"Cohen's d = {secondary['cohens_d']:+.2f} ({secondary['pos']}/{secondary['neg']} +/-)\n\n"
        f"At n = {n_paired} the effect size is the primary evidence and the p-value secondary "
        f"(the section 4.12 convention for the RF replication).\n\n"
        f"## FDR family contribution (plan section 6.3)\n\n"
        f"Two new paired Wilcoxon tests: primary raw p = {primary['p_raw']:.4g}, "
        f"secondary raw p = {secondary['p_raw']:.4g}.\n"
    )
    print(f"\nwrote {OUT/'p3_external_cd_tests.csv'}, {OUT/'p3_verdict.md'}, {OUT/'p3_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
