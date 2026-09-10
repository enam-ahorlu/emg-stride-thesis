#!/usr/bin/env python3
"""
w2_g1_stats.py  --  Stage G1 statistics and the §1.4 gate.

interaction_s = (ResNet-SE+CD_s - ResNet-SE_s) - (ResNet+CD_s - ResNet_s)
computed per subject, then paired across the 40 (§6). The ResNet+CD gain
(CD - base, paired) drives the §1.4 gate; the interaction's significance and
sign decide S vs R.

Baseline: the FRESH plain-ResNet run (results_cd_resnet_noaug_repro), per the
1 Sep revision -- both differences must be era-internal. The OLD baseline
(results_cnn_loso_resnet) is also reported as a sensitivity check.

Reuses window_ablation_stats.py for BCa (10k, seed 42) and Holm.
Reads only. Writes results_w2_g1/.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

from window_ablation_stats import load_arm, bca_ci, cohens_d_paired, holm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_w2_g1"
OUT.mkdir(exist_ok=True)

ARMS = {
    "ResNet-SE+CD": "results_cnn_aug_resnet_se_chandrop",   # old era, reused
    "ResNet-SE":    "results_cnn_loso_resnet_se",            # old era, reused
    "ResNet+CD":    "results_cd_resnet_nose_chandrop",       # fresh, G1
    "ResNet(fresh)":"results_cd_resnet_noaug_repro",         # fresh baseline
    "ResNet(old)":  "results_cnn_loso_resnet",               # old baseline, reused
}
GATES = {  # §7 reused-arm gates, read from CSV, ±0.002
    "SimpleEMGCNN, no aug":      ("results_cnn_loso_simple_repro", 0.7602),
    "ResNet, no SE, no aug":     ("results_cnn_loso_resnet",       0.7563),
    "ResNet-SE, no aug":         ("results_cnn_loso_resnet_se",    0.7822),
    "ResNet-SE + CD, p=0.2":     ("results_cnn_aug_resnet_se_chandrop", 0.8395),
}


def one_sample(vec: np.ndarray, label: str) -> dict:
    p = 1.0 if np.allclose(vec, 0) else float(
        stats.wilcoxon(vec, zero_method="wilcox", alternative="two-sided").pvalue)
    sd = vec.std(ddof=1)
    lo, hi = bca_ci(vec)
    return {"comparison": label, "delta_pp": float(vec.mean() * 100), "p_raw": p,
            "cohens_d": float(vec.mean() / sd) if sd > 0 else 0.0,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "n": len(vec)}


def paired(a, b, label) -> dict:
    d = a - b
    p = 1.0 if np.allclose(d, 0) else float(
        stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    lo, hi = bca_ci(d)
    return {"comparison": label, "delta_pp": float(d.mean() * 100), "p_raw": p,
            "cohens_d": cohens_d_paired(a, b), "ci_low_pp": lo * 100,
            "ci_high_pp": hi * 100, "n": len(a)}


def main() -> int:
    print("=== §7 reproduction gates (read from CSV, ±0.002) ===")
    gate_ok = True
    for name, (d, exp) in GATES.items():
        got = float(load_arm(d, "resnet").mean())
        ok = abs(got - exp) <= 0.002
        gate_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL':4}  {name:<26} got {got:.4f}  expected {exp:.4f}")
    v = {k: load_arm(d, "resnet") for k, d in ARMS.items()}
    print(f"\n  drift check: fresh plain-ResNet {v['ResNet(fresh)'].mean():.4f} vs "
          f"old {v['ResNet(old)'].mean():.4f}  "
          f"({(v['ResNet(fresh)'].mean()-v['ResNet(old)'].mean())*100:+.2f} pp)")
    if not gate_ok:
        print("\nA §7 gate failed. Stop and report; do not adjust expected values.")
        return 3

    rows = []
    for base_label, base in (("fresh", v["ResNet(fresh)"]), ("old", v["ResNet(old)"])):
        inter = (v["ResNet-SE+CD"] - v["ResNet-SE"]) - (v["ResNet+CD"] - base)
        fam = [
            one_sample(inter, f"interaction contrast [base={base_label}]"),
            paired(v["ResNet+CD"], base, f"ResNet+CD gain (CD - base) [base={base_label}]"),
            paired(v["ResNet-SE+CD"], v["ResNet-SE"], f"ResNet-SE+CD gain (SE+CD - SE) [base={base_label}]"),
        ]
        holm(fam)
        rows += fam

    pd.DataFrame(rows).to_csv(OUT / "w2_g1_tests.csv", index=False)

    # --- §1.4 gate, on the FRESH baseline ---
    inter_fresh = next(r for r in rows if r["comparison"] == "interaction contrast [base=fresh]")
    gain_fresh = next(r for r in rows if r["comparison"].startswith("ResNet+CD gain") and "fresh" in r["comparison"])
    gain_pp = gain_fresh["delta_pp"]
    inter_sig = inter_fresh["p_holm"] < 0.05
    inter_pos = inter_fresh["delta_pp"] > 0

    if gain_pp < 2.0 and inter_sig and inter_pos:
        letter, meaning = "S", "SE-dependent. §4.8's claim holds. Gate OPEN -> G2-G4."
    elif 2.0 <= gain_pp < 4.0:
        letter, meaning = "M", ("Mixed. Both residual depth and SE contribute. Gate OPEN but "
                                "§4.8 must be rewritten to apportion before G2-G4.")
    elif gain_pp >= 4.0 and not inter_sig:
        letter, meaning = "R", ("Residual-dependent. SE is NOT the mechanism; §4.8's claim is "
                                "wrong and must be corrected. Gate CLOSED. Stop and report.")
    else:
        letter, meaning = "?", ("Does not fit S/M/R cleanly (e.g. gain>=4 but interaction "
                                "significant, or gain<2 but interaction not significant+positive). "
                                "Report the numbers and escalate.")

    params = {"resnet": 546020, "resnet_se": 557276}
    lines = [
        f"# W-2 Stage G1 verdict\n",
        f"## Outcome {letter}\n", meaning, "",
        f"ResNet+CD gain (fresh baseline): {gain_pp:+.2f} pp "
        f"(Holm p = {gain_fresh['p_holm']:.4g}, d = {gain_fresh['cohens_d']:.2f}, "
        f"95% CI [{gain_fresh['ci_low_pp']:+.2f}, {gain_fresh['ci_high_pp']:+.2f}] pp)",
        f"Interaction contrast (fresh):    {inter_fresh['delta_pp']:+.2f} pp "
        f"(Holm p = {inter_fresh['p_holm']:.4g}, d = {inter_fresh['cohens_d']:.2f}, "
        f"95% CI [{inter_fresh['ci_low_pp']:+.2f}, {inter_fresh['ci_high_pp']:+.2f}] pp), "
        f"{'significant' if inter_sig else 'not significant'}",
        "",
        "## Arm means (LOSO macro-F1, n=40)",
        f"- ResNet-SE+CD : {v['ResNet-SE+CD'].mean():.4f}  (old, reused)",
        f"- ResNet-SE    : {v['ResNet-SE'].mean():.4f}  (old, reused)",
        f"- ResNet+CD    : {v['ResNet+CD'].mean():.4f}  (fresh, G1)",
        f"- ResNet       : {v['ResNet(fresh)'].mean():.4f}  (fresh baseline)  |  old {v['ResNet(old)'].mean():.4f}",
        "",
        f"## Parameter counts: resnet {params['resnet']:,}  |  resnet_se {params['resnet_se']:,}  "
        f"(SE adds {params['resnet_se']-params['resnet']:,}; comparison is not capacity-matched)",
        "",
        "## Sensitivity: same contrasts on the OLD baseline",
    ]
    for r in rows:
        if "base=old" in r["comparison"]:
            lines.append(f"- {r['comparison']}: {r['delta_pp']:+.2f} pp "
                         f"(Holm p = {r['p_holm']:.4g}, d = {r['cohens_d']:.2f})")

    (OUT / "w2_g1_verdict.md").write_text("\n".join(lines), encoding="utf8")
    (OUT / "w2_g1_outcome.json").write_text(json.dumps(
        {"outcome": letter, "gain_pp_fresh": gain_pp,
         "interaction_pp_fresh": inter_fresh["delta_pp"],
         "interaction_p_holm_fresh": inter_fresh["p_holm"],
         "interaction_significant": bool(inter_sig)}, indent=2), encoding="utf8")

    print("\n" + "\n".join(lines))
    print(f"\nwrote {OUT}/w2_g1_tests.csv, w2_g1_verdict.md, w2_g1_outcome.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
