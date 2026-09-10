#!/usr/bin/env python3
"""
w5_depth_capacity_stats.py  --  W-5: separating depth from capacity.

Per-arm channel-dropout gain (chandrop - noaug, paired over 40 subjects), then
the two interaction contrasts, each formed per subject:

  I_shallow = BASE_gain - SHALLOW_gain
  I_narrow  = BASE_gain - NARROW_gain

Holm across the family of two interactions; the per-arm gains sit beside it.
Grid from EXPERIMENT_PLAN_DEPTH_VS_CAPACITY.md §5, with explicit cells for
N, R and X. Estimators reused from window_ablation_stats.py (verified in W-1).

Reads only. Writes results_w5/.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

from window_ablation_stats import bca_ci, cohens_d_paired, holm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_w5"; OUT.mkdir(exist_ok=True)

ARMS = {
    "base_noaug":     "results_w3_nores_noaug",      # reused (W-3)
    "base_chandrop":  "results_w3_nores_chandrop",   # reused (W-3)
    "shallow_noaug":  "results_w5_shallow_noaug",
    "shallow_chandrop": "results_w5_shallow_chandrop",
    "narrow_noaug":   "results_w5_narrow_noaug",
    "narrow_chandrop": "results_w5_narrow_chandrop",
}
BASE_NOAUG_REF = 0.7718        # §5 X-threshold anchor
TRAIN_DROP = 3.0              # pp below BASE_NOAUG_REF that trips outcome X
LOSE_FLOOR = 2.0             # pp point-estimate floor for "loses the effect"


def load(dirname: str) -> pd.Series:
    d = ROOT / dirname
    hits = sorted(d.glob("cnn_arch_subjectwise.csv"))
    if not hits:
        raise SystemExit(f"missing {d}/cnn_arch_subjectwise.csv -- has run_w5_depth_capacity.sh finished?")
    df = pd.read_csv(hits[0])
    if df["subject"].duplicated().any():
        raise SystemExit(f"{dirname}: duplicated subject rows; a --resume double-appended")
    if len(df) != 40:
        raise SystemExit(f"{dirname}: {len(df)} rows, expected 40")
    return df.set_index("subject")["f1_macro"].sort_index()


def onesample(vec: np.ndarray, label: str) -> dict:
    p = 1.0 if np.allclose(vec, 0) else float(
        stats.wilcoxon(vec, zero_method="wilcox", alternative="two-sided").pvalue)
    sd = vec.std(ddof=1)
    lo, hi = bca_ci(vec)
    return {"comparison": label, "delta_pp": float(vec.mean() * 100), "p_raw": p,
            "cohens_d": float(vec.mean() / sd) if sd > 0 else 0.0,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "n": len(vec)}


def paired(a, b, label) -> dict:
    d = (a - b).to_numpy()
    p = 1.0 if np.allclose(d, 0) else float(
        stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    lo, hi = bca_ci(d)
    return {"comparison": label, "delta_pp": float(d.mean() * 100), "p_raw": p,
            "cohens_d": cohens_d_paired(a.to_numpy(), b.to_numpy()),
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "n": len(a)}


def main() -> int:
    v = {k: load(d) for k, d in ARMS.items()}

    # per-subject gains
    g_base    = v["base_chandrop"]    - v["base_noaug"]
    g_shallow = v["shallow_chandrop"] - v["shallow_noaug"]
    g_narrow  = v["narrow_chandrop"]  - v["narrow_noaug"]

    gains = [
        onesample(g_base.to_numpy(),    "BASE gain (chandrop - noaug)"),
        onesample(g_shallow.to_numpy(), "SHALLOW-MATCHED gain"),
        onesample(g_narrow.to_numpy(),  "NARROW-MATCHED gain"),
    ]

    I_shallow = (g_base - g_shallow).to_numpy()
    I_narrow  = (g_base - g_narrow).to_numpy()
    fam = [onesample(I_shallow, "interaction: BASE - SHALLOW"),
           onesample(I_narrow,  "interaction: BASE - NARROW")]
    holm(fam)

    pd.DataFrame(gains + fam).to_csv(OUT / "w5_depth_capacity_tests.csv", index=False)
    pd.DataFrame({k: v[k] for k in ARMS}).to_csv(OUT / "w5_depth_capacity_pairs.csv")

    # --- trainability (X) ---
    base_n = float(v["base_noaug"].mean())
    sh_n   = float(v["shallow_noaug"].mean())
    na_n   = float(v["narrow_noaug"].mean())
    x_arms = []
    for name, m in (("SHALLOW-MATCHED", sh_n), ("NARROW-MATCHED", na_n)):
        if (BASE_NOAUG_REF - m) * 100 > TRAIN_DROP:
            x_arms.append((name, m))

    def loses(row):
        return row["p_holm"] < 0.05 and row["delta_pp"] >= LOSE_FLOOR

    sh_row = fam[0]; na_row = fam[1]
    sh_lose, na_lose = loses(sh_row), loses(na_row)
    # reversal: a reduced arm's gain LARGER than BASE by Holm-sig >= 2.0 pp
    reversal = [r["comparison"] for r in fam
               if r["p_holm"] < 0.05 and r["delta_pp"] <= -LOSE_FLOOR]

    if x_arms:
        letter = "X"
        meaning = ("The measure broke for " + ", ".join(n for n, _ in x_arms) +
                   f" (no-aug baseline >{TRAIN_DROP} pp below BASE {BASE_NOAUG_REF}). "
                   "That arm did not train comparably; its interaction is not read.")
    elif reversal:
        letter = "R"
        meaning = ("Reversal: " + "; ".join(reversal) + " shows a channel-dropout gain "
                   "LARGER than BASE by a Holm-significant >=2.0 pp margin. Not anticipated. "
                   "Report the numbers, force no letter, escalate before writing.")
    elif sh_lose and not na_lose:
        letter, meaning = "D", "Depth. Cutting depth at matched capacity costs the augmentation; cutting capacity does not."
    elif na_lose and not sh_lose:
        letter, meaning = "C", "Capacity. Cutting capacity at matched depth costs the augmentation; cutting depth does not."
    elif sh_lose and na_lose:
        letter, meaning = "B", "Both. Depth and capacity are jointly required at this scale."
    else:
        letter, meaning = "N", ("Neither. The +4.24 pp step is not attributable to depth or capacity "
                                "within this range; kernel width and stem remain untested. Informative negative.")

    lines = [f"# W-5 depth vs capacity: verdict\n", f"## Outcome {letter}\n", meaning, "",
             "## Per-arm channel-dropout gain (chandrop - noaug, paired, n=40)"]
    for r in gains:
        lines.append(f"- {r['comparison']}: {r['delta_pp']:+.2f} pp "
                     f"(p = {r['p_raw']:.4g}, d = {r['cohens_d']:.2f}, "
                     f"95% BCa [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] pp)")
    lines += ["", "## Interaction contrasts (Holm across the family of two)"]
    for r in fam:
        lines.append(f"- {r['comparison']}: {r['delta_pp']:+.2f} pp "
                     f"(raw p = {r['p_raw']:.4g}, Holm p = {r['p_holm']:.4g}, d = {r['cohens_d']:.2f}, "
                     f"95% BCa [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] pp) "
                     f"-> {'LOSES the effect' if loses(r) else 'does not lose the effect'}")
    lines += ["", "## No-augmentation baselines (trainability, §5 X-check)",
              f"- BASE {base_n:.4f} (ref {BASE_NOAUG_REF})",
              f"- SHALLOW-MATCHED {sh_n:.4f}  ({(sh_n-BASE_NOAUG_REF)*100:+.2f} pp vs BASE)",
              f"- NARROW-MATCHED {na_n:.4f}  ({(na_n-BASE_NOAUG_REF)*100:+.2f} pp vs BASE)"]

    (OUT / "w5_verdict.md").write_text("\n".join(lines), encoding="utf8")
    (OUT / "w5_outcome.json").write_text(json.dumps(
        {"outcome": letter,
         "gain_base_pp": gains[0]["delta_pp"], "gain_shallow_pp": gains[1]["delta_pp"],
         "gain_narrow_pp": gains[2]["delta_pp"],
         "I_shallow_pp": sh_row["delta_pp"], "I_shallow_holm_p": sh_row["p_holm"],
         "I_narrow_pp": na_row["delta_pp"], "I_narrow_holm_p": na_row["p_holm"],
         "baselines": {"base": base_n, "shallow": sh_n, "narrow": na_n}}, indent=2), encoding="utf8")

    print("\n".join(lines))
    print(f"\nwrote {OUT}/w5_verdict.md, w5_outcome.json, w5_depth_capacity_tests.csv, w5_depth_capacity_pairs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
