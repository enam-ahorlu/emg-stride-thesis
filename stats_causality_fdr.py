# stats_causality_fdr.py
# ---------------------------------------------------------------------------
# Emits the PRC 2026 causality experiments (E-C1, E-C3, E-C4) into the schema
# stats_unified_fdr.py consumes: comparison, delta_pp, cohens_d, p.
#
# E-C2 contributes no tests: it is a descriptive robustness check on one rung's
# regulariser, with no paired comparison to make.
#
# Wilcoxon p-values are taken from the CSVs the experiments already wrote, so
# no test is re-run here. Paired Cohen's d is computed where the source CSV
# did not carry it.
# ---------------------------------------------------------------------------
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)

def dz(a, b):
    """Paired Cohen's d (d_z) for a - b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean() / d.std(ddof=1))

rows = []

# ---- E-C1: alignment-ladder downstream F1 --------------------------------
lad = pd.read_csv(ROOT / "results_alignment_ladder_loso" / "alignment_ladder_loso_stats.csv")
label = {"rung4_vs_rung3": "Full whitening vs per-subject z-score (alignment ladder, LOSO F1)",
         "rung3_vs_rung0": "Per-subject z-score vs global z-score (alignment ladder, LOSO F1)"}
for _, r in lad.iterrows():
    rows.append(dict(comparison=label.get(r["comparison"], r["comparison"]),
                     delta_pp=round(float(r["mean_diff"]) * 100, 1),
                     cohens_d=round(float(r["cohens_d_paired"]), 3),
                     p=float(r["wilcoxon_p"])))

# ---- E-C3: calibration-buffer composition, each mode vs mixed100 ----------
bc = pd.read_csv(ROOT / "results_buffer_composition" / "buffer_composition_subjectwise.csv")
wx = pd.read_csv(ROOT / "results_buffer_composition" / "buffer_composition_wilcoxon.csv")
piv = bc.pivot_table(index=["subject", "model"], columns="mode", values="f1_excl").reset_index()
pretty = {"soft": "soft ensemble", "SVM": "SVM", "SVM_PROBA": "SVM (probability)", "RESNET_SE": "ResNet-SE+CD"}
for _, r in wx.iterrows():
    m, mode = r["model"], r["mode"]
    sub = piv[piv["model"] == m]
    if mode not in sub.columns or "mixed100" not in sub.columns:
        raise SystemExit(f"missing column for {m}/{mode}")
    a, b = sub[mode].to_numpy(float), sub["mixed100"].to_numpy(float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    p = float(r["wilcoxon_p"])
    p = max(p, 1e-12)                      # CSV stores underflow as 0.0
    rows.append(dict(comparison=f"Calibration buffer {mode} vs mixed100 [{pretty.get(m, m)}]",
                     delta_pp=round(float(r["mean_delta_vs_mixed"]) * 100, 1),
                     cohens_d=round(dz(a, b), 3), p=p))

# ---- E-C4: CORAL regulariser sweep vs per-subject z-score -----------------
cw = pd.read_csv(ROOT / "results_coral_lam_sweep" / "coral_lam_sweep_vs_persubj_wilcoxon.csv")
ps = pd.read_csv(next((ROOT / "results_loso_freq_persubj").glob("*SVM_nested_loso_subjectwise.csv")))
ps = ps.sort_values("heldout_subject")["f1_macro"].to_numpy(float)
for _, r in cw.iterrows():
    lam = r["comparison"].split("_vs_")[0].replace("coral_lam", "")
    cs = pd.read_csv(next((ROOT / "results_coral_lam_sweep" / f"lam_{lam}").glob("*subjectwise.csv")))
    cs = cs.sort_values("subject")["f1_macro"].to_numpy(float)
    rows.append(dict(comparison=f"CORAL (lambda = {lam}) vs per-subject z-score [SVM]",
                     delta_pp=round(float(r["mean_diff"]) * 100, 1),
                     cohens_d=round(dz(cs, ps), 3), p=float(r["wilcoxon_p"])))

df = pd.DataFrame(rows)
df.to_csv(OUT / "causality_stats_fdr.csv", index=False)
print(f"wrote {OUT / 'causality_stats_fdr.csv'}  ({len(df)} tests)")
print(df.to_string(index=False))