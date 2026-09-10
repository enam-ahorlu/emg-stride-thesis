#!/usr/bin/env python3
"""
p6_extension_stats.py
=====================
Stage P-6 optional extension (EXPERIMENT_PLAN_CD_PARITY.md section 5B.3). Two
curves over one shared multiplicative-variance axis on resnet_se, per-subject
norm, 250 ms:

  channel dropout (Bernoulli, form = zeroing): the existing four-rate sweep
    results_cd_rate_p0.1 / p0.2 / p0.3 / p0.5, multiplicative SD = sqrt(p(1-p))
  gain jitter (uniform, form = continuous): SD in {0.30, 0.40, 0.50}
    results_p6_gainjitter_resnet_se_sd0.30 / sd0.40 / sd0.50

sqrt(p(1-p)) gives 0.30 at p = 0.1, 0.40 at p = 0.2, 0.50 at p = 0.5, so the
three matched points are (p0.1, sd0.30), (p0.2, sd0.40), (p0.5, sd0.50).

If the two curves superimpose, the injected per-channel multiplicative variance
is the operative quantity and the form is irrelevant. If they diverge, where
they diverge says how the form matters. 2.0 pp materiality floor from W-5.

Estimators from window_ablation_stats.py.
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
from window_ablation_stats import bca_ci, cohens_d_paired

OUT = ROOT / "results_parity"
OUT.mkdir(exist_ok=True)
FLOOR_PP = 2.0

# matched points: (multiplicative SD, chandrop dir, gainjitter dir)
POINTS = [
    (0.30, "results_cd_rate_p0.1", "results_p6_gainjitter_resnet_se_sd0.30"),
    (0.40, "results_cd_rate_p0.2", "results_p6_gainjitter_resnet_se_sd0.40"),
    (0.50, "results_cd_rate_p0.5", "results_p6_gainjitter_resnet_se_sd0.50"),
]
# full chandrop sweep for the plotted curve
CD_SWEEP = [(0.1, 0.300, "results_cd_rate_p0.1"), (0.2, 0.400, "results_cd_rate_p0.2"),
            (0.3, 0.458, "results_cd_rate_p0.3"), (0.5, 0.500, "results_cd_rate_p0.5")]
GJ_SWEEP = [(0.30, "results_p6_gainjitter_resnet_se_sd0.30"),
            (0.40, "results_p6_gainjitter_resnet_se_sd0.40"),
            (0.50, "results_p6_gainjitter_resnet_se_sd0.50")]
NOAUG = "results_cnn_aug_resnet_se_none"


def load(dirname: str) -> pd.Series:
    p = ROOT / dirname / "cnn_arch_subjectwise.csv"
    if not p.exists():
        sys.exit(f"missing {p}")
    df = pd.read_csv(p)
    sc = "subject" if "subject" in df.columns else "heldout_subject"
    df = df[[sc, "f1_macro"]].dropna().rename(columns={sc: "subject"})
    if df["subject"].duplicated().any():
        sys.exit(f"{p} has duplicated subjects; a --resume double-appended")
    if df["subject"].nunique() != 40:
        sys.exit(f"{p} has {df['subject'].nunique()} subjects, expected 40")
    return df.set_index("subject")["f1_macro"].sort_index().rename(dirname)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, run, out = len(ps), 0.0, [0.0] * len(ps)
    for k, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (m - k)))
        out[i] = run
    return out


def main() -> int:
    print("=" * 78)
    print("STAGE P-6 EXTENSION: variance axis, channel dropout vs gain jitter")
    print("=" * 78)

    na = load(NOAUG)
    print(f"\nno-augmentation baseline (resnet_se): {na.mean():.4f}")

    # ---- plotted curves ----
    print("\n--- channel dropout sweep (form = Bernoulli zeroing) ---")
    cd_curve = []
    for p, sd, d in CD_SWEEP:
        s = load(d)
        cd_curve.append((sd, s.mean(), s))
        print(f"  p = {p}  SD = {sd:.3f}   mean F1 {s.mean():.4f}")
    print("\n--- gain jitter sweep (form = uniform continuous) ---")
    gj_curve = []
    for sd, d in GJ_SWEEP:
        s = load(d)
        gj_curve.append((sd, s.mean(), s))
        print(f"  SD = {sd:.3f}            mean F1 {s.mean():.4f}")

    # ---- matched-point paired contrasts (gain jitter - channel dropout) ----
    print("\n--- matched-variance paired contrasts (gain jitter minus channel dropout, n=40) ---")
    rows, praw = [], []
    for sd, cd_dir, gj_dir in POINTS:
        cd_s, gj_s = load(cd_dir), load(gj_dir)
        if not cd_s.index.equals(gj_s.index):
            sys.exit(f"subject mismatch at SD {sd}")
        dd = (gj_s - cd_s).to_numpy()
        lo, hi = bca_ci(dd)
        w = stats.wilcoxon(gj_s.to_numpy(), cd_s.to_numpy())
        dz = cohens_d_paired(gj_s.to_numpy(), cd_s.to_numpy())
        praw.append(float(w.pvalue))
        rows.append({"sd": sd, "cd_dir": cd_dir, "gj_dir": gj_dir,
                     "cd_mean": float(cd_s.mean()), "gj_mean": float(gj_s.mean()),
                     "delta_pp": float(dd.mean() * 100), "lo_pp": lo * 100, "hi_pp": hi * 100,
                     "p_raw": float(w.pvalue), "d": dz,
                     "pos": int((dd > 0).sum()), "neg": int((dd < 0).sum())})
    ph = holm(praw)
    for r, hp in zip(rows, ph):
        r["p_holm"] = hp
        print(f"  SD {r['sd']:.2f}: chandrop {r['cd_mean']:.4f}  gain jitter {r['gj_mean']:.4f}  "
              f"delta {r['delta_pp']:+.2f} pp  95% BCa [{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}]  "
              f"raw p = {r['p_raw']:.4g}  Holm p = {hp:.4g}  d = {r['d']:+.2f}  "
              f"({r['pos']}/{r['neg']} +/-)")

    max_abs = max(abs(r["delta_pp"]) for r in rows)
    any_material_sig = any(r["p_holm"] < 0.05 and abs(r["delta_pp"]) >= FLOOR_PP for r in rows)
    any_sig = any(r["p_holm"] < 0.05 for r in rows)

    if not any_sig and max_abs < FLOOR_PP:
        verdict = ("SUPERIMPOSE. No matched-variance contrast is Holm-significant and the largest "
                   f"gap is {max_abs:.2f} pp, below the 2.0 pp floor. Over the shared variance "
                   "axis the two forms are interchangeable: the injected per-channel "
                   "multiplicative variance is the operative quantity and the form of the "
                   "perturbation is not. This is the extension's clean result and reinforces P-6 "
                   "outcome V.")
        letter = "SUPERIMPOSE"
    elif any_material_sig:
        where = ", ".join(f"SD {r['sd']:.2f} ({r['delta_pp']:+.2f} pp)"
                          for r in rows if r["p_holm"] < 0.05 and abs(r["delta_pp"]) >= FLOOR_PP)
        verdict = (f"DIVERGE. The curves separate materially at {where}. Where they diverge says "
                   "how the form matters; report the location and direction, do not average it "
                   "away.")
        letter = "DIVERGE"
    else:
        where = ", ".join(f"SD {r['sd']:.2f} ({r['delta_pp']:+.2f} pp, Holm p = {r['p_holm']:.3g})"
                          for r in rows if r["p_holm"] < 0.05)
        verdict = (f"MOSTLY SUPERIMPOSE. A statistically detectable but sub-2.0 pp separation "
                   f"appears at {where}; the largest gap is {max_abs:.2f} pp. Variance is the "
                   "operative quantity; the form contributes a small, immaterial edge. Consistent "
                   "with P-6 outcome V.")
        letter = "MOSTLY_SUPERIMPOSE"
    print("\n" + "=" * 78)
    print(f"P-6 EXTENSION: {verdict}")
    print("=" * 78)

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot([sd for sd, _, _ in cd_curve], [m * 100 for _, m, _ in cd_curve],
                "o-", label="channel dropout (Bernoulli, zeroing)", color="#1f77b4")
        ax.plot([sd for sd, _, _ in gj_curve], [m * 100 for _, m, _ in gj_curve],
                "s--", label="gain jitter (uniform, continuous)", color="#d62728")
        ax.axhline(na.mean() * 100, color="0.6", lw=1, ls=":", label="no augmentation")
        ax.set_xlabel("per-channel multiplicative SD")
        ax.set_ylabel("LOSO macro-F1 (%)")
        ax.set_title("P-6 extension: variance axis, resnet_se")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(OUT / "p6_extension_variance_axis.png", dpi=150)
        print(f"\nwrote {OUT/'p6_extension_variance_axis.png'}")
    except Exception as e:
        print(f"  (figure skipped: {e})")

    pd.DataFrame(rows).round(5).to_csv(OUT / "p6_extension_tests.csv", index=False)
    res = {"stage": "P-6 extension", "verdict": letter, "verdict_text": verdict,
           "noaug_mean": float(na.mean()),
           "chandrop_curve": [{"sd": sd, "mean_f1": m} for sd, m, _ in cd_curve],
           "gainjitter_curve": [{"sd": sd, "mean_f1": m} for sd, m, _ in gj_curve],
           "matched_points": rows}
    json.dump(res, open(OUT / "p6_extension_outcome.json", "w"), indent=2)

    (OUT / "p6_extension_verdict.md").write_text(
        f"# Stage P-6 extension: {letter}\n\n{verdict}\n\n"
        f"## Curves (resnet_se, per-subject norm, 250 ms), mean LOSO macro-F1\n\n"
        f"| multiplicative SD | channel dropout (Bernoulli) | gain jitter (uniform) |\n|---|---|---|\n"
        + "".join(
            f"| {sd:.3f} | {cm:.4f} | {gm} |\n"
            for (sd, cm, _), gm in [
                ((cd_curve[i][0], cd_curve[i][1], None),
                 f"{next((g[1] for g in gj_curve if abs(g[0]-cd_curve[i][0])<1e-6), float('nan')):.4f}"
                 if any(abs(g[0]-cd_curve[i][0])<1e-6 for g in gj_curve) else "n/a")
                for i in range(len(cd_curve))])
        + f"\nno-augmentation baseline {na.mean():.4f}.\n\n"
        f"## Matched-variance paired contrasts (gain jitter minus channel dropout, Holm across 3)\n\n"
        + "".join(
            f"- SD {r['sd']:.2f}: {r['delta_pp']:+.2f} pp, 95% BCa [{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}] pp, "
            f"raw p = {r['p_raw']:.4g}, Holm p = {r['p_holm']:.4g}, d = {r['d']:+.2f}\n" for r in rows)
        + f"\nLargest gap {max_abs:.2f} pp against a 2.0 pp materiality floor.\n\n"
        f"## FDR family contribution\n\n"
        f"Three new paired Wilcoxon tests (matched-variance contrasts): "
        + ", ".join(f"SD {r['sd']:.2f} raw p = {r['p_raw']:.4g}" for r in rows)
        + ". Section 4.17 not edited.\n"
    )
    print(f"wrote {OUT/'p6_extension_tests.csv'}, {OUT/'p6_extension_verdict.md'}, "
          f"{OUT/'p6_extension_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
