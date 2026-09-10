#!/usr/bin/env python3
"""
b4_regen_ladder.py
==================
B4.1 of EXPERIMENT_PLAN_AUDIT_REMEDIATION.md. The three derived alignment-ladder
files in results_alignment_ladder_loso/ carry f1_macro_mean for rungs 0 and 3
only; rungs 1, 2, 4 are blank, and the summary/stats files hold only those two
rungs and the one contrast. The five per-subject files are complete and their
means reproduce Table 4.16 exactly. Regenerate the three derived files, complete
for all five rungs, verifying against Table 4.16 BEFORE overwriting, with the
originals backed up.

No thesis file edited. New paired Wilcoxon contrasts are reported for the FDR
family; Section 4.17 is not touched.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
LADDER = ROOT / "results_alignment_ladder_loso"
ARCHIVE = ROOT.parent / "_ARCHIVE" / "prebackups" / f"b4_ladder_{datetime.now(timezone.utc):%Y%m%d}"

TABLE_4_16 = {0: 0.7094, 1: 0.7482, 2: 0.7186, 3: 0.7767, 4: 0.6752}
RUNG_NAMES = {0: "global_z", 1: "mean_center", 2: "scale_only", 3: "mean_scale",
              4: "full_whiten_recolor"}
TOL = 0.001


def load_persubject() -> dict[int, pd.Series]:
    out = {}
    for r in range(5):
        p = LADDER / f"ladder_loso_{r}_SVM_subjectwise.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        df = pd.read_csv(p)
        out[r] = df.set_index("subject")["f1_macro"].sort_index()
    return out


def main() -> int:
    ps = load_persubject()

    print("--- verify per-subject means reproduce Table 4.16 BEFORE any overwrite ---")
    ok = True
    for r in range(5):
        m = ps[r].mean()
        d = m - TABLE_4_16[r]
        flag = abs(d) <= TOL
        ok &= flag
        print(f"  rung {r} {RUNG_NAMES[r]:22} mean {m:.4f}  vs Table 4.16 {TABLE_4_16[r]:.4f}  "
              f"({d:+.4f})  {'OK' if flag else 'MISMATCH'}")
    if not ok:
        sys.exit("per-subject means do not match Table 4.16; NOT overwriting anything. Investigate.")
    idx0 = ps[0].index
    if not all(ps[r].index.equals(idx0) for r in range(5)):
        sys.exit("per-subject files do not share a subject index; NOT overwriting.")
    print("  all five rungs reproduce Table 4.16 to within 0.001; safe to regenerate.")

    # ---- back up originals ----
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for fn in ("alignment_ladder_full.csv", "alignment_ladder_loso_summary.csv",
               "alignment_ladder_loso_stats.csv"):
        src = LADDER / fn
        if src.exists():
            shutil.copy2(src, ARCHIVE / fn)
    print(f"  originals backed up to {ARCHIVE}")

    # ---- regenerate full.csv: keep geometry cols, fill f1_macro_mean for all 5 ----
    full = pd.read_csv(LADDER / "alignment_ladder_full.csv")
    full["f1_macro_mean"] = full["rung"].map(lambda r: round(ps[int(r)].mean(), 6))
    full.to_csv(LADDER / "alignment_ladder_full.csv", index=False)

    # ---- regenerate summary.csv: all 5 rungs ----
    summ = pd.DataFrame([{"rung": r, "name": RUNG_NAMES[r],
                          "f1_mean": round(ps[r].mean(), 6),
                          "f1_sd": round(ps[r].std(ddof=1), 6), "n": len(ps[r])}
                         for r in range(5)])
    summ.to_csv(LADDER / "alignment_ladder_loso_summary.csv", index=False)

    # ---- regenerate stats.csv: rung{1,2,3,4}_vs_rung0 and rung4_vs_rung3 ----
    def contrast(hi: int, lo: int) -> dict:
        a, b = ps[hi].to_numpy(), ps[lo].to_numpy()
        w = stats.wilcoxon(a, b)
        diff = a - b
        dz = diff.mean() / diff.std(ddof=1)
        return {"comparison": f"rung{hi}_vs_rung{lo}", "n": len(a),
                "mean_diff": round(float(diff.mean()), 6),
                "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
                "cohens_d_paired": round(float(dz), 4)}
    contrasts = [contrast(1, 0), contrast(2, 0), contrast(3, 0), contrast(4, 0), contrast(4, 3)]
    pd.DataFrame(contrasts).to_csv(LADDER / "alignment_ladder_loso_stats.csv", index=False)

    print("\n--- regenerated ---")
    print(pd.read_csv(LADDER / "alignment_ladder_loso_summary.csv").to_string(index=False))
    print()
    print(pd.read_csv(LADDER / "alignment_ladder_loso_stats.csv").to_string(index=False))

    print("\n--- new paired Wilcoxon contrasts for the FDR family (raw p; Section 4.17 not touched) ---")
    prior = {"rung3_vs_rung0"}   # already in the family
    for c in contrasts:
        tag = "(already in family)" if c["comparison"] in prior else "(NEW)"
        print(f"  {c['comparison']:<16} raw p = {c['wilcoxon_p']:.4g}  d = {c['cohens_d_paired']:+.3f}  {tag}")

    (LADDER / "B4_REGEN_NOTE.md").write_text(
        f"# B4.1 regeneration note ({datetime.now(timezone.utc):%Y-%m-%d})\n\n"
        f"alignment_ladder_full.csv, alignment_ladder_loso_summary.csv and "
        f"alignment_ladder_loso_stats.csv were regenerated from the five complete "
        f"ladder_loso_{{0..4}}_SVM_subjectwise.csv files. The per-subject means reproduce "
        f"Table 4.16 (0.7094 / 0.7482 / 0.7186 / 0.7767 / 0.6752) to within 0.001. Originals "
        f"are in `_ARCHIVE/prebackups/{ARCHIVE.name}/`. The geometry columns "
        f"(mmd_removed_pct, w1_removed_pct, subject_probe_bal_acc, silhouette_by_class) were "
        f"not touched. New paired Wilcoxon contrasts rung1/2/4_vs_rung0 and rung4_vs_rung3 are "
        f"reported for the FDR recompute; Section 4.17 was not edited.\n"
    )
    print(f"\nwrote {LADDER/'B4_REGEN_NOTE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
