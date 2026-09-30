#!/usr/bin/env python3
"""
src/kc23_d1_mixedlm.py
====================
The KC-D1 secondary model (docs/plans/EXPERIMENT_PLAN_KC23_DEEP.md D1.5 item 3): F1 ~ arm + (1 | subject) + (1 | seed), statsmodels MixedLM,
fixed effect and interval, for every two-arm registered contrast. Run under 06_Code/.venv_stats (statsmodels lives ONLY there;
06_Code/.venv stays frozen for reproduction): src/kc23_d1_replicate_stats.py starts this script with that interpreter when its own
environment has no statsmodels. This file imports only pandas and statsmodels and reads and writes CSV, so it is independent of
the Python version of the caller.

  python src/kc23_d1_mixedlm.py --in d1_arm_subject_f1.csv --out D1_secondary_mixedlm.csv

Input: d1_arm_subject_f1.csv (arm, realization, subject, f1). The published runs (realization "published") are a sensitivity
analysis and are excluded, as in every D1 letter. Output: one row per contrast (contrast, arm_a, arm_b, fixed_effect_pp, lo_pp,
hi_pp, n_obs, status); a contrast whose fit fails gets NaN and the error in `status`, never a silent omission. Exit 1 and no
output file when the input is unusable.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

TWO_ARM_CONTRASTS = {"C1": ("R2", "R1"), "C2": ("R3", "R2"), "C3": ("R14", "R13"), "C4": ("R16", "R15"),
                     "C5": ("R1", "R4"), "C8": ("R4", "R6"), "C9": ("R2", "R10"), "C10": ("R11", "R10"),
                     "C11": ("R2", "R11"), "C13": ("ENS_SOFT", "R2"), "C14": ("R12", "R2"), "C15": ("R9", "R8"),
                     "C16a": ("R13", "R2"), "C16b": ("R17", "R2"), "C16c": ("R15", "R2")}


def fit_all(af: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    d = af[af["realization"].astype(str) != "published"].copy()
    d["seed"] = d["realization"].astype(str)
    rows = []
    for cid, (a, b) in TWO_ARM_CONTRASTS.items():
        sub = d[d["arm"].isin([a, b])].copy()
        row = {"contrast": cid, "arm_a": a, "arm_b": b, "n_obs": int(len(sub))}
        try:
            if sub["arm"].nunique() != 2:
                raise ValueError("one of the two arms is absent")
            sub["is_a"] = (sub["arm"] == a).astype(float)
            sub["one"] = 1
            md = smf.mixedlm("f1 ~ is_a", sub, groups=sub["one"],
                             vc_formula={"subject": "0 + C(subject)", "seed": "0 + C(seed)"}).fit(reml=True)
            ci = md.conf_int().loc["is_a"]
            row.update(fixed_effect_pp=float(md.params["is_a"] * 100), lo_pp=float(ci.iloc[0] * 100),
                       hi_pp=float(ci.iloc[1] * 100), status="ok")
        except Exception as e:                       # the SECONDARY model must not halt the stage; say so
            row.update(fixed_effect_pp=float("nan"), lo_pp=float("nan"), hi_pp=float("nan"),
                       status=f"fit failed: {type(e).__name__}: {e}")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.unlink(missing_ok=True)
    try:
        af = pd.read_csv(args.inp)
        need = {"arm", "realization", "subject", "f1"}
        if not need <= set(af.columns) or af.empty:
            raise ValueError(f"{args.inp} lacks {sorted(need - set(af.columns))} or is empty")
        res = fit_all(af)
    except Exception as e:
        print(f"[d1-mixedlm] FAIL, no output written: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    res.to_csv(out, index=False)
    import statsmodels
    print(f"[d1-mixedlm] statsmodels {statsmodels.__version__}: {int((res['status'] == 'ok').sum())} of {len(res)} contrasts fitted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
