#!/usr/bin/env python3
"""
src/kc23_d1_replicate_stats.py
=============================
KC-D1 stats/gate. docs/plans/EXPERIMENT_PLAN_KC23_DEEP.md "KC-D1. The replicate
programme", D1.4 to D1.7. The long pole of the programme: the reproduction
gate, the registered contrasts (C1-C17), the KC-D6-adjacent descriptive
stacking row C13b (decision D-6a, docs/kc23/KC23_HALT.md), and the headline gate.

D1.4 reproduction gate (seed 42 re-run only): R2 within +/-1.5pt of 0.8395,
R1 within +/-1.5pt of 0.782, R10 pre-adaptation within +/-1.5pt of 0.787 or
0.772. FAIL -> ESCALATE ("code drift since publication").

D1.5 per contrast:
  ESTABLISHED:    the realization-averaged Wilcoxon survives BH within the
                   KC-D1 family, AND the same sign holds in >= 4/5
                   realizations (Tier A) or 3/3 (Tier B)
  AMBIGUOUS:       exactly one of the two holds
  NOT ESTABLISHED: neither holds
Null contrasts (C15, the C16 plateau pairs) use TOST equivalence (+/-1.0pt)
instead of a two-sided difference test.

D1.6 headline gate: published R2/ensemble/R12/global values within the
realization mean +/- 2SD -> H1 (stays as version of record); outside -> H2
(ESCALATE).

D1.7: C1 or C12 landing NOT ESTABLISHED -> ESCALATE. Any other established
contrast landing AMBIGUOUS/NOT ESTABLISHED is reported, not halted.

C13b (DECISION D-6a, docs/kc23/KC23_HALT.md, 24 September 2026): a DESCRIPTIVE row
beside C13, comparing the published stacking combiner against the soft-vote
ensemble, read against the measured realization SD -- not a registered
hypothesis, no ESTABLISHED/AMBIGUOUS label, no gate.

Conformance pass, 26 September 2026 (docs/kc23/KC23_PREREG_CONFORMANCE.md). The decision
functions below (reproduction_gate, classify_contrast, classify_headline,
benjamini_hochberg, stacking_descriptive) are unchanged from 0ba3575. What changed
is the analysis around them:
  - BH runs over the FIXED registered family (kc23_d1_aggregate.REGISTERED_BH_FAMILY:
    C1-C14, C16c, C17a, C17b = 17 tests). Before, it ran over whichever contrasts
    happened to be present, which shrinks the family and is anti-conservative.
  - C16 is three comparisons: the two plateau pairs (C16a R13-R2, C16b R17-R2) are
    nulls (TOST) and the fall (C16c R15-R2) is a difference test. Before, all of
    C16 was treated as a null.
  - Tier A has 5 realizations from the same code era (seeds 42, 7, 123, 1001, 2026; Enam's ruling of
    26 September). The published runs (jobs/kc23_d1_published_runs.csv) are a SENSITIVITY analysis only: every
    letter, the escalation rule and the headline gate use the seeds alone, and the with-published letters are
    reported beside them with any difference stated.
  - the run-variance deliverable (SD of the 40-fold mean per arm, the pooled SD with
    its degrees of freedom and chi-square 95% interval, the per-fold SD distribution).
  - the secondary model (F1 ~ arm + (1|subject) + (1|seed), statsmodels MixedLM) is
    fitted when statsmodels is importable; if it is not, the verdict SAYS it was not
    computed. The letters do not depend on it (D1.5 items 1 and 2 carry the verdict).
  - required inputs are enforced by directory; a missing input exits 20 with a verdict
    that has no outcome line; the contrasts nothing produces are no longer silently
    omitted (every registered contrast is produced by src/kc23_d1_aggregate.py).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from kc23_d1_aggregate import (ALL_CONTRAST_IDS, DESCRIPTIVE_CONTRASTS, HEADLINE_ARMS, NULL_CONTRASTS,
                               REGISTERED_BH_FAMILY)
from kc23_stats_common import cohens_d_paired, bca_ci, tost_equivalence, print_gate_header
from kc23_stats_common import write_no_outcome_verdict

PUBLISHED_R2 = 0.8395
PUBLISHED_R1 = 0.782
PUBLISHED_R10_PRE = (0.787, 0.772)  # either published figure is an acceptable gate target
REPRO_TOL = 0.015

PUBLISHED_ENSEMBLE = 0.858
PUBLISHED_R12 = 0.860
PUBLISHED_GLOBAL = 0.772

NULL_SET = set(NULL_CONTRASTS)

# Which inputs each directory it serves must hold. Decided by the directory name because the queue calls every
# gate as `python <gate> --out <out_dir>` with nothing else. Any other name is an error, never a permissive default.
REQUIRED_BY_DIR_SUFFIX = {"repro_check": {"repro"}, "d1_stats": {"repro", "contrasts", "headline"}}

CONTRASTS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
            "C12", "C13", "C14", "C15", "C16", "C17"]
TWO_ARM_CONTRASTS = {"C1": ("R2", "R1"), "C2": ("R3", "R2"), "C3": ("R14", "R13"), "C4": ("R16", "R15"),
                     "C5": ("R1", "R4"), "C8": ("R4", "R6"), "C9": ("R2", "R10"), "C10": ("R11", "R10"),
                     "C11": ("R2", "R11"), "C13": ("ENS_SOFT", "R2"), "C14": ("R12", "R2"), "C15": ("R9", "R8"),
                     "C16a": ("R13", "R2"), "C16b": ("R17", "R2"), "C16c": ("R15", "R2")}


def reproduction_gate(r1_mean: float, r2_mean: float, r10_pre_mean: float) -> tuple[str, dict]:
    ok_r2 = abs(r2_mean - PUBLISHED_R2) <= REPRO_TOL
    ok_r1 = abs(r1_mean - PUBLISHED_R1) <= REPRO_TOL
    ok_r10 = any(abs(r10_pre_mean - p) <= REPRO_TOL for p in PUBLISHED_R10_PRE)
    ok = ok_r2 and ok_r1 and ok_r10
    return ("PASS" if ok else "FAIL"), {"r2_ok": ok_r2, "r1_ok": ok_r1, "r10_ok": ok_r10,
                                        "r2_mean": r2_mean, "r1_mean": r1_mean, "r10_pre_mean": r10_pre_mean}


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    return out.tolist()


def classify_contrast(realization_avg_diff: np.ndarray, per_realization_sign_agree: int,
                      n_realizations: int, p_bh: float, tier: str, is_null: bool = False) -> tuple[str, dict]:
    if is_null:
        eq = tost_equivalence(realization_avg_diff, np.zeros_like(realization_avg_diff))
        letter = "EQUIVALENT" if eq["equivalent"] else "NOT_EQUIVALENT"
        return letter, eq

    need_agree = 4 if tier == "A" else 3
    survives_bh = p_bh < 0.05
    sign_consistent = per_realization_sign_agree >= need_agree

    if survives_bh and sign_consistent:
        letter = "ESTABLISHED"
    elif survives_bh or sign_consistent:
        letter = "AMBIGUOUS"
    else:
        letter = "NOT ESTABLISHED"
    return letter, {"survives_bh": survives_bh, "sign_consistent": sign_consistent,
                    "n_realizations_agree": per_realization_sign_agree, "n_realizations": n_realizations,
                    "p_bh": p_bh}


def classify_headline(published: float, realization_mean: float, realization_sd: float) -> tuple[str, dict]:
    lo, hi = realization_mean - 2 * realization_sd, realization_mean + 2 * realization_sd
    within = lo <= published <= hi
    return ("H1" if within else "H2"), {"published": published, "realization_mean": realization_mean,
                                        "realization_sd": realization_sd, "lo": lo, "hi": hi, "within": within}


def stacking_descriptive(f1_stacking: np.ndarray, f1_softvote: np.ndarray, realization_sd: float) -> dict:
    diff = f1_stacking - f1_softvote
    edge_pp = float(diff.mean() * 100.0)
    return {"stacking_mean": float(f1_stacking.mean()), "softvote_mean": float(f1_softvote.mean()),
           "edge_pp": edge_pp, "realization_sd_pp": realization_sd * 100.0,
           "edge_within_run_variance": abs(edge_pp) < realization_sd * 100.0}


# ---------------------------------------------------------------------------------------------- analysis layer
class InputError(Exception):
    pass


def analyse_contrast(g: pd.DataFrame) -> dict:
    """One contrast's rows (realization, subject, diff): the realization-averaged paired test (per subject, average
    over the realizations, then Wilcoxon, dz, BCa, subjects improved), and the seed-level consistency (the mean
    difference within each shared realization, its mean and SD, and how many realizations share the overall sign)."""
    per_subj = g.groupby("subject")["diff"].mean().sort_index()
    avg = per_subj.to_numpy(float)
    by_real = g.groupby("realization")["diff"].mean()
    p_raw = 1.0 if np.allclose(avg, 0) else float(stats.wilcoxon(avg, zero_method="wilcox", alternative="two-sided").pvalue)
    lo, hi = bca_ci(avg)
    sign = np.sign(by_real.mean())
    return {"avg": avg, "n_realizations": int(len(by_real)), "p_raw": p_raw, "mean_diff_pp": float(avg.mean() * 100.0),
            "dz": cohens_d_paired(avg, np.zeros_like(avg)), "bca_lo_pp": float(lo * 100.0), "bca_hi_pp": float(hi * 100.0),
            "n_improved": int((avg > 0).sum()), "seed_mean_pp": float(by_real.mean() * 100.0),
            "seed_sd_pp": float(by_real.std(ddof=1) * 100.0) if len(by_real) > 1 else float("nan"),
            "n_agree": int((np.sign(by_real) == sign).sum())}


def contrast_table(df: pd.DataFrame, include_published: bool) -> dict[str, dict]:
    """Letters for every contrast, with BH over the FIXED registered family."""
    d = df if include_published else df[df["realization"].astype(str) != "published"]
    out: dict[str, dict] = {}
    for cid in ALL_CONTRAST_IDS:
        g = d[d["contrast"] == cid]
        if g.empty:
            raise InputError(f"contrast {cid} has no rows")
        a = analyse_contrast(g)
        a["tier"] = g["tier"].iloc[0]
        out[cid] = a
    p_bh = benjamini_hochberg([out[c]["p_raw"] for c in REGISTERED_BH_FAMILY])
    for c, pb in zip(REGISTERED_BH_FAMILY, p_bh):
        out[c]["p_bh"] = pb
        out[c]["letter"], out[c]["detail"] = classify_contrast(out[c]["avg"], out[c]["n_agree"], out[c]["n_realizations"],
                                                               pb, out[c]["tier"])
    for c in NULL_SET:
        out[c]["letter"], out[c]["detail"] = classify_contrast(out[c]["avg"], 0, out[c]["n_realizations"], 1.0,
                                                               out[c]["tier"], is_null=True)
    for c in DESCRIPTIVE_CONTRASTS:
        out[c]["letter"] = "descriptive"
    return out


def run_variance(af: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The D1.5 run-variance deliverable, on the NEW realizations only (the published run is a different code state)."""
    d = af[af["realization"].astype(str) != "published"]
    rows, dfs, variances, fold_sds = [], [], [], []
    for arm, g in d.groupby("arm"):
        means = g.groupby("realization")["f1"].mean()
        if len(means) < 2:
            continue
        rows.append({"arm": arm, "n_realizations": int(len(means)), "mean_f1": float(means.mean()),
                     "sd_of_40fold_mean": float(means.std(ddof=1))})
        dfs.append(len(means) - 1)
        variances.append(means.var(ddof=1))
        piv = g.pivot_table(index="subject", columns="realization", values="f1")
        fold_sds.extend(piv.std(axis=1, ddof=1).dropna().tolist())
    df = int(sum(dfs))
    pooled_var = float(np.sum(np.array(variances) * np.array(dfs)) / df)
    lo = float(np.sqrt(df * pooled_var / stats.chi2.ppf(0.975, df)))
    hi = float(np.sqrt(df * pooled_var / stats.chi2.ppf(0.025, df)))
    fs = np.array(fold_sds)
    summ = {"pooled_sd_pt": float(np.sqrt(pooled_var) * 100), "df": df, "chi2_lo_pt": lo * 100, "chi2_hi_pt": hi * 100,
            "fold_sd_median_pt": float(np.median(fs) * 100), "fold_sd_q25_pt": float(np.quantile(fs, 0.25) * 100),
            "fold_sd_q75_pt": float(np.quantile(fs, 0.75) * 100), "fold_sd_max_pt": float(fs.max() * 100),
            "n_arm_fold_cells": int(len(fs))}
    return pd.DataFrame(rows), summ


STATS_PYTHON = Path(__file__).resolve().parents[1] / ".venv_stats" / "Scripts" / "python.exe"
MIXEDLM_SCRIPT = Path(__file__).resolve().parents[1] / "src/kc23_d1_mixedlm.py"


def _mixedlm_in_stats_env(af: pd.DataFrame) -> tuple[str, pd.DataFrame | None]:
    """statsmodels is NOT installed in .venv (frozen for reproduction); it lives in .venv_stats, used only for this model. The
    fit runs in that interpreter on a CSV round trip. Anything that goes wrong is stated in the verdict, never silent."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix="kc23_d1_mlm_") as tmp:
        inp, outp = Path(tmp) / "af.csv", Path(tmp) / "mixedlm.csv"
        af.to_csv(inp, index=False)
        r = subprocess.run([str(STATS_PYTHON), str(MIXEDLM_SCRIPT), "--in", str(inp), "--out", str(outp)],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not outp.exists():
            return ("Secondary mixed model: NOT computed, the statsmodels environment (.venv_stats) failed "
                    f"({(r.stderr or r.stdout).strip()[-200:]})."), None
        res = pd.read_csv(outp)
    ok = res[res["status"] == "ok"]
    failed = res[res["status"] != "ok"]
    msg = (f"Secondary mixed model (F1 ~ arm + (1|subject) + (1|seed)) fitted in .venv_stats ({r.stdout.strip().split(':')[0].split(']')[-1].strip() or 'statsmodels'}): "
           f"{len(ok)} of {len(res)} two-arm contrasts; see D1_secondary_mixedlm.csv.")
    if len(failed):
        msg += " Not fitted: " + ", ".join(f"{x.contrast} ({x.status})" for x in failed.itertuples()) + "."
    return msg, res


def secondary_model(af: pd.DataFrame) -> tuple[str, pd.DataFrame | None]:
    """D1.5 item 3: F1 ~ arm + (1|subject) + (1|seed) via statsmodels MixedLM, for the two-arm contrasts. Returns (a sentence for
    the verdict, a table or None). In-process when statsmodels is importable; otherwise in the separate .venv_stats environment;
    otherwise the sentence says NOT computed. The letters never depend on it (D1.5 items 1 and 2 carry the verdict)."""
    try:
        import statsmodels  # noqa: F401
    except Exception:
        if STATS_PYTHON.exists():
            return _mixedlm_in_stats_env(af)
        return ("Secondary mixed model (F1 ~ arm + (1|subject) + (1|seed)): NOT computed, statsmodels is not installed in "
                "this environment and .venv_stats does not exist. The letters do not depend on it (D1.5 items 1 and 2 "
                "carry the verdict)."), None
    import importlib.util
    spec = importlib.util.spec_from_file_location("kc23_d1_mixedlm", MIXEDLM_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.fit_all(af)
    return ("Secondary mixed model (F1 ~ arm + (1|subject) + (1|seed)) fitted; see D1_secondary_mixedlm.csv."), res


def required_inputs(out_dir: Path):
    for suffix, req in REQUIRED_BY_DIR_SUFFIX.items():
        if out_dir.name.endswith(suffix):
            return req
    return None


def _no_outcome(out_dir: Path, reason: str) -> int:
    print(f"[D1] FAIL (no outcome computed): {reason}", file=sys.stderr)
    write_no_outcome_verdict(out_dir / "D1_VERDICT.md", "KC-D1 verdict", reason)
    return 20


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise InputError(f"required input missing: {path}")
    return pd.read_csv(path)


def run(out_dir: Path, require=None) -> int:
    """require: the set of inputs this directory MUST hold ({"repro"}, or repro/contrasts/headline). Fail closed: a
    missing required input, or no input at all, exits 20 with a verdict that has no outcome line."""
    files = {"repro": out_dir / "d1_reproduction_inputs.csv", "contrasts": out_dir / "d1_contrasts.csv",
             "headline": out_dir / "d1_headline_inputs.csv"}
    missing = [k for k in (require or ()) if not files[k].exists()]
    if missing:
        return _no_outcome(out_dir, f"required input(s) missing in {out_dir.name}: {[files[k].name for k in missing]}")
    if not any(f.exists() for f in files.values()):
        return _no_outcome(out_dir, f"no D1 input present in {out_dir.name}")
    exit_code = 0

    # D1.4 reproduction gate. (Fixed 2026-09-24: this letter was once computed, used, and then silently dropped from the
    # verdict by a shadowed local name.)
    repro_letter, repro_detail = None, None
    if files["repro"].exists():
        g = pd.read_csv(files["repro"]).set_index("arm")["f1_mean"]
        repro_letter, repro_detail = reproduction_gate(float(g["R1"]), float(g["R2"]), float(g["R10_pre"]))
        print_gate_header("KC-D1 reproduction gate", repro_letter,
                          "Continue." if repro_letter == "PASS" else "ESCALATE: code drift since publication.")
        print(f"  {repro_detail}")
        if repro_letter == "FAIL":
            exit_code = 20

    verdict = ["# KC-D1 verdict\n"]
    if repro_letter is not None:
        verdict.append(f"- **reproduction: {repro_letter}**")
    tables, notes = [], []

    if "contrasts" in (require or ()) and exit_code != 20:
        try:
            df = _read(files["contrasts"])
            need = {"contrast", "tier", "realization", "subject", "diff"}
            if not need <= set(df.columns):
                raise InputError(f"d1_contrasts.csv lacks {sorted(need - set(df.columns))}")
            have = set(df["contrast"])
            gaps = [c for c in ALL_CONTRAST_IDS if c not in have]
            if gaps:
                raise InputError(f"contrasts absent from d1_contrasts.csv: {gaps}")
            af = _read(out_dir / "d1_arm_subject_f1.csv")
            c17 = _read(out_dir / "d1_c17_factors.csv")
            primary = contrast_table(df, include_published=False)          # the seeds alone: the published run is never a realization
            sensitivity = contrast_table(df, include_published=True)
        except InputError as e:
            return _no_outcome(out_dir, str(e))

        rows = []
        for cid in ALL_CONTRAST_IDS:
            p, s = primary[cid], sensitivity[cid]
            label = p["letter"]
            shown = "C16 plateau pair, null" if cid in ("C16a", "C16b") else ""
            verdict.append(f"- **{cid}: {label}**" if label != "descriptive" else f"- {cid} (descriptive, D-6a): stacking "
                           f"minus soft vote {p['mean_diff_pp']:+.2f} pt")
            rows.append({"contrast": cid, "letter": label, "letter_with_published": s["letter"], "tier": p["tier"],
                         "n_realizations": p["n_realizations"], "n_realizations_with_published": s["n_realizations"],
                         "mean_diff_pp": p["mean_diff_pp"], "dz": p["dz"], "bca_lo_pp": p["bca_lo_pp"],
                         "bca_hi_pp": p["bca_hi_pp"], "n_improved": p["n_improved"], "p_raw": p["p_raw"],
                         "p_bh": p.get("p_bh", float("nan")), "seed_mean_pp": p["seed_mean_pp"],
                         "seed_sd_pp": p["seed_sd_pp"], "n_seeds_same_sign": p["n_agree"], "note": shown})
            print_gate_header(f"KC-D1 {cid}", label, "")
        res = pd.DataFrame(rows)
        res.to_csv(out_dir / "D1_contrasts_verdict.csv", index=False)
        differing = res[(res["letter"] != res["letter_with_published"]) & (res["letter"] != "descriptive")]
        if len(differing):
            notes.append("Sensitivity (the published run added as an extra realization) changes the letter for: " + ", ".join(
                f"{r.contrast} ({r.letter} -> {r.letter_with_published})" for r in differing.itertuples()) + ".")
        else:
            notes.append("Sensitivity (the published run added as an extra realization): no contrast changes letter.")
        tables.append("| contrast | letter | with published (sensitivity) | mean diff (pt) | dz | p | p BH | improved | seeds same sign |\n"
                      "|---|---|---|---|---|---|---|---|---|\n" +
                      "\n".join(f"| {r.contrast} | {r.letter} | {r.letter_with_published} | {r.mean_diff_pp:+.2f} | {r.dz:+.2f} | "
                                f"{r.p_raw:.3g} | {'' if pd.isna(r.p_bh) else format(r.p_bh, '.3g')} | {r.n_improved}/40 | "
                                f"{r.n_seeds_same_sign}/{r.n_realizations} |" for r in res.itertuples()))
        for critical in ("C1", "C12"):
            if primary[critical]["letter"] == "NOT ESTABLISHED":
                exit_code = 20
                print(f"[D1] {critical} NOT ESTABLISHED -> ESCALATE (Finding C's core changes).")
                notes.append(f"{critical} is NOT ESTABLISHED: ESCALATE (D1.7).")
        # C13b, the descriptive stacking row, against the realization SD of the soft vote
        new = af[af["realization"].astype(str) != "published"]
        soft_sd = float(new[new["arm"] == "ENS_SOFT"].groupby("realization")["f1"].mean().std(ddof=1))
        soft = new[new["arm"] == "ENS_SOFT"].groupby("subject")["f1"].mean().sort_index()
        stack = new[new["arm"] == "ENS_STACK"].groupby("subject")["f1"].mean().sort_index()
        c13b = stacking_descriptive(stack.to_numpy(), soft.to_numpy(), soft_sd)
        pd.DataFrame([c13b]).to_csv(out_dir / "D1_C13b_stacking.csv", index=False)
        notes.append(f"C13b (descriptive): stacking {c13b['edge_pp']:+.2f} pt over the soft vote against a realization SD of "
                     f"{c13b['realization_sd_pp']:.2f} pt (within run variance: {c13b['edge_within_run_variance']}).")
        for label, gg in c17.groupby("comparison"):
            notes.append(f"C17, occlusion reduction factor {label}: {gg['factor'].mean():.2f}x, SD {gg['factor'].std(ddof=1):.2f} "
                         f"across {len(gg)} realizations.")
        rv, rv_sum = run_variance(af)
        rv.to_csv(out_dir / "D1_run_variance.csv", index=False)
        pd.DataFrame([rv_sum]).to_csv(out_dir / "D1_run_variance_pooled.csv", index=False)
        notes.append(f"Run variance: pooled SD of the 40-fold mean {rv_sum['pooled_sd_pt']:.2f} pt on {rv_sum['df']} degrees of "
                     f"freedom (chi-square 95% interval {rv_sum['chi2_lo_pt']:.2f} to {rv_sum['chi2_hi_pt']:.2f} pt); per-fold SD "
                     f"across realizations, median {rv_sum['fold_sd_median_pt']:.2f} pt (IQR {rv_sum['fold_sd_q25_pt']:.2f} to "
                     f"{rv_sum['fold_sd_q75_pt']:.2f}, max {rv_sum['fold_sd_max_pt']:.2f}) over {rv_sum['n_arm_fold_cells']} arm-fold cells.")
        sec, sec_tab = secondary_model(af)
        notes.append(sec)
        if sec_tab is not None:
            sec_tab.to_csv(out_dir / "D1_secondary_mixedlm.csv", index=False)

    if "headline" in (require or ()) and exit_code != 20:
        try:
            h = _read(files["headline"])
            if not {"arm", "realization_mean", "realization_sd"} <= set(h.columns) or set(h["arm"]) != set(HEADLINE_ARMS):
                raise InputError(f"d1_headline_inputs.csv must hold exactly the arms {list(HEADLINE_ARMS)}")
        except InputError as e:
            return _no_outcome(out_dir, str(e))
        published = {"R2": PUBLISHED_R2, "ensemble": PUBLISHED_ENSEMBLE, "R12": PUBLISHED_R12, "global": PUBLISHED_GLOBAL}
        checks = {}
        for arm, pub in published.items():
            row = h[h["arm"] == arm].iloc[0]
            letter, detail = classify_headline(pub, float(row["realization_mean"]), float(row["realization_sd"]))
            checks[arm] = {"letter": letter, **detail}
            verdict.append(f"- **headline ({arm}): {letter}**")
            print_gate_header(f"KC-D1 headline ({arm})", letter, "")
        pd.DataFrame([{"arm": a, **v} for a, v in checks.items()]).to_csv(out_dir / "D1_headline_verdict.csv", index=False)
        if any(v["letter"] == "H2" for v in checks.values()):
            exit_code = 20
            notes.append("A headline arm is H2: ESCALATE (D1.6), whether to report the realization mean as the headline is "
                         "Enam's decision.")

    out_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(verdict) + "\n"
    if tables:
        body += "\n" + "\n\n".join(tables) + "\n"
    if notes:
        body += "\n" + "\n\n".join(notes) + "\n"
    (out_dir / "D1_VERDICT.md").write_text(body, encoding="utf-8")
    return exit_code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    req = required_inputs(out)
    if req is None:
        sys.exit(_no_outcome(out, f"cannot tell which inputs {out.name!r} must hold (expected a name ending in "
                                  f"{sorted(REQUIRED_BY_DIR_SUFFIX)})"))
    sys.exit(run(out, req))


if __name__ == "__main__":
    main()
