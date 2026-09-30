#!/usr/bin/env python3
"""
src/kc23_d6_aggregate.py
======================
KC-D6 aggregator. Rewritten 26 September 2026 (docs/kc23/KC23_PREREG_CONFORMANCE.md). The
earlier version fed `class_probe_tgt_bacc` (a CLASS probe on the target subject)
into the gate as "subject_probe", and used the SUBJECT as the Page-test
"realization"; the unseen-subject probe the plan names (D0.3, embed_probes.csv)
was never written by the D6 runners. They now take --instrument, and this reads
it. Nothing is defaulted.

Per arm (family, knob, seed) it reads the run directory:
  run_config.json                   checked against the arm (seed, knob, mode, normalization, augmentation)
  adv_subjectwise.csv | deep_coral_subjectwise.csv   f1_macro
  alignment_subjectwise.csv         domain_probe_bacc (source against target), feat_norm_src
  training_log.csv                  per fold: the validation loss at the best epoch, and any non-finite loss
  instr/embed_probes.csv            subject_probe_bacc (unseen subjects), class silhouette, class probe
and requires the same 40 folds in every file, with no NaN in any measure.

Divergence (plan 6.3, fixed): an arm has diverged if the source validation loss at its best epoch exceeds twice
that of the same seed's lambda_max = 0 arm, or any loss is NaN. Applied with the lambda 0 comparison to the ADV
family (adv_marginal has a lambda 0 arm); SFC and ADV-PS have no zero arm in their own harness, so only the NaN clause
applies to them. A diverged arm is flagged, never dropped; the retry with --grad-clip 5.0 is a job-level step (a
retry directory, if present, is NOT silently substituted here).

Modes (--require):
  sanity        d6_sanity.csv                      ADV lambda 0, seed 42
  manipulation  d6_family_<family>.csv x3 (seed 42), d6_arm_divergence.csv
  outcome       d6_family_<family>.csv (seeds 42, 7, 123) for every family whose manipulation gate letter was
                G-PASS or G-WEAK (read from --gates), d6_arm_divergence.csv
  mechanism     d6_mechanism_arms.csv, d6_mechanism_meta.csv: ADV (with the within-class subject probe) at seeds 42, 7, 123, the
                ADV collapse point (the smallest lambda_max above the peak whose realization-mean F1 is 2 pts or more below
                the peak), and the ADV-C runs (oracle target labels) at the collapse lambda and the next grid value up; ADV-CDAN
                arms too when their directories exist. No collapse: ADV-C is not run and the meta file says so (plan 6.2).
  secondary     d6_secondary.csv: ADV at its best lambda against KC-D1's R2, R10 (post) and R11, and ADV-PS at each lambda
                against R2 (the per-subject harness without an adversary), per subject and realization (plan 6.6).

Divergence retry (plan 6.3): an arm that diverged is retried ONCE with --grad-clip 5.0 into <arm directory>__retry
(src/kc23_d6_retry_job_gen.py). Manipulation, outcome and mechanism refuse to run (exit 1) while a diverged arm has no retry
directory; a retry replaces its arm, judged by the same rule against the same seed's lambda 0 arm, and an arm that diverges
again stays diverged (a result: training destroyed at that strength), flagged, never dropped. Output rows carry
diverged_original and retried.
Any missing or malformed input exits 1 and writes nothing.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_run_loader import read_config

FAMILY_KNOBS = {
    "adv_marginal": [0, 0.03, 0.1, 0.3, 1, 3, 10],
    "sfc": [0.1, 1, 10, 100, 1000],
    "advps": [0.1, 1, 10],
}
FAMILY_DIR = {
    "adv_marginal": lambda knob, seed: f"results/kc23_d6_adv_marginal_l{knob}_s{seed}",
    "sfc": lambda knob, seed: f"results/kc23_d6_sfc_w{knob}_s{seed}",
    "advps": lambda knob, seed: f"results/kc23_d6_advps_l{knob}_s{seed}",
}
SUBJECTWISE = {"adv_marginal": "adv_subjectwise.csv", "sfc": "deep_coral_subjectwise.csv",
               "advps": "adv_subjectwise.csv"}
KNOWN_FAMILIES = list(FAMILY_KNOBS)
STAGE1_SEED = 42
OUTCOME_SEEDS = [42, 7, 123]
N_FOLDS = 40
DIVERGENCE_RATIO = 2.0
GRAD_CLIP_RETRY = 5.0
RETRY_SUFFIX = "__retry"
COLLAPSE_DROP = 0.02                 # plan 6.2: "at least 2 pts below the ADV peak"
MECH_DIR = {"advc": lambda knob, seed: f"results/kc23_d6_advc_l{knob}_s{seed}",
            "cdan": lambda knob, seed: f"results/kc23_d6_cdan_l{knob}_s{seed}"}
MECH_MODE = {"advc": ("classcond", True), "cdan": ("cdan", False)}
MEASURES = ["f1", "domain_probe_bacc", "subject_probe_bacc", "class_silhouette", "class_probe_bacc", "feat_norm_src",
            "coral_embed_normdev", "best_val_loss"]
WITHIN_CLASS = "subject_probe_within_class_bacc"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    return pd.read_csv(path)


def _expect_config(family: str, knob, seed: int, cfg: dict, label: str, retry: bool = False) -> None:
    def need(k, v):
        if cfg.get(k) != v:
            raise ValueError(f"{label}: run_config {k}={cfg.get(k)!r}, expected {v!r} (wrong arm or mislabelled)")
    gc = cfg.get("grad_clip")
    if retry:
        if gc is None or abs(float(gc) - GRAD_CLIP_RETRY) > 1e-9:
            raise ValueError(f"{label}: a retry directory must have been run with --grad-clip {GRAD_CLIP_RETRY}, found {gc!r}")
    elif gc is not None:
        raise ValueError(f"{label}: grad_clip={gc!r} on an arm that is not a retry")
    need("seed", seed)
    need("arch", "resnet_se")
    need("augmentation", "chandrop")
    need("epochs", 40)
    need("batch", 256)
    if family in ("adv_marginal", "advps"):
        need("adv_mode", "marginal")
        need("norm_mode", "global" if family == "adv_marginal" else "per_subject")
        if abs(float(cfg.get("adv_lambda", -1)) - float(knob)) > 1e-9:
            raise ValueError(f"{label}: adv_lambda={cfg.get('adv_lambda')!r}, expected {knob}")
    else:
        need("coral_normalize", "l2")
        need("target_pass", "train")
        if abs(float(cfg.get("coral_lambda", -1)) - float(knob)) > 1e-9:
            raise ValueError(f"{label}: coral_lambda={cfg.get('coral_lambda')!r}, expected {knob}")


def _fold_series(df: pd.DataFrame, col: str, label: str) -> pd.Series:
    if "subject" not in df.columns or col not in df.columns:
        raise ValueError(f"{label}: lacks subject/{col}")
    if df["subject"].duplicated().any() or len(df) != N_FOLDS:
        raise ValueError(f"{label}: needs {N_FOLDS} unique folds, found {len(df)} rows")
    if df[col].isna().any():
        raise ValueError(f"{label}: NaN in {col}")
    return df.set_index("subject")[col].astype(float).sort_index()


def load_arm(root: Path, family: str, knob, seed: int, retry: bool = False, within_class: bool = False) -> pd.DataFrame:
    d = root / (FAMILY_DIR[family](knob, seed) + (RETRY_SUFFIX if retry else ""))
    label = d.name
    _expect_config(family, knob, seed, read_config(d), label, retry=retry)
    sw = _read(d / SUBJECTWISE[family])
    al = _read(d / "alignment_subjectwise.csv")
    ep = _read(d / "instr" / "embed_probes.csv")
    tl = _read(d / "training_log.csv")

    out = pd.DataFrame({"f1": _fold_series(sw, "f1_macro", f"{label} f1")})
    out["domain_probe_bacc"] = _fold_series(al, "domain_probe_bacc", f"{label} domain probe")
    out["feat_norm_src"] = _fold_series(al, "feat_norm_src", f"{label} embedding norm")
    if family == "sfc":   # the unit-norm check on the embedding the CORAL term sees (Enam, 26 Sept): required, never defaulted
        out["coral_embed_normdev"] = _fold_series(al, "coral_embed_normdev_max", f"{label} normalised-embedding norm deviation")
    else:
        out["coral_embed_normdev"] = float("nan")
    out["subject_probe_bacc"] = _fold_series(ep, "subject_probe_bacc", f"{label} unseen-subject probe")
    out["class_silhouette"] = _fold_series(ep, "held_out_class_silhouette", f"{label} class silhouette")
    out["class_probe_bacc"] = _fold_series(ep, "held_out_class_probe_bacc", f"{label} class probe")
    if within_class:     # the mechanism test's invariance measure (plan 6.6); the runner writes it only with --within-class-probe
        out[WITHIN_CLASS] = _fold_series(ep, WITHIN_CLASS, f"{label} within-class subject probe")
    if set(out.index) != set(sw["subject"]):
        raise ValueError(f"{label}: subject sets differ between files")

    best_loss, nonfinite = {}, {}
    for s, g in tl.groupby("subject"):
        vl = g["val_loss"].to_numpy(float)
        bad = (~np.isfinite(vl)).any() or (("diverged" in g.columns) and (g["diverged"].fillna(0) == 1).any())
        best = g[g["best"] == 1] if "best" in g.columns else g.iloc[0:0]
        best_loss[int(s)] = float(best["val_loss"].iloc[-1]) if len(best) else float("nan")
        nonfinite[int(s)] = bool(bad or not len(best))
    if set(best_loss) != set(out.index):
        raise ValueError(f"{label}: training_log folds differ from the result folds")
    out["best_val_loss"] = pd.Series(best_loss)
    out["nonfinite"] = pd.Series(nonfinite)
    out["family"], out["knob"], out["realization"] = family, knob, seed
    return out.reset_index().rename(columns={"index": "subject"})


def arm_divergence(arms: dict, family: str) -> pd.DataFrame:
    """arms: {(knob, seed): fold frame}. Diverged per the plan's rule; NaN-only where there is no lambda 0 arm."""
    rows = []
    for (knob, seed), df in arms.items():
        nonfinite = bool(df["nonfinite"].any())
        mean_loss = float(df["best_val_loss"].mean(skipna=True)) if df["best_val_loss"].notna().any() else float("nan")
        ratio = float("nan")
        exceeds = False
        if family == "adv_marginal":
            zero = arms.get((0, seed))
            if zero is None:
                raise ValueError(f"adv_marginal seed {seed}: no lambda 0 arm to judge divergence against")
            zero_loss = float(zero["best_val_loss"].mean(skipna=True))
            ratio = mean_loss / zero_loss if zero_loss > 0 else float("inf")
            exceeds = bool(knob != 0 and ratio > DIVERGENCE_RATIO)
        rows.append({"family": family, "knob": knob, "realization": seed, "mean_best_val_loss": mean_loss,
                     "ratio_to_lambda0": ratio, "nonfinite_loss": nonfinite,
                     "diverged": bool(nonfinite or exceeds)})
    return pd.DataFrame(rows)


def _orig_knob(family: str, value):
    """The grid's own knob object (10, not 10.0): directory names are written from it. Pandas turns a mixed int/float column
    into floats, and 'l10.0' is not the directory the run wrote."""
    for k in FAMILY_KNOBS[family]:
        if float(k) == float(value):
            return k
    raise ValueError(f"{family}: knob {value!r} is not on the grid {FAMILY_KNOBS[family]}")


def diverged_arms(root: Path, family: str, seeds: list[int]) -> list[tuple]:
    """(knob, seed, directory name) of every ORIGINAL arm that diverged by the plan's rule, whether or not a retry exists."""
    arms = {(k, sd): load_arm(root, family, k, sd) for sd in seeds for k in FAMILY_KNOBS[family]}
    div = arm_divergence(arms, family)
    return [(_orig_knob(family, r.knob), int(r.realization), FAMILY_DIR[family](_orig_knob(family, r.knob), int(r.realization)))
            for r in div.itertuples() if r.diverged]


def build_family(root: Path, family: str, seeds: list[int], within_class: bool = False):
    arms = {(k, s): load_arm(root, family, k, s, within_class=within_class) for s in seeds for k in FAMILY_KNOBS[family]}
    div0 = arm_divergence(arms, family)
    need = [(_orig_knob(family, r.knob), int(r.realization)) for r in div0.itertuples() if r.diverged]
    unresolved = []
    for k, sd in need:
        rd = root / (FAMILY_DIR[family](k, sd) + RETRY_SUFFIX)
        if not rd.exists():
            unresolved.append(FAMILY_DIR[family](k, sd))
            continue
        arms[(k, sd)] = load_arm(root, family, k, sd, retry=True, within_class=within_class)
    if unresolved:
        raise ValueError(f"{family}: arm(s) {unresolved} diverged and have no {RETRY_SUFFIX} directory; run the one retry with "
                         f"--grad-clip {GRAD_CLIP_RETRY} first (src/kc23_d6_retry_job_gen.py), never drop or substitute silently")
    div = arm_divergence(arms, family)
    div["diverged_original"] = [(_orig_knob(family, r.knob), int(r.realization)) in need for r in div.itertuples()]
    div["retried"] = div["diverged_original"]
    long = pd.concat(arms.values(), ignore_index=True)
    long = long.merge(div[["knob", "realization", "diverged", "diverged_original", "retried"]],
                      on=["knob", "realization"], how="left")
    cols = (["family", "knob", "realization", "subject"] + MEASURES + ([WITHIN_CLASS] if within_class else [])
            + ["nonfinite", "diverged", "diverged_original", "retried"])
    return long[cols], div


# --------------------------------------------------------------------------- the mechanism test (plan 6.6)
def collapse_point(mean_f1: dict, knobs: list) -> dict:
    """ADV peak and collapse. mean_f1: {knob: realization-mean F1}. The collapse point is the smallest lambda_max ABOVE the
    peak whose F1 is 2 pts or more below the peak (a lambda_max below the peak that is low because it is still rising is not a
    collapse); the next grid value up from it is the second ADV-C value. None when ADV never falls 2 pts below its peak."""
    peak_k = max(knobs, key=lambda k: mean_f1[k])
    peak = float(mean_f1[peak_k])
    after = [k for k in knobs if float(k) > float(peak_k)]
    collapse = next((k for k in after if float(mean_f1[k]) <= peak - COLLAPSE_DROP + 1e-12), None)
    nxt = next((k for k in after if collapse is not None and float(k) > float(collapse)), None)
    return {"peak_knob": peak_k, "peak_f1": peak, "collapse_knob": collapse, "next_knob": nxt}


def adv_mean_f1(root: Path, seeds: list[int]) -> tuple[dict, pd.DataFrame]:
    long, _ = build_family(root, "adv_marginal", seeds, within_class=True)
    per = long.groupby(["knob", "realization"])["f1"].mean().groupby("knob").mean()
    return {_orig_knob("adv_marginal", k): float(v) for k, v in per.items()}, long


def load_mech_arm(root: Path, kind: str, knob, seed: int) -> pd.DataFrame:
    d = root / MECH_DIR[kind](knob, seed)
    label = d.name
    mode, oracle = MECH_MODE[kind]
    cfg = read_config(d)
    for k, v in (("seed", seed), ("arch", "resnet_se"), ("augmentation", "chandrop"), ("epochs", 40), ("batch", 256),
                 ("norm_mode", "global"), ("adv_mode", mode)):
        if cfg.get(k) != v:
            raise ValueError(f"{label}: run_config {k}={cfg.get(k)!r}, expected {v!r} (wrong arm or mislabelled)")
    if abs(float(cfg.get("adv_lambda", -1)) - float(knob)) > 1e-9:
        raise ValueError(f"{label}: adv_lambda={cfg.get('adv_lambda')!r}, expected {knob}")
    if bool(cfg.get("oracle_target_labels", False)) != oracle:
        raise ValueError(f"{label}: oracle_target_labels={cfg.get('oracle_target_labels')!r}, expected {oracle}")
    sw = _read(d / "adv_subjectwise.csv")
    if "oracle" not in sw.columns or bool(sw["oracle"].astype(bool).all()) != oracle or bool(sw["oracle"].astype(bool).any()) != oracle:
        raise ValueError(f"{label}: every output row must carry oracle={oracle}")
    al = _read(d / "alignment_subjectwise.csv")
    ep = _read(d / "instr" / "embed_probes.csv")
    tl = _read(d / "training_log.csv")
    out = pd.DataFrame({"f1": _fold_series(sw, "f1_macro", f"{label} f1")})
    out["domain_probe_bacc"] = _fold_series(al, "domain_probe_bacc", f"{label} domain probe")
    out["subject_probe_bacc"] = _fold_series(ep, "subject_probe_bacc", f"{label} unseen-subject probe")
    out[WITHIN_CLASS] = _fold_series(ep, WITHIN_CLASS, f"{label} within-class subject probe")
    if set(out.index) != set(sw["subject"]):
        raise ValueError(f"{label}: subject sets differ between files")
    bad = {int(s): bool((~np.isfinite(g["val_loss"].to_numpy(float))).any() or
                        (("diverged" in g.columns) and (g["diverged"].fillna(0) == 1).any())) for s, g in tl.groupby("subject")}
    out["nonfinite"] = pd.Series(bad)
    out["family"], out["knob"], out["realization"] = kind, knob, seed
    return out.reset_index().rename(columns={"index": "subject"})


def build_mechanism(root: Path, seeds: list[int] = OUTCOME_SEEDS) -> dict[str, pd.DataFrame]:
    mean_f1, adv_long = adv_mean_f1(root, seeds)
    cp = collapse_point(mean_f1, FAMILY_KNOBS["adv_marginal"])
    meta = {**cp, "not_run": cp["collapse_knob"] is None, "seeds": ";".join(map(str, seeds)),
            "adv_f1_by_knob": ";".join(f"{k}:{mean_f1[k]:.4f}" for k in FAMILY_KNOBS["adv_marginal"])}
    if meta["not_run"]:
        return {"d6_mechanism_meta.csv": pd.DataFrame([meta])}
    knobs = [cp["collapse_knob"]] + ([cp["next_knob"]] if cp["next_knob"] is not None else [])
    keep_adv = sorted({cp["peak_knob"], *knobs}, key=float)
    rows = [adv_long[adv_long["knob"].isin(keep_adv)].assign(kind="adv")[
        ["kind", "knob", "realization", "subject", "f1", "domain_probe_bacc", "subject_probe_bacc", WITHIN_CLASS, "nonfinite", "diverged"]]]
    for kind in ("advc", "cdan"):
        present = [(k, sd) for k in knobs for sd in seeds if (root / MECH_DIR[kind](k, sd)).exists()]
        if kind == "cdan" and not present:
            continue                                   # ADV-CDAN is optional (only after ADV-C lands C-M1)
        missing = [(k, sd) for k in knobs for sd in seeds if (k, sd) not in present]
        if missing:
            raise ValueError(f"{kind}: runs missing for (lambda, seed) {missing}")
        for k, sd in present:
            df = load_mech_arm(root, kind, k, sd)
            df["diverged"] = df["nonfinite"]
            rows.append(df.assign(kind=kind)[["kind", "knob", "realization", "subject", "f1", "domain_probe_bacc",
                                              "subject_probe_bacc", WITHIN_CLASS, "nonfinite", "diverged"]])
    return {"d6_mechanism_meta.csv": pd.DataFrame([meta]),
            "d6_mechanism_arms.csv": pd.concat(rows, ignore_index=True)}


# --------------------------------------------------------------------------- secondary contrasts (plan 6.6)
def _seeds_available(root: Path, family: str, seeds: list[int]) -> list[int]:
    """The seeds for which EVERY knob of the family has a run directory (Stage 2 exists only for families that passed the gate)."""
    return [sd for sd in seeds if all((root / FAMILY_DIR[family](k, sd)).exists() for k in FAMILY_KNOBS[family])]


def build_secondary(root: Path, seeds: list[int] = OUTCOME_SEEDS) -> pd.DataFrame:
    """Seeds per family are those with every knob run: Stage 2 (seeds 7, 123) exists only for a family that passed the manipulation
    gate. Seed 42 is required for both; the number of realizations behind each contrast is reported by the stats."""
    from kc23_d1_aggregate import load_arm as d1_arm
    adv_seeds = _seeds_available(root, "adv_marginal", seeds)
    ps_seeds = _seeds_available(root, "advps", seeds)
    if STAGE1_SEED not in adv_seeds or STAGE1_SEED not in ps_seeds:
        raise ValueError(f"secondary contrasts need every adv_marginal and advps arm at seed {STAGE1_SEED} (found ADV {adv_seeds}, "
                         f"ADV-PS {ps_seeds})")
    adv_long, _ = build_family(root, "adv_marginal", adv_seeds)
    ok = adv_long[~adv_long["diverged"]]
    if ok.empty:
        raise ValueError("adv_marginal: every arm diverged; there is no best lambda_max to contrast")
    best = _orig_knob("adv_marginal", ok.groupby(["knob", "realization"])["f1"].mean().groupby("knob").mean().idxmax())
    ps_long, _ = build_family(root, "advps", ps_seeds)
    rows = []
    for sd in adv_seeds:
        base = {a: d1_arm(root, a, sd) for a in ("R2", "R10", "R11")}
        adv = adv_long[(adv_long["knob"] == best) & (adv_long["realization"] == sd)].set_index("subject")["f1"].sort_index()
        for a, ref in base.items():
            for subj, v in (adv - ref.reindex(adv.index)).items():
                rows.append({"contrast": f"ADV(best lambda={best})-{a}", "realization": sd, "subject": int(subj), "diff": float(v)})
    for sd in ps_seeds:
        r2 = d1_arm(root, "R2", sd)
        for lam in FAMILY_KNOBS["advps"]:
            ps = ps_long[(ps_long["knob"] == lam) & (ps_long["realization"] == sd)].set_index("subject")["f1"].sort_index()
            for subj, v in (ps - r2.reindex(ps.index)).items():
                rows.append({"contrast": f"ADV-PS(lambda={lam})-R2", "realization": sd, "subject": int(subj), "diff": float(v)})
    df = pd.DataFrame(rows)
    if df["diff"].isna().any():
        raise ValueError("secondary contrasts: a subject is missing from one of the arms")
    return df


def build_sanity(root: Path) -> pd.DataFrame:
    d = root / FAMILY_DIR["adv_marginal"](0, STAGE1_SEED)
    _expect_config("adv_marginal", 0, STAGE1_SEED, read_config(d), d.name)
    df = _read(d / "adv_subjectwise.csv")
    if "adv_lambda" not in df.columns or df[df["adv_lambda"] == 0].empty:
        raise ValueError(f"{d.name}: no adv_lambda == 0 rows (no fallback to other lambdas)")
    lam0 = df[df["adv_lambda"] == 0]
    if lam0["subject"].duplicated().any() or len(lam0) != N_FOLDS or lam0["f1_macro"].isna().any():
        raise ValueError(f"{d.name}: needs {N_FOLDS} unique non-NaN lambda 0 folds, found {len(lam0)}")
    return lam0[["subject", "f1_macro"]].rename(columns={"f1_macro": "f1"}).reset_index(drop=True)


def passing_families(gates_csv: Path) -> dict[str, str]:
    """Families whose manipulation letter is G-PASS or G-WEAK and whose normalization check (SFC) did not fail."""
    g = _read(gates_csv)
    out = {}
    for _, r in g.iterrows():
        item = str(r["item"])
        if item.startswith("manipulation_") and str(r["letter"]) in ("G-PASS", "G-WEAK"):
            norm_ok = r["norm_ok"] if "norm_ok" in g.columns and pd.notna(r["norm_ok"]) else True
            if bool(norm_ok):
                out[item[len("manipulation_"):]] = str(r["letter"])
    return out


FILES = ["d6_sanity.csv"] + [f"d6_family_{f}.csv" for f in KNOWN_FAMILIES] + [
    "d6_arm_divergence.csv", "d6_no_passing_family.csv", "d6_mechanism_arms.csv", "d6_mechanism_meta.csv", "d6_secondary.csv"]


def run(out_dir: Path, root: Path, require: str | None = None, gates: Path | None = None) -> int:
    """Fail closed. --require picks what must be complete; exit 1 and no output file otherwise."""
    if require not in ("sanity", "manipulation", "outcome", "mechanism", "secondary"):
        print("[d6-aggregate] FAIL: --require sanity|manipulation|outcome|mechanism|secondary is mandatory", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in FILES:
        (out_dir / n).unlink(missing_ok=True)
    try:
        written: dict[str, pd.DataFrame] = {}
        if require == "sanity":
            written["d6_sanity.csv"] = build_sanity(root)
        elif require == "mechanism":
            written.update(build_mechanism(root))
        elif require == "secondary":
            written["d6_secondary.csv"] = build_secondary(root)
        else:
            if require == "manipulation":
                families = {f: STAGE1_SEED for f in KNOWN_FAMILIES}
                seeds_of = {f: [STAGE1_SEED] for f in KNOWN_FAMILIES}
            else:
                gp = passing_families(gates or (root / "results/kc23_d6_manipulation_check" / "D6_gates.csv"))
                seeds_of = {f: OUTCOME_SEEDS for f in gp}
                if not gp:   # plan 6.7: a G-FAIL on every family is itself the result; say so instead of writing nothing
                    written['d6_no_passing_family.csv'] = pd.DataFrame({'note': ['no family passed the manipulation gate']})
            divs = []
            for f, seeds in seeds_of.items():
                fam, div = build_family(root, f, seeds)
                written[f"d6_family_{f}.csv"] = fam
                divs.append(div)
            if divs:
                written["d6_arm_divergence.csv"] = pd.concat(divs, ignore_index=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[d6-aggregate] FAIL, no output written: {e}", file=sys.stderr)
        return 1
    for name, df in written.items():
        df.to_csv(out_dir / name, index=False)
    print(f"[d6-aggregate] wrote {sorted(written)} ({require})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--require", choices=["sanity", "manipulation", "outcome", "mechanism", "secondary"], default=None)
    ap.add_argument("--gates", default=None, help="outcome mode: D6_gates.csv from the manipulation check")
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.root), args.require, Path(args.gates) if args.gates else None))


if __name__ == "__main__":
    main()
