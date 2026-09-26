#!/usr/bin/env python3
"""
kc23_d6_aggregate.py
======================
KC-D6 aggregator. Rewritten 26 September 2026 (KC23_PREREG_CONFORMANCE.md). The
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
    "adv_marginal": lambda knob, seed: f"results_kc23_d6_adv_marginal_l{knob}_s{seed}",
    "sfc": lambda knob, seed: f"results_kc23_d6_sfc_w{knob}_s{seed}",
    "advps": lambda knob, seed: f"results_kc23_d6_advps_l{knob}_s{seed}",
}
SUBJECTWISE = {"adv_marginal": "adv_subjectwise.csv", "sfc": "deep_coral_subjectwise.csv",
               "advps": "adv_subjectwise.csv"}
KNOWN_FAMILIES = list(FAMILY_KNOBS)
STAGE1_SEED = 42
OUTCOME_SEEDS = [42, 7, 123]
N_FOLDS = 40
DIVERGENCE_RATIO = 2.0
MEASURES = ["f1", "domain_probe_bacc", "subject_probe_bacc", "class_silhouette", "class_probe_bacc", "feat_norm_src",
            "best_val_loss"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    return pd.read_csv(path)


def _expect_config(family: str, knob, seed: int, cfg: dict, label: str) -> None:
    def need(k, v):
        if cfg.get(k) != v:
            raise ValueError(f"{label}: run_config {k}={cfg.get(k)!r}, expected {v!r} (wrong arm or mislabelled)")
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


def load_arm(root: Path, family: str, knob, seed: int) -> pd.DataFrame:
    d = root / FAMILY_DIR[family](knob, seed)
    label = d.name
    _expect_config(family, knob, seed, read_config(d), label)
    sw = _read(d / SUBJECTWISE[family])
    al = _read(d / "alignment_subjectwise.csv")
    ep = _read(d / "instr" / "embed_probes.csv")
    tl = _read(d / "training_log.csv")

    out = pd.DataFrame({"f1": _fold_series(sw, "f1_macro", f"{label} f1")})
    out["domain_probe_bacc"] = _fold_series(al, "domain_probe_bacc", f"{label} domain probe")
    out["feat_norm_src"] = _fold_series(al, "feat_norm_src", f"{label} embedding norm")
    out["subject_probe_bacc"] = _fold_series(ep, "subject_probe_bacc", f"{label} unseen-subject probe")
    out["class_silhouette"] = _fold_series(ep, "held_out_class_silhouette", f"{label} class silhouette")
    out["class_probe_bacc"] = _fold_series(ep, "held_out_class_probe_bacc", f"{label} class probe")
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


def build_family(root: Path, family: str, seeds: list[int]):
    arms = {(k, s): load_arm(root, family, k, s) for s in seeds for k in FAMILY_KNOBS[family]}
    div = arm_divergence(arms, family)
    long = pd.concat(arms.values(), ignore_index=True)
    long = long.merge(div[["knob", "realization", "diverged"]], on=["knob", "realization"], how="left")
    cols = ["family", "knob", "realization", "subject"] + MEASURES + ["nonfinite", "diverged"]
    return long[cols], div


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


FILES = ["d6_sanity.csv"] + [f"d6_family_{f}.csv" for f in KNOWN_FAMILIES] + ["d6_arm_divergence.csv",
                                                                              "d6_no_passing_family.csv"]


def run(out_dir: Path, root: Path, require: str | None = None, gates: Path | None = None) -> int:
    """Fail closed. --require picks what must be complete; exit 1 and no output file otherwise."""
    if require not in ("sanity", "manipulation", "outcome"):
        print("[d6-aggregate] FAIL: --require sanity|manipulation|outcome is mandatory", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in FILES:
        (out_dir / n).unlink(missing_ok=True)
    try:
        written: dict[str, pd.DataFrame] = {}
        if require == "sanity":
            written["d6_sanity.csv"] = build_sanity(root)
        else:
            if require == "manipulation":
                families = {f: STAGE1_SEED for f in KNOWN_FAMILIES}
                seeds_of = {f: [STAGE1_SEED] for f in KNOWN_FAMILIES}
            else:
                gp = passing_families(gates or (root / "results_kc23_d6_manipulation_check" / "D6_gates.csv"))
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
    ap.add_argument("--require", choices=["sanity", "manipulation", "outcome"], default=None)
    ap.add_argument("--gates", default=None, help="outcome mode: D6_gates.csv from the manipulation check")
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.root), args.require, Path(args.gates) if args.gates else None))


if __name__ == "__main__":
    main()
