#!/usr/bin/env python3
"""
kc23_d1_aggregate.py
======================
Turns the KC-D1 run folders (results_kc23_d1_r<N>_s<seed>/), the per-seed ensembles
(kc23_d1_ensemble.py) and the published runs into the inputs kc23_d1_replicate_stats.py reads:
d1_reproduction_inputs.csv, d1_contrasts.csv, d1_headline_inputs.csv, d1_arm_subject_f1.csv and
d1_c17_factors.csv. EXPERIMENT_PLAN_KC23_DEEP.md D1.2 to D1.6.

Rewritten 26 September 2026 (KC23_PREREG_CONFORMANCE.md). The first version wired only R1-R7, R10, R12-R17
and said so; it left C10, C11, C13, C13b, C15, C16, C17 and the ensemble and global headline arms out of
the D1 verdict. Every registered contrast is now produced, with these definitions:

  C1 R2-R1   C2 R3-R2   C3 R14-R13   C4 R16-R15   C5 R1-R4   C6 (R2-R1)-(R5-R4)   C7 (R5-R4)-(R7-R6)
  C8 R4-R6   C9 R2-R10(post)   C10 R11-R10(post)   C11 R2-R11   C12 R2-SVM(fixed 0.777)
  C13 per-seed soft vote - R2   C14 R12-R2   C15 R9-R8 (a null)
  C16 is THREE comparisons: C16a R13-R2 and C16b R17-R2 (the plateau pairs, nulls) and C16c R15-R2 (the fall)
  C17 the occlusion reduction, per-subject summed drop (kc23_run_loader; negative drops kept):
      C17a R1-R2 and C17b R4-R5 (positive = the augmented arm has the smaller drop)
  C13b (decision D-6a, descriptive): stacking - soft vote
and every diff is per subject, per realization, on the SAME seed for every arm in the contrast.

Realizations (plan D1.2): Tier A = seeds 42, 7, 123, 1001 plus the PUBLISHED seed-42 run as a fifth "where one
exists"; Tier B = seeds 42, 7, 123. A published run is used only for arms listed in kc23_d1_published_runs.csv,
each checked against a number the plan itself quotes; a contrast gets the fifth realization only if EVERY arm in
it has a published run. d1_contrasts.csv labels it realization = "published"; the stats report every analysis
with and without it (the plan's sensitivity).

Headline inputs (D1.6): R2, the per-seed ensemble (soft vote), R12, and "global", the global-normalization
ResNet-SE+CD baseline, taken as R10's pre-adaptation F1 per seed (Enam, 26 September 2026). The realization mean
and SD are over the NEW realizations only, so the published value is not in its own reference distribution.

--require repro   the reproduction inputs (R1, R2, R10-pre at seed 42) only (the gate that runs first)
--require full    everything above; every contrast must be complete
Any missing, mislabelled or incomplete input exits 1 and leaves none of its output files behind.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_run_loader import check_config, load_f1, load_instr, N_SUBJECTS_SIAT, reduction_factor

ROOT = Path(__file__).resolve().parent
N_SUBJECTS = N_SUBJECTS_SIAT
TIER_A_SEEDS = [42, 7, 123, 1001]
TIER_B_SEEDS = [42, 7, 123]
TIER_B_ARMS = {"R13", "R14", "R15", "R16", "R17"}
PUBLISHED_SVM = 0.777      # C12: R2 - SVM (fixed, 77.7), plan D1.5
PUBLISHED_RUNS_CSV = ROOT / "kc23_d1_published_runs.csv"

# arm -> (subjectwise file, f1 column, config expectations). None = the run script does not dump run_config.json
# (R10, R11), so the arm is identified by its directory name alone; "seed_only" = check the seed only.
_ARCH = "cnn_arch_subjectwise.csv"
ARM_SPEC = {
    "R1": (_ARCH, "f1_macro", dict(arch="resnet_se", augmentation="none")),
    "R2": (_ARCH, "f1_macro", dict(arch="resnet_se", augmentation="chandrop", chandrop_p=0.2)),
    "R3": (_ARCH, "f1_macro", dict(arch="resnet_se", augmentation="gainjitter", gain_sd=0.40)),
    "R4": (_ARCH, "f1_macro", dict(arch="resnet", augmentation="none")),
    "R5": (_ARCH, "f1_macro", dict(arch="resnet", augmentation="chandrop", chandrop_p=0.2)),
    "R6": (_ARCH, "f1_macro", dict(arch="resnet_nores", augmentation="none")),
    "R7": (_ARCH, "f1_macro", dict(arch="resnet_nores", augmentation="chandrop", chandrop_p=0.2)),
    "R8": ("per_subject_metrics_cnn_loso.csv", "f1_macro", "seed_only"),
    "R9": ("per_subject_metrics_cnn_loso.csv", "f1_macro", "seed_only"),
    "R10": ("adabn_subjectwise.csv", "f1_macro", None),
    "R10pre": ("adabn_subjectwise.csv", "f1_pre_adabn", None),
    "R11": ("deep_coral_subjectwise.csv", "f1_macro", None),
    "R12": (_ARCH, "f1_macro", dict(arch="resnet_se", augmentation="chandrop", chandrop_p=0.2, window_ms_token="w400")),
    "R13": (_ARCH, "f1_macro", dict(arch="resnet_se", augmentation="chandrop", chandrop_p=0.1)),
    "R14": (_ARCH, "f1_macro", dict(arch="resnet_se", augmentation="gainjitter", gain_sd=0.30)),
    "R15": (_ARCH, "f1_macro", dict(arch="resnet_se", augmentation="chandrop", chandrop_p=0.5)),
    "R16": (_ARCH, "f1_macro", dict(arch="resnet_se", augmentation="gainjitter", gain_sd=0.50)),
    "R17": (_ARCH, "f1_macro", dict(arch="resnet_se", augmentation="chandrop", chandrop_p=0.3)),
}
ARM_DIR_KEY = {"R10pre": "R10"}          # R10pre is a second column of the R10 directory
INSTRUMENTED = ["R1", "R2", "R4", "R5"]  # the C17 arms

CONTRASTS = {   # id -> [(coefficient, key)]; key = arm | "OCC:"+arm | ENS_SOFT | ENS_STACK | "CONST:<value>"
    "C1": [(1, "R2"), (-1, "R1")], "C2": [(1, "R3"), (-1, "R2")], "C3": [(1, "R14"), (-1, "R13")],
    "C4": [(1, "R16"), (-1, "R15")], "C5": [(1, "R1"), (-1, "R4")],
    "C6": [(1, "R2"), (-1, "R1"), (-1, "R5"), (1, "R4")], "C7": [(1, "R5"), (-1, "R4"), (-1, "R7"), (1, "R6")],
    "C8": [(1, "R4"), (-1, "R6")], "C9": [(1, "R2"), (-1, "R10")], "C10": [(1, "R11"), (-1, "R10")],
    "C11": [(1, "R2"), (-1, "R11")], "C12": [(1, "R2"), (-1, f"CONST:{PUBLISHED_SVM}")],
    "C13": [(1, "ENS_SOFT"), (-1, "R2")], "C14": [(1, "R12"), (-1, "R2")], "C15": [(1, "R9"), (-1, "R8")],
    "C16a": [(1, "R13"), (-1, "R2")], "C16b": [(1, "R17"), (-1, "R2")], "C16c": [(1, "R15"), (-1, "R2")],
    "C17a": [(1, "OCC:R1"), (-1, "OCC:R2")], "C17b": [(1, "OCC:R4"), (-1, "OCC:R5")],
    "C13b": [(1, "ENS_STACK"), (-1, "ENS_SOFT")],
}
REGISTERED_BH_FAMILY = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
                        "C16c", "C17a", "C17b"]
NULL_CONTRASTS = ["C15", "C16a", "C16b"]
DESCRIPTIVE_CONTRASTS = ["C13b"]
ALL_CONTRAST_IDS = list(CONTRASTS)
HEADLINE_ARMS = {"R2": "R2", "ensemble": "ENS_SOFT", "R12": "R12", "global": "R10pre"}   # headline name -> key
OUTPUT_FILES = ("d1_reproduction_inputs.csv", "d1_contrasts.csv", "d1_headline_inputs.csv", "d1_arm_subject_f1.csv",
                "d1_c17_factors.csv")
# kept for callers that import the older names
EXTRACTABLE_CONTRASTS = ALL_CONTRAST_IDS
HEADLINE_ARM_NAMES = list(HEADLINE_ARMS)


def arm_seeds(arm: str) -> list[int]:
    return TIER_B_SEEDS if arm in TIER_B_ARMS else TIER_A_SEEDS


def _arm_dir(root: Path, arm: str, seed: int) -> Path:
    return root / f"results_kc23_d1_{ARM_DIR_KEY.get(arm, arm).lower()}_s{seed}"


def load_arm(root: Path, arm: str, seed: int) -> pd.Series:
    """Per-subject F1 of one arm at one seed. Raises on anything missing, mislabelled or incomplete."""
    fname, col, cfg = ARM_SPEC[arm]
    d = _arm_dir(root, arm, seed)
    p = d / fname
    if not p.exists():
        raise FileNotFoundError(f"missing input: {p}")
    if isinstance(cfg, dict):
        check_config(d, seed=seed, **cfg)
    elif cfg == "seed_only":
        check_config(d, seed=seed, arch=None, norm=None)
    df = pd.read_csv(p)
    if "model" in df.columns:                                   # per_subject_metrics_cnn_loso.csv
        df = df[df["model"] == "CNN"]
    if not {"subject", col} <= set(df.columns):
        raise ValueError(f"{d.name}/{fname} lacks subject/{col}")
    if df["subject"].duplicated().any() or len(df) != N_SUBJECTS or df[col].isna().any():
        raise ValueError(f"{d.name}: needs {N_SUBJECTS} unique non-NaN subjects, found {len(df)} rows")
    return df.set_index("subject")[col].astype(float).sort_index()


def load_published(root: Path) -> dict[str, pd.Series]:
    """Published runs from kc23_d1_published_runs.csv, each checked against the number the plan quotes."""
    path = root / "kc23_d1_published_runs.csv"
    if not path.exists():
        path = PUBLISHED_RUNS_CSV
    m = pd.read_csv(path)
    out = {}
    for _, r in m.iterrows():
        f = root / r["directory"] / r["subjectwise_file"]
        if not f.exists():
            raise FileNotFoundError(f"published run missing: {f}")
        df = pd.read_csv(f)
        col = r["f1_column"]
        if col not in df.columns or "subject" not in df.columns:
            raise ValueError(f"{f.name}: lacks subject/{col}")
        if df["subject"].duplicated().any() or len(df) != N_SUBJECTS or df[col].isna().any():
            raise ValueError(f"{r['directory']}: published run needs {N_SUBJECTS} unique non-NaN subjects")
        s = df.set_index("subject")[col].astype(float).sort_index()
        if pd.notna(r["quoted_value"]) and abs(float(s.mean()) - float(r["quoted_value"])) > float(r["tolerance"]):
            raise ValueError(f"{r['directory']} ({r['arm']}): mean {s.mean():.4f} is not the {r['quoted_value']} the "
                             f"plan quotes (tolerance {r['tolerance']}): wrong mapping, not used")
        out[str(r["arm"])] = s
    return out


def load_ensemble(root: Path, seed: int) -> pd.DataFrame:
    p = root / f"results_kc23_d1_ensemble_s{seed}" / f"ensemble_s{seed}.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing input: {p}")
    df = pd.read_csv(p)
    if not {"subject", "soft", "stacking"} <= set(df.columns) or df["subject"].duplicated().any() or len(df) != N_SUBJECTS:
        raise ValueError(f"{p.name}: needs subject/soft/stacking for {N_SUBJECTS} unique subjects")
    return df.set_index("subject").sort_index()


class Data:
    """Every value a contrast can use, by key and realization (a seed, or "published")."""

    def __init__(self):
        self.v: dict[str, dict] = {}

    def put(self, key, real, series):
        self.v.setdefault(key, {})[real] = series

    def realizations(self, key) -> set:
        return set(self.v.get(key, {}))


def load_all(root: Path, arms: list[str], with_ensemble=True, with_occlusion=True) -> Data:
    d = Data()
    for arm in arms:
        for seed in arm_seeds(arm):
            d.put(arm, seed, load_arm(root, arm, seed))
    if with_occlusion:
        for arm in INSTRUMENTED:
            for seed in arm_seeds(arm):
                run_dir = _arm_dir(root, arm, seed)
                f1 = load_f1(run_dir)
                d.put(f"OCC:{arm}", seed, load_instr(run_dir, set(f1.index), probes=False)["occlusion_sum"])
    if with_ensemble:
        for seed in TIER_A_SEEDS:
            e = load_ensemble(root, seed)
            d.put("ENS_SOFT", seed, e["soft"])
            d.put("ENS_STACK", seed, e["stacking"])
    for k, s in load_published(root).items():
        d.put(k, "published", s)
    return d


def _keys(terms):
    return [k for _, k in terms if not k.startswith("CONST:")]


def contrast_realizations(data: Data, terms) -> list:
    keys = _keys(terms)
    common = set.intersection(*(data.realizations(k) for k in keys)) if keys else set()
    return sorted([r for r in common if r != "published"]) + (["published"] if "published" in common else [])


def contrast_tier(terms) -> str:
    return "B" if any(k.split(":")[-1] in TIER_B_ARMS for _, k in terms) else "A"


def build_contrast_rows(data: Data, ids: list[str]) -> tuple[list[dict], dict[str, list]]:
    rows, incomplete = [], {}
    for cid in ids:
        terms = CONTRASTS[cid]
        reals = contrast_realizations(data, terms)
        tier = contrast_tier(terms)
        want = TIER_B_SEEDS if tier == "B" else TIER_A_SEEDS
        if any(w not in reals for w in want):
            incomplete[cid] = [w for w in want if w not in reals]
            continue
        first = data.v[_keys(terms)[0]]
        for real in reals:
            acc = pd.Series(0.0, index=first[real].index)
            for coef, key in terms:
                acc = acc + coef * (float(key.split(":")[1]) if key.startswith("CONST:") else data.v[key][real])
            for subj, dv in acc.items():
                rows.append({"contrast": cid, "tier": tier, "realization": real, "subject": int(subj), "diff": float(dv)})
    return rows, incomplete


def build_reproduction_inputs(root: Path) -> pd.DataFrame | None:
    try:
        r1, r2, r10 = load_arm(root, "R1", 42), load_arm(root, "R2", 42), load_arm(root, "R10pre", 42)
    except (FileNotFoundError, ValueError) as e:
        print(f"[d1-aggregate] reproduction inputs (R1/R2/R10-pre, seed 42): not complete ({e})")
        return None
    return pd.DataFrame([{"arm": "R1", "f1_mean": float(r1.mean())}, {"arm": "R2", "f1_mean": float(r2.mean())},
                         {"arm": "R10_pre", "f1_mean": float(r10.mean())}])


def build_headline_rows(data: Data) -> list[dict]:
    rows = []
    for name, key in HEADLINE_ARMS.items():
        seeds = [s for s in TIER_A_SEEDS if s in data.realizations(key)]
        if seeds != TIER_A_SEEDS:
            raise ValueError(f"headline arm {name}: seeds {seeds} present, need {TIER_A_SEEDS}")
        means = [float(data.v[key][s].mean()) for s in seeds]
        rows.append({"arm": name, "realization_mean": float(np.mean(means)), "realization_sd": float(np.std(means, ddof=1))})
    return rows


def build_arm_subject_f1(data: Data) -> pd.DataFrame:
    rows = []
    for key, byreal in data.v.items():
        if key.startswith("OCC:"):
            continue
        for real, s in byreal.items():
            for subj, v in s.items():
                rows.append({"arm": key, "realization": real, "subject": int(subj), "f1": float(v)})
    return pd.DataFrame(rows)


def build_c17_factors(data: Data) -> pd.DataFrame:
    rows = []
    for label, ref, arm in (("R2 against R1", "R1", "R2"), ("R5 against R4", "R4", "R5")):
        for seed in TIER_A_SEEDS:
            rows.append({"comparison": label, "realization": seed,
                         "factor": reduction_factor(data.v[f"OCC:{ref}"][seed], data.v[f"OCC:{arm}"][seed])})
    return pd.DataFrame(rows)


def run(out_dir: Path, root: Path, require: str | None = None) -> int:
    """Fail closed. require: "repro", "full" or None (whatever is ready, at least one output). Exit 1 and no output
    file on failure, so a file from an earlier partial run cannot stand in for a missing input."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (out_dir / name).unlink(missing_ok=True)
    written: dict[str, pd.DataFrame] = {}
    problems: list[str] = []
    repro = build_reproduction_inputs(root)
    if repro is not None:
        written["d1_reproduction_inputs.csv"] = repro
    elif require in ("repro", "full"):
        problems.append("reproduction inputs (R1/R2/R10-pre, seed 42) are not complete")

    if require == "full":
        try:
            data = load_all(root, list(ARM_SPEC))
            rows, incomplete = build_contrast_rows(data, ALL_CONTRAST_IDS)
            if incomplete:
                problems.append(f"contrasts not ready (missing realizations): {incomplete}")
            else:
                written["d1_contrasts.csv"] = pd.DataFrame(rows)
                written["d1_headline_inputs.csv"] = pd.DataFrame(build_headline_rows(data))
                written["d1_arm_subject_f1.csv"] = build_arm_subject_f1(data)
                written["d1_c17_factors.csv"] = build_c17_factors(data)
        except (FileNotFoundError, ValueError, KeyError) as e:
            problems.append(str(e))
    if require is None and not written and not problems:
        problems.append("nothing was ready to write")
    if problems:
        print("[d1-aggregate] FAIL, no output written: " + "; ".join(problems), file=sys.stderr)
        return 1
    for name, df in written.items():
        df.to_csv(out_dir / name, index=False)
    print(f"[d1-aggregate] wrote {sorted(written)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--require", choices=["repro", "full"], default=None,
                    help="repro: the seed-42 reproduction inputs must be complete; full: everything, every contrast "
                         "complete. Default: at least one output. Exit 1 otherwise.")
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.root).resolve(), args.require))


if __name__ == "__main__":
    main()
