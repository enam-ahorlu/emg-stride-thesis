#!/usr/bin/env python3
"""
kc23_c5_leak_stats.py
=======================
KC-C5 stats/gate. EXPERIMENT_PLAN_KC23_CLASSICAL.md "KC-C5. Decomposing the
subject-dependent 'leak'", C5.4.

Plateau guard g*: the smallest g in {1,2,4,8,16} at which B-g changes by less
than 0.5pt from B-(2g). No plateau by g=16 is itself outcome L3.

Delta_overlap = P50 - P0
Delta_autocorr = P0 - I-g*
Delta_drift = I-g* - B-g*

  L1: Delta_overlap >= 60% of (P50 - B-g*)         -> "overlap leak" stands, sized
  L2: Delta_drift >= 40% of the total (P50 - B-g*)  -> split-protocol effect, named components
  L3: no plateau by g=16, OR (g* != 1 AND B-g* differs from the published blocked
      SD figure by > 1pt)                            -> ESCALATE

L1 and L2 are not mutually exclusive (both can fire); L3 overrides both when
it fires (the decomposition itself is untrustworthy).

Published published_blocked_sd: the pre-KC23 published movement-blocked SD
figure (g implicitly 1, from results_b8_sd), passed in or read from the
existing results_b8_sd/b8_base_w250_compare.csv sd_new_blocked column
(mean over models given, or a single value for one model).

File discovery, fixed 2026-09-24: b8_movement_blocked_sd.py's --scheme path
(run_single_scheme) writes b8_<tag>_<scheme>_g<guard>_<cv_unit>_subjectwise.csv
with one WIDE column per model (subject, SVM, RF, LDA) -- not the
p50_subjectwise.csv / b{g}_subjectwise.csv narrow f1_macro files this script
originally assumed (which nothing ever wrote; the gate would have crashed the
first time it was actually invoked with real data). <tag> is derived from
--out's own directory name (results_kc23_c5_leak_siat -> kc23_c5_siat), since
the queue always invokes a gate_script as `python <gate> --out <out_dir>`
with no other arguments. The decomposition runs on one model at a time
(--model, default SVM, the classical headline model used throughout this
programme); a missing scheme file is a FAIL, never a fallback.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import paired_test, print_gate_header, write_no_outcome_verdict

ROOT = Path(__file__).resolve().parent
# The published movement-blocked SD figure that L3's second clause compares B-g* against, per dataset. The queue
# invokes this gate with --out only, so before 2026-09-25 published_blocked_sd was always None there and that L3
# clause was silently never evaluated. ENABL3S has no published blocked-SD file (results_b8_sd's "ext" rows are
# SIAT's extended features, n = 40), so for it the clause is reported as NOT evaluated rather than skipped quietly.
PUBLISHED_B8_FILE = {"siat": "results_b8_sd/b8_base_w250_compare.csv"}


def load_published_blocked_sd(dataset: str, model: str):
    """(value or None, note). Raises FileNotFoundError/ValueError if the dataset HAS a published file that is
    missing or lacks the model: that is a missing input, not a reason to skip the clause."""
    rel = PUBLISHED_B8_FILE.get(dataset)
    if rel is None:
        return None, (f"L3 published-figure clause NOT evaluated: no published blocked SD exists for dataset "
                      f"{dataset!r}")
    f = ROOT / rel
    if not f.exists():
        raise FileNotFoundError(f"published blocked-SD file missing: {f}")
    df = pd.read_csv(f)
    row = df[df["model"] == model]
    if row.empty or "sd_new_blocked" not in df.columns:
        raise ValueError(f"{f.name} has no sd_new_blocked row for model {model!r}")
    v = float(row["sd_new_blocked"].iloc[0])
    return v, f"L3 published-figure clause evaluated against {model} sd_new_blocked {v:.4f} ({rel})"


def find_plateau(b_by_g: dict[int, np.ndarray], guards=(1, 2, 4, 8, 16)) -> tuple[int | None, dict]:
    """b_by_g: {guard: per-subject F1 array}. Returns (g_star or None, detail)."""
    detail = {}
    for i, g in enumerate(guards[:-1]):
        g2 = guards[i + 1]
        if g not in b_by_g or g2 not in b_by_g:
            continue
        change_pp = abs(b_by_g[g2].mean() - b_by_g[g].mean()) * 100.0
        detail[g] = change_pp
        if change_pp < 0.5:
            return g, detail
    return None, detail


def classify_l(p50: np.ndarray, p0: np.ndarray, i_gstar: np.ndarray, b_gstar: np.ndarray,
              g_star: int | None, published_blocked_sd: float | None) -> tuple[list[str], dict]:
    letters = []
    total_pp = (p50.mean() - b_gstar.mean()) * 100.0 if g_star is not None else float("nan")
    overlap_pp = (p50.mean() - p0.mean()) * 100.0
    autocorr_pp = (p0.mean() - i_gstar.mean()) * 100.0 if g_star is not None else float("nan")
    drift_pp = (i_gstar.mean() - b_gstar.mean()) * 100.0 if g_star is not None else float("nan")

    if g_star is None:
        letters.append("L3")
    else:
        if published_blocked_sd is not None and g_star != 1:
            gap = abs(b_gstar.mean() - published_blocked_sd) * 100.0
            if gap > 1.0:
                letters.append("L3")
        if total_pp and total_pp != 0 and not np.isnan(total_pp):
            if overlap_pp >= 0.60 * total_pp:
                letters.append("L1")
            if drift_pp >= 0.40 * total_pp:
                letters.append("L2")

    detail = {"g_star": g_star, "total_pp": total_pp, "overlap_pp": overlap_pp,
             "autocorr_pp": autocorr_pp, "drift_pp": drift_pp}
    return letters, detail


def tag_from_search_root(search_root: Path) -> str:
    name = search_root.name
    prefix = "results_kc23_c5_leak_"
    dataset = name[len(prefix):] if name.startswith(prefix) else name
    return f"kc23_c5_{dataset}"


def find_scheme_file(search_root: Path, tag: str, scheme: str, guard: float | None = None) -> Path | None:
    """Each scheme job writes into its OWN subdirectory of search_root
    (results_kc23_c5_leak_<dataset>/<scheme_tag>/...) -- never a shared
    directory, since is_complete() would then mark the second job to reach
    that directory "already done" the instant the first one finishes writing
    ANY *subjectwise.csv there. Searched one level deep accordingly."""
    pat = (f"*/b8_{tag}_{scheme}_g{guard:g}_*_subjectwise.csv" if guard is not None
          else f"*/b8_{tag}_{scheme}_g*_*_subjectwise.csv")
    matches = sorted(search_root.glob(pat))
    return matches[0] if matches else None


def load_model_f1(path: Path, model: str) -> "pd.DataFrame":
    df = pd.read_csv(path)
    return df[["subject", model]].rename(columns={model: "f1_macro"})


def run(out_dir: Path, published_blocked_sd: float | None = None, model: str = "SVM") -> int:
    """out_dir: the triggering job's OWN subdirectory (results_kc23_c5_leak_
    <dataset>/<scheme_tag>); its PARENT is where every sibling scheme's own
    subdirectory lives, and where this gate's verdict is written."""
    from kc23_stats_common import require_complete
    search_root = out_dir.parent
    tag = tag_from_search_root(search_root)

    def load(scheme, guard=None, label=""):
        f = find_scheme_file(search_root, tag, scheme, guard)
        if f is None:
            print(f"[C5] MISSING: no {label or scheme} file found under {search_root} (pattern "
                 f"*/b8_{tag}_{scheme}_g*_*_subjectwise.csv)", file=sys.stderr)
            return None
        return require_complete(load_model_f1(f, model), 40, label or scheme)

    p50 = load("pooled_random", label="P50")
    p0 = load("pooled_random_nonoverlap", label="P0")
    if p50 is None or p0 is None:
        write_no_outcome_verdict(search_root / "C5_VERDICT.md", "KC-C5 verdict",
                                 "the pooled-random (P50) or pooled non-overlap (P0) input is missing")
        return 20

    if published_blocked_sd is None:
        dataset = search_root.name[len("results_kc23_c5_leak_"):] if search_root.name.startswith(
            "results_kc23_c5_leak_") else search_root.name
        try:
            published_blocked_sd, published_note = load_published_blocked_sd(dataset, model)
        except (FileNotFoundError, ValueError) as e:
            print(f"[C5] FAIL (no outcome computed): {e}", file=sys.stderr)
            write_no_outcome_verdict(search_root / "C5_VERDICT.md", "KC-C5 verdict", str(e))
            return 20
    else:
        published_note = f"L3 published-figure clause evaluated against the supplied value {published_blocked_sd:.4f}"
    print(f"[C5] {published_note}")

    b_by_g = {}
    for g in (1, 2, 4, 8, 16):
        f1 = load("blocked", guard=float(g), label=f"B-{g}")
        if f1 is not None:
            b_by_g[g] = f1
    i_by_g = {}
    for g in (1, 4, 16):
        f1 = load("interleaved", guard=float(g), label=f"I-{g}")
        if f1 is not None:
            i_by_g[g] = f1

    g_star, plateau_detail = find_plateau(b_by_g)
    print(f"[C5] plateau detail (delta pp between consecutive guards): {plateau_detail}, g*={g_star}")

    if g_star is not None and g_star in i_by_g and g_star in b_by_g:
        letters, detail = classify_l(p50, p0, i_by_g[g_star], b_by_g[g_star], g_star, published_blocked_sd)
    else:
        letters, detail = (["L3"], {"g_star": g_star, "note": "no matching I-g or B-g for g_star"})

    reading = []
    if "L1" in letters:
        reading.append("Overlap leak stands as named, with its size.")
    if "L2" in letters:
        reading.append("Reported as a split-protocol effect with components; 'leak' limited to overlap.")
    if "L3" in letters:
        reading.append("ESCALATE. Table 4.2 and the abstract's gap figure change.")
    if not letters:
        reading.append("Neither L1 nor L2 threshold met; report the raw decomposition.")

    print_gate_header("KC-C5", ",".join(letters) or "none", " ".join(reading))
    print(f"  {detail}")

    search_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([detail]).to_csv(search_root / "C5_decomposition.csv", index=False)
    (search_root / "C5_VERDICT.md").write_text(
        f"# KC-C5 verdict\n\n**Outcome(s): {letters}**\n\n{' '.join(reading)}\n\n{published_note}.\n", encoding="utf-8")

    return 20 if "L3" in letters else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--published-blocked-sd", type=float, default=None)
    ap.add_argument("--model", default="SVM", choices=["SVM", "RF", "LDA"])
    args = ap.parse_args()
    sys.exit(run(Path(args.out), args.published_blocked_sd, args.model))


if __name__ == "__main__":
    main()
