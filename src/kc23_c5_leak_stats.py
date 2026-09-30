#!/usr/bin/env python3
"""
src/kc23_c5_leak_stats.py
=======================
KC-C5 stats/gate. docs/plans/EXPERIMENT_PLAN_KC23_CLASSICAL.md "KC-C5. Decomposing the
subject-dependent 'leak'", C5.3 and C5.4.

Plateau guard g*: the smallest g in {1,2,4,8} at which B-g changes by less than 0.5pt from B-(2g). No plateau by g=16 is
itself outcome L3.

  Delta_overlap  = P50 - P0
  Delta_autocorr = P0 - I-g*
  Delta_drift    = I-g* - B-g*

  L1: Delta_overlap >= 60% of (P50 - B-g*)         -> "overlap leak" stands, sized
  L2: Delta_drift   >= 40% of the total (P50 - B-g*) -> split-protocol effect, named components
  L3: no plateau by g=16, OR (g* != 1 AND B-g* differs from the published blocked SD by > 1pt)   -> ESCALATE
L1 and L2 are not exclusive; L3 is reported beside them when it fires (the decomposition itself is then untrustworthy).

Conformance pass, 26 September 2026 (docs/kc23/KC23_PREREG_CONFORMANCE.md); no C5 data existed:
  - The plan runs SVM, RF and LDA; only SVM was decomposed. The decomposition and the letters are now computed for each
    of the three models. The exit code escalates if any model's letter escalates.
  - DESIGN CORRECTION, Enam's ruling of 26 September (pre-data). The plan premised that the published Table 4.2 classical
    SD figures pool all 40 subjects in one cross-validation. src/b8_movement_blocked_sd.py's eval_sd loops over subjects: those
    figures are PER-SUBJECT models under both the random and the blocked split (the thesis caption is what is wrong; text
    item TA10). So per_subject is the PRIMARY cv-unit for every arm (P50, P0, B-g, I-g) on SIAT and ENABL3S, and the L3
    clause compares like with like (per-subject B-g* against the published per-subject blocked SD). W-B1 and the pooled
    arms are dropped: their only purpose was a naming question the code settles. (An earlier draft of this reader, written
    the same day, took the arms from the pooled cv-unit and added W-B1; it never saw data.)
  - A missing B-g or I-g arm was silently dropped and the gate carried on; every arm (P50, P0, B-1..16, I-1/4/16) is now
    required for every model, and a missing one is a fail-closed exit 20 with no letter line.
  - The plan runs I only at g in {1, 4, 16}. When the plateau lands on g* in {2, 8} there is no I-g* arm, so Delta_autocorr
    and Delta_drift, and with them L1 and L2, are not defined on the registered arms: the letter is "L-OUT" (exit 10),
    not the earlier blanket L3. (An extra I-g* run would be an amendment to the plan; not done here.)
  - Nothing fired, or a non-positive total (P50 - B-g* <= 0), is outside the plan's table: "L-NONE" / "L-OUT", exit 10,
    not an empty list that the queue's letter check could not read.
The published blocked SD for L3's second clause is results/b8_sd sd_new_blocked (per-subject model, g = 1), which is now the
like-for-like comparator; ENABL3S has none, and the clause is then reported as NOT evaluated. B-1 (per-subject, g = 1) against
that published figure is also reported, as a reproduction check (SIAT, SVM and RF).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import paired_test, print_gate_header, require_complete, write_no_outcome_verdict

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "kc23_c5_leak_"
MODELS = ("SVM", "RF", "LDA")
B_GUARDS = (1, 2, 4, 8, 16)
I_GUARDS = (1, 4, 16)
N_BY_DATASET = {"siat": 40, "enabl3s": 10}
PUBLISHED_B8_FILE = {"siat": "results/b8_sd/b8_base_w250_compare.csv"}


class InputError(Exception):
    pass


def load_published_blocked_sd(dataset: str, model: str, root: Path | None = None):
    """(value or None, note). Raises InputError if the dataset HAS a published file that is missing or lacks the model."""
    root = root or ROOT
    rel = PUBLISHED_B8_FILE.get(dataset)
    if rel is None:
        return None, (f"L3 published-figure clause NOT evaluated: no published blocked SD exists for dataset {dataset!r}")
    f = root / rel
    if not f.exists():
        raise InputError(f"published blocked-SD file missing: {f}")
    df = pd.read_csv(f)
    if "sd_new_blocked" not in df.columns:
        raise InputError(f"{f.name} has no sd_new_blocked column")
    row = df[df["model"] == model]
    if row.empty:      # the published file has SVM and RF only: no published figure exists for LDA, so the clause cannot be evaluated
        return None, f"L3 published-figure clause NOT evaluated for {model}: {rel} has no published blocked SD for it"
    v = float(row["sd_new_blocked"].iloc[0])
    return v, f"L3 published-figure clause evaluated against {model} sd_new_blocked {v:.4f} ({rel})"


def find_plateau(b_by_g: dict[int, np.ndarray], guards=B_GUARDS) -> tuple[int | None, dict]:
    """b_by_g: {guard: per-subject F1 array}. Returns (g_star or None, detail)."""
    detail = {}
    for i, g in enumerate(guards[:-1]):
        g2 = guards[i + 1]
        change_pp = abs(b_by_g[g2].mean() - b_by_g[g].mean()) * 100.0
        detail[g] = change_pp
        if change_pp < 0.5:
            return g, detail
    return None, detail


def classify_l(p50: np.ndarray, p0: np.ndarray, i_by_g: dict, b_by_g: dict, g_star: int | None,
               published_blocked_sd: float | None) -> tuple[list[str], dict]:
    if g_star is None:
        return ["L3"], {"g_star": None, "note": "no plateau by g = 16"}
    letters = []
    b = b_by_g[g_star]
    total_pp = (p50.mean() - b.mean()) * 100.0
    overlap_pp = (p50.mean() - p0.mean()) * 100.0
    detail = {"g_star": g_star, "total_pp": total_pp, "overlap_pp": overlap_pp,
              "autocorr_pp": float("nan"), "drift_pp": float("nan")}
    if published_blocked_sd is not None and g_star != 1 and abs(b.mean() - published_blocked_sd) * 100.0 > 1.0:
        letters.append("L3")
    if g_star not in i_by_g:
        letters.append("L-OUT")             # no I-g* among the registered arms: Delta_autocorr, Delta_drift undefined
        return letters, detail
    i = i_by_g[g_star]
    detail["autocorr_pp"] = (p0.mean() - i.mean()) * 100.0
    detail["drift_pp"] = (i.mean() - b.mean()) * 100.0
    if total_pp <= 0:
        letters.append("L-OUT")             # blocked is not below pooled: the shares are undefined
        return letters, detail
    if overlap_pp >= 0.60 * total_pp:
        letters.append("L1")
    if detail["drift_pp"] >= 0.40 * total_pp:
        letters.append("L2")
    if not letters:
        letters.append("L-NONE")            # neither share threshold met and no L3
    return letters, detail


def tag_from_search_root(search_root: Path) -> str:
    name = search_root.name
    dataset = name[len(PREFIX):] if name.startswith(PREFIX) else name
    return f"kc23_c5_{dataset}"


def find_scheme_file(search_root: Path, tag: str, scheme: str, guard: float | None = None,
                     cv_unit: str = "per_subject") -> Path | None:
    """Each scheme job writes into its OWN subdirectory of search_root, searched one level deep. The cv-unit is part of
    the file name, so a stray pooled file (a smoke test) can never be read as an arm."""
    pat = f"*/b8_{tag}_{scheme}_g{guard:g}_{cv_unit}_subjectwise.csv" if guard is not None \
        else f"*/b8_{tag}_{scheme}_g*_{cv_unit}_subjectwise.csv"
    matches = sorted(search_root.glob(pat))
    return matches[0] if matches else None


def load_model_f1(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if model not in df.columns:
        raise InputError(f"{path.name}: no column for model {model}")
    return df[["subject", model]].rename(columns={model: "f1_macro"})


def resolve_search_root(out_dir: Path) -> Path:
    return out_dir if out_dir.name.startswith(PREFIX) else out_dir.parent


def _fail(search_root: Path, reason: str) -> int:
    print(f"[C5] MISSING (fail closed): {reason}", file=sys.stderr)
    write_no_outcome_verdict(search_root / "C5_VERDICT.md", "KC-C5 verdict", reason)
    (search_root / "C5_decomposition.csv").unlink(missing_ok=True)
    return 20


def run(out_dir: Path, published_blocked_sd: float | None = None, models=MODELS, root: Path | None = None) -> int:
    """out_dir: results/kc23_c5_leak_<dataset> itself, or one of its scheme subdirectories (the older gate wiring)."""
    search_root = resolve_search_root(out_dir)
    tag = tag_from_search_root(search_root)
    dataset = search_root.name[len(PREFIX):] if search_root.name.startswith(PREFIX) else search_root.name
    n_subj = N_BY_DATASET.get(dataset, 40)

    def load(model, scheme, guard=None, cv_unit="per_subject", label=""):
        f = find_scheme_file(search_root, tag, scheme, guard, cv_unit)
        if f is None:
            raise InputError(f"no {label or scheme} ({cv_unit}) file under {search_root} "
                             f"(pattern */b8_{tag}_{scheme}_g{'*' if guard is None else f'{guard:g}'}_{cv_unit}_subjectwise.csv)")
        return require_complete(load_model_f1(f, model), n_subj, f"{label or scheme} {model}")

    rows, lines, notes, fired_all = [], [], [], []
    try:
        for model in models:
            p50 = load(model, "pooled_random", label="P50")
            p0 = load(model, "pooled_random_nonoverlap", label="P0")
            b_by_g = {g: load(model, "blocked", g, label=f"B-{g}") for g in B_GUARDS}
            i_by_g = {g: load(model, "interleaved", g, label=f"I-{g}") for g in I_GUARDS}
            if published_blocked_sd is None:
                pub, note = load_published_blocked_sd(dataset, model, root)
            else:
                pub, note = published_blocked_sd, f"L3 published-figure clause evaluated against the supplied value {pub_fmt(published_blocked_sd)}"
            g_star, plateau_detail = find_plateau(b_by_g)
            letters, detail = classify_l(p50, p0, i_by_g, b_by_g, g_star, pub)
            row = {"model": model, **detail, "letters": ",".join(letters),
                   "p50": p50.mean(), "p0": p0.mean(), "b_gstar": (b_by_g[g_star].mean() if g_star else float("nan")),
                   "b1": b_by_g[1].mean()}
            if pub is not None:
                row["b1_minus_published_blocked_pp"] = (b_by_g[1].mean() - pub) * 100.0
            rows.append(row)
            notes.append(f"{model}: {note}. Plateau detail (pt between consecutive guards): "
                         f"{ {k: round(v, 2) for k, v in plateau_detail.items()} }, g*={g_star}.")
            lines.append(f"**{model} outcome: {', '.join(letters)}**")
            fired_all.extend(letters)
    except (InputError, FileNotFoundError, ValueError) as e:
        return _fail(search_root, str(e))

    print_gate_header("KC-C5", "; ".join(f"{r['model']} {r['letters']}" for r in rows),
                      "L1 overlap leak stands; L2 split-protocol components; L3 ESCALATE; L-OUT / L-NONE outside the table.")
    search_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(search_root / "C5_decomposition.csv", index=False)     # SVM first: the CNN job generator reads row 0
    repro_md = "\n".join(f"- {r['model']}: B-1 {r['b1']:.4f}, {r['b1_minus_published_blocked_pp']:+.2f} pt against the published "
                         f"per-subject blocked SD" for r in rows if "b1_minus_published_blocked_pp" in r) or "(no published figure for this dataset)"
    dec_md = "\n".join(f"- {r['model']}: total {r['total_pp']:+.2f} pt, overlap {r['overlap_pp']:+.2f}, autocorrelation "
                       f"{r['autocorr_pp']:+.2f}, drift {r['drift_pp']:+.2f} (g*={r['g_star']})" for r in rows if r.get("total_pp") is not None
                       and not (isinstance(r["total_pp"], float) and np.isnan(r["total_pp"])))
    (search_root / "C5_VERDICT.md").write_text(
        f"# KC-C5 verdict ({dataset})\n\n" + "\n".join(lines) + "\n\n## Decomposition\n\n" + (dec_md or "(none defined)") +
        "\n\n## B-1 against the published per-subject blocked SD (reproduction check)\n\n" + repro_md + "\n\n## Notes\n\n"
        + "\n".join(notes) + "\n",
        encoding="utf-8")

    if "L3" in fired_all:
        return 20
    if any(l in ("L-OUT", "L-NONE") for l in fired_all):
        return 10
    return 0


def pub_fmt(v) -> str:
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--published-blocked-sd", type=float, default=None)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    sys.exit(run(Path(args.out), args.published_blocked_sd, tuple(m.strip().upper() for m in args.models.split(",") if m.strip())))


if __name__ == "__main__":
    main()
