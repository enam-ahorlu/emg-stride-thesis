#!/usr/bin/env python3
"""
kc23_s3_benchmark.py
======================
KC-S3 step 3. EXPERIMENT_PLAN_KC23_DEPLOYMENT.md "KC-S3. The active-only benchmark, completed", S3.2 items 2 and 3 and S3.3:
"A single benchmark table (results_kc23_s3_active_benchmark/benchmark_active_only.csv) and S3_VERDICT.md, reporting any
change in the class hierarchy for families not previously run. There are no halting letters."

One row per family x normalization on the active-only windows/features: macro-F1 over the 40 held-out subjects (mean, SD) and
the DNS->WAK critical-error rate (true DNS windows predicted WAK, pooled over windows and averaged over subjects; the
definition run_aonly_ensemble.py already uses). Families: LDA, SVM, RF, SimpleEMGCNN, ResNet-SE, ResNet-SE+CD and the
ensemble; SVM-X and HGB only "if KC-C3 lands P2 or P3" (read from C3_VERDICT.md; when the verdict is absent the condition
cannot be decided and the benchmark stops).

Cells that already existed are read from their published directories (run only the missing cells): SVM and RF
(results_aonly_global, results_aonly_persubj), ResNet-SE+CD per-subject (results_aonly_resnet_se_cd_persubj) and the ensemble
(results_aonly_ensemble, member "soft", per-subject normalization only). Every other cell is read from its results_kc23_s3_*
directory. Per-window predictions come from predictions_folds/*_y_{true,pred}.npy (classical, LDA) or proba/*_subKK.npz
(deep). A missing or incomplete cell stops the script with exit 2, no benchmark file and a verdict with no outcome line: an
absent cell is not a low score.

The class hierarchy is reported per normalization (families ranked by F1), with each family marked "prior" or "new"; the
"full class set" column is the same family's F1 on the rest-included benchmark where that run exists as a directory, so a
family whose order against another changes between the two benchmarks is listed.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import write_no_outcome_verdict

ROOT = Path(__file__).resolve().parent
N_SUBJECTS = 40
DNS, WAK = 0, 3
NORMS = ["global", "per_subject"]
BASE_FAMILIES = ["LDA", "SVM", "RF", "simple", "resnet_se", "resnet_se_cd", "ensemble"]
CONDITIONAL_FAMILIES = ["SVMX", "HGB"]          # only if KC-C3 lands P2 or P3
LABEL = {"LDA": "LDA", "SVM": "SVM", "RF": "RF", "simple": "SimpleEMGCNN", "resnet_se": "ResNet-SE", "resnet_se_cd": "ResNet-SE+CD",
         "ensemble": "Ensemble (soft vote)", "SVMX": "SVM-X", "HGB": "HGB"}

# (family, norm) -> (kind, directory, model token)   kind: classical | lda | cnn | ensemble
PRIOR = {
    ("SVM", "global"): ("classical", "results_aonly_global", "SVM"),
    ("RF", "global"): ("classical", "results_aonly_global", "RF"),
    ("SVM", "per_subject"): ("classical", "results_aonly_persubj", "SVM"),
    ("RF", "per_subject"): ("classical", "results_aonly_persubj", "RF"),
    ("resnet_se_cd", "per_subject"): ("cnn", "results_aonly_resnet_se_cd_persubj", None),
    ("ensemble", "per_subject"): ("ensemble", "results_aonly_ensemble", "soft"),
}
# The same family on the rest-included ("full class set") benchmark, where a real directory holds it.
FULL_SET = {
    ("SVM", "per_subject"): ("results_loso_freq_persubj", "SVM"), ("RF", "per_subject"): ("results_loso_freq_persubj", "RF"),
    ("LDA", "per_subject"): ("results_lda_persubj", None), ("LDA", "global"): ("results_lda_global", None),
    ("resnet_se_cd", "per_subject"): ("results_cnn_aug_resnet_se_chandrop", None),
}


class InputError(Exception):
    pass


def cell_source(family: str, norm: str) -> tuple[str, str, str | None, bool]:
    """(kind, dir, token, prior)"""
    if (family, norm) in PRIOR:
        k, d, t = PRIOR[(family, norm)]
        return k, d, t, True
    if family == "LDA":
        return "lda", f"results_kc23_s3_lda_{norm}", None, False
    if family in ("simple", "resnet_se", "resnet_se_cd"):
        return "cnn", f"results_kc23_s3_{family}_{norm}", None, False
    if family in ("SVMX", "HGB"):
        return "classical", f"results_kc23_s3_{family.lower()}_{norm}", family.replace("SVMX", "SVM"), False
    raise InputError(f"no source defined for {family} / {norm}")


def crit(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    m = y_true == DNS
    return int((y_pred[m] == WAK).sum()), int(m.sum())


def _subjectwise_f1(d: Path, kind: str, token: str | None) -> pd.DataFrame:
    if kind == "lda":
        f = d / "lda_subjectwise.csv"
    elif kind == "cnn":
        f = d / "cnn_arch_subjectwise.csv"
    elif kind == "ensemble":
        f = d / "aonly_ensemble_subjectwise.csv"
    else:
        cands = sorted(d.glob(f"*__{token}_nested_loso_subjectwise.csv"))
        f = cands[0] if cands else d / f"*__{token}_nested_loso_subjectwise.csv"
    if not f.exists():
        raise InputError(f"missing input: {f}")
    df = pd.read_csv(f)
    if kind == "ensemble":
        df = df[df["member"] == token].rename(columns={"macro_f1": "f1_macro"})
    scol = "subject" if "subject" in df.columns else "heldout_subject"
    df = df.rename(columns={scol: "subject"})[["subject", "f1_macro"] + (["crit_err"] if kind == "ensemble" else [])]
    if df["subject"].duplicated().any() or len(df) != N_SUBJECTS:
        raise InputError(f"{f}: needs {N_SUBJECTS} unique subjects, found {len(df)} rows")
    return df.sort_values("subject").reset_index(drop=True)


def _window_predictions(d: Path, kind: str, token: str | None, subject: int):
    if kind == "cnn":
        cands = sorted((d / "proba").glob(f"*_sub{subject:02d}.npz"))
        if not cands:
            raise InputError(f"missing input: {d / 'proba'}/*_sub{subject:02d}.npz (per-window probabilities for the DNS->WAK rate)")
        with np.load(cands[0]) as z:
            return z["y_true"].astype(int), z["proba"].argmax(1)
    tok = "LDA" if kind == "lda" else token
    yt = sorted((d / "predictions_folds").glob(f"*_{tok}_sub{subject:02d}_y_true.npy"))
    yp = sorted((d / "predictions_folds").glob(f"*_{tok}_sub{subject:02d}_y_pred.npy"))
    if not yt or not yp:
        raise InputError(f"missing input: {d / 'predictions_folds'}/*_{tok}_sub{subject:02d}_y_{{true,pred}}.npy")
    return np.load(yt[0]).astype(int), np.load(yp[0]).astype(int)


def cell_row(family: str, norm: str, root: Path) -> dict:
    kind, dname, token, prior = cell_source(family, norm)
    d = root / dname
    f1 = _subjectwise_f1(d, kind, token)
    if kind == "ensemble":
        # results_aonly_ensemble stores the per-subject critical-error rate itself; pooled window counts are not kept
        rates = f1["crit_err"].to_numpy(float)
        pooled = float("nan")
    else:
        num = den = 0
        rates = []
        for s in f1["subject"]:
            yt, yp = _window_predictions(d, kind, token, int(s))
            n, m = crit(yt, yp)
            num, den = num + n, den + m
            rates.append(n / m if m else np.nan)
        pooled = num / den if den else float("nan")
        rates = np.array(rates, float)
    full = float("nan")
    if (family, norm) in FULL_SET:
        fd, ft = FULL_SET[(family, norm)]
        try:
            full = float(_subjectwise_f1(root / fd, "classical" if ft else ("lda" if "lda" in fd else "cnn"), ft)["f1_macro"].mean())
        except InputError:
            full = float("nan")
    return {"family": family, "label": LABEL[family], "normalization": norm, "f1_mean": float(f1["f1_macro"].mean()),
            "f1_sd": float(f1["f1_macro"].std(ddof=1)), "n_subjects": len(f1),
            "dns_to_wak_pooled": pooled, "dns_to_wak_subject_mean": float(np.nanmean(rates)),
            "full_class_set_f1_mean": full, "provenance": "prior" if prior else "new", "source": dname}


def c3_status(root: Path) -> tuple[bool, str]:
    """(SVM-X and HGB cells required?, sentence for the verdict). The plan's condition is "if KC-C3 lands P2 or P3"; a
    P-OUT (outside the pre-registered grid) does not meet it, so the extra cells are not required (Enam, 26 September)."""
    v = root / "results_kc23_c3_ensemble" / "C3_VERDICT.md"
    if not v.exists():
        raise InputError(f"{v} missing: whether SVM-X and HGB belong in the benchmark depends on KC-C3 landing P2 or P3")
    m = re.search(r"\*\*Outcomes:\s*([^*\n]+)\*\*", v.read_text(encoding="utf-8"))
    if not m:
        raise InputError(f"{v} has no 'Outcomes:' line (KC-C3 did not produce a letter)")
    letters = [t.strip() for t in m.group(1).split(",") if t.strip()]
    if {"P2", "P3"} & set(letters):
        return True, f"KC-C3 landed {', '.join(letters)}: the SVM-X and HGB cells are included."
    if "P-OUT" in letters:
        return False, ("KC-C3 landed P-OUT (outside the pre-registered grid). The plan's condition for the extra SVM-X and HGB "
                       "cells is P2 or P3, so they are not required and were not run.")
    return False, f"KC-C3 landed {', '.join(letters)}, not P2 or P3: the SVM-X and HGB cells are not required."


def hierarchy(table: pd.DataFrame) -> tuple[str, list[str]]:
    """Per-norm ranking text and the list of pairs whose order differs from the full-class-set benchmark."""
    lines, flips = [], []
    for norm in NORMS:
        t = table[table["normalization"] == norm].sort_values("f1_mean", ascending=False)
        lines.append(f"- {norm}: " + " > ".join(f"{r.label} {r.f1_mean:.3f}{'' if r.provenance == 'prior' else ' (new)'}" for r in t.itertuples()))
        have = t.dropna(subset=["full_class_set_f1_mean"])
        for i, a in enumerate(have.itertuples()):
            for b in list(have.itertuples())[i + 1:]:
                if a.full_class_set_f1_mean < b.full_class_set_f1_mean:      # a beats b here but not on the full set
                    flips.append(f"{norm}: {a.label} ({a.f1_mean:.3f}) ranks above {b.label} ({b.f1_mean:.3f}) on active-only windows "
                                 f"but below it on the full class set ({a.full_class_set_f1_mean:.3f} against {b.full_class_set_f1_mean:.3f})")
    return "\n".join(lines), flips


def run(out_dir: Path, root: Path | None = None) -> int:
    root = root or ROOT
    try:
        extra_needed, c3_sentence = c3_status(root)
        families = BASE_FAMILIES + (CONDITIONAL_FAMILIES if extra_needed else [])
        rows = [cell_row(f, n, root) for f in families for n in NORMS
                if not (f == "ensemble" and n == "global")]          # the published active-only ensemble is per-subject only
    except InputError as e:
        print(f"[S3] FAIL (no benchmark written): {e}", file=sys.stderr)
        write_no_outcome_verdict(out_dir / "S3_VERDICT.md", "KC-S3 verdict", str(e))
        (out_dir / "benchmark_active_only.csv").unlink(missing_ok=True)
        return 2
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "benchmark_active_only.csv", index=False)
    ranking, flips = hierarchy(table)
    md = ["| family | normalization | F1 (mean, SD) | DNS to WAK (pooled / subject mean) | full class set F1 | source |", "|---|---|---|---|---|---|"]
    for r in table.itertuples():
        md.append(f"| {r.label} | {r.normalization} | {r.f1_mean:.4f} ({r.f1_sd:.4f}) | "
                  f"{'n/a' if np.isnan(r.dns_to_wak_pooled) else f'{r.dns_to_wak_pooled:.3f}'} / {r.dns_to_wak_subject_mean:.3f} | "
                  f"{'n/a' if np.isnan(r.full_class_set_f1_mean) else f'{r.full_class_set_f1_mean:.4f}'} | {r.source} ({r.provenance}) |")
    new_fams = sorted({r.label for r in table.itertuples() if r.provenance == "new"})
    (out_dir / "S3_VERDICT.md").write_text(
        "# KC-S3 verdict\n\nDescriptive only (no halting letters). Active-only benchmark, SIAT-LLMD, 250 ms, 40 held-out subjects.\n\n"
        + "\n".join(md) + "\n\n## Class hierarchy\n\n" + ranking + "\n\nFamilies run for the first time on active-only windows: "
        + (", ".join(new_fams) or "none") + ".\n\n"
        + ("Order changes against the full-class-set benchmark:\n\n" + "\n".join(f"- {f}" for f in flips) + "\n" if flips else
           "No pair of families with a full-class-set counterpart changes order.\n")
        + f"\n{c3_sentence}\n"
        + "\nThe ensemble row is the published active-only soft vote (results_aonly_ensemble), per-subject normalization only; its "
          "DNS to WAK rate is the per-subject mean it stores (no pooled figure).\n", encoding="utf-8")
    print(f"[S3] benchmark: {len(table)} cells written")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results_kc23_s3_active_benchmark")
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
