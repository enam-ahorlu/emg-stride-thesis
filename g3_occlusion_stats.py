#!/usr/bin/env python3
"""
g3_occlusion_stats.py
=====================
W-2 Stage G3, re-scoped. Does channel dropout change which electrodes the model
leans on, and does it make that reliance transfer across people?

Reads two instrumented runs and applies the pre-registered rule in
EXPERIMENT_PLAN_G3_OCCLUSION.md sections 3 and 4. No GPU.

  results_cd_resnet_nose_chandrop/instr/occlusion.csv   channel dropout p=0.2  (G1)
  results_g3_noaug_instr/instr/occlusion.csv            no augmentation        (G3)

Both are --arch resnet, SE-free, per-subject normalization, seed 42. Estimators
come from window_ablation_stats.py, verified by hand in W-1.

Usage:
  python g3_occlusion_stats.py
  python g3_occlusion_stats.py --cd <dir> --base <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from window_ablation_stats import bca_ci, cohens_d_paired  # verified in W-1

SEED = 42
N_PERM = 10_000
N_CH = 9


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_profiles(path: Path) -> tuple[pd.DataFrame, int]:
    """Return a (subjects x 9) frame of drop_pp, and the count of negative entries."""
    if not path.exists():
        sys.exit(f"missing {path}. Has run_g3_occlusion.sh completed?")
    df = pd.read_csv(path)
    need = {"subject", "channel", "drop_pp", "f1_full"}
    if not need.issubset(df.columns):
        sys.exit(f"{path} lacks {need - set(df.columns)}")
    if df.duplicated(["subject", "channel"]).any():
        sys.exit(f"{path} has duplicate (subject, channel) rows; a --resume has "
                 "double-appended. Delete the instr directory and re-run.")
    wide = df.pivot(index="subject", columns="channel", values="drop_pp").sort_index()
    if wide.shape[1] != N_CH:
        sys.exit(f"{path} has {wide.shape[1]} channels, expected {N_CH}")
    if wide.isna().any().any():
        sys.exit(f"{path} has missing cells; the run is incomplete")
    return wide, int((wide.to_numpy() < 0).sum())


def normalized(wide: pd.DataFrame) -> np.ndarray:
    """Clip negatives to zero and scale each subject's profile to sum to one.

    Normalizing removes the scale difference between a 0.825 model and a 0.760
    one, which a raw spread comparison would otherwise partly measure.
    """
    a = np.clip(wide.to_numpy(), 0.0, None)
    s = a.sum(axis=1, keepdims=True)
    flat = (s.ravel() <= 0)
    if flat.any():
        print(f"  WARNING: {int(flat.sum())} subject(s) have no positive drop at all; "
              "their profile is set uniform and they carry no information here",
              file=sys.stderr)
        a[flat.ravel()] = 1.0 / N_CH
        s[flat] = 1.0
    return a / s


# --------------------------------------------------------------------------
# the two quantities
# --------------------------------------------------------------------------
def concentration(prof: np.ndarray) -> np.ndarray:
    """Per subject: SD across the 9 normalized channel weights. Lower = flatter."""
    return prof.std(axis=1, ddof=1)


def consistency(prof: np.ndarray, method: str = "spearman") -> np.ndarray:
    """Per subject: mean correlation of its profile against the other 39.

    Forming a per-subject value is what makes a paired test possible. Comparing
    two pooled all-pairs correlations would give one number per condition and
    no way to test it.
    """
    n = prof.shape[0]
    if method == "spearman":
        r = stats.spearmanr(prof, axis=1).statistic
    else:
        r = np.corrcoef(prof)
    r = np.asarray(r, dtype=float)
    np.fill_diagonal(r, np.nan)
    return np.nanmean(r, axis=1)


def randomization_p(cd: np.ndarray, base: np.ndarray, method: str,
                    n_perm: int = N_PERM, seed: int = SEED) -> float:
    """Two-sided subject-level randomization test on the difference in mean consistency.

    The 40 per-subject consistency values are not independent, since every pair
    of subjects feeds two of them, so the paired Wilcoxon is anticonservative.
    Under the null a subject's two profiles are exchangeable, so swap them
    independently per subject and rebuild both matrices from scratch.
    """
    rng = np.random.default_rng(seed)
    obs = consistency(cd, method).mean() - consistency(base, method).mean()
    n = cd.shape[0]
    hits = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        a = np.where(swap[:, None], base, cd)
        b = np.where(swap[:, None], cd, base)
        stat = consistency(a, method).mean() - consistency(b, method).mean()
        if abs(stat) >= abs(obs) - 1e-15:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# --------------------------------------------------------------------------
def paired_row(label: str, cd: np.ndarray, base: np.ndarray) -> dict:
    d = cd - base
    lo, hi = bca_ci(d)
    w = stats.wilcoxon(cd, base)
    return {"label": label, "mean": d.mean(), "lo": lo, "hi": hi,
            "p": float(w.pvalue), "d": cohens_d_paired(cd, base),
            "pos": int((d > 0).sum()), "neg": int((d < 0).sum())}


def holm(rows: list[dict]) -> list[dict]:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m, run = len(rows), 0.0
    for k, i in enumerate(order):
        adj = min(1.0, rows[i]["p"] * (m - k))
        run = max(run, adj)              # enforce monotonicity
        rows[i]["p_holm"] = run
    return rows


def show(r: dict, scale: float = 1.0, unit: str = "") -> None:
    print(f"  {r['label']:<44} {r['mean']*scale:+8.4f}{unit}  "
          f"95% BCa [{r['lo']*scale:+.4f}, {r['hi']*scale:+.4f}]  "
          f"p = {r['p']:.4g}  Holm p = {r['p_holm']:.4g}  "
          f"d = {r['d']:+.2f}  ({r['pos']}/{r['neg']} +/-)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cd", default="results_cd_resnet_nose_chandrop")
    ap.add_argument("--base", default="results_g3_noaug_instr")
    args = ap.parse_args()

    cd_w, cd_neg = load_profiles(ROOT / args.cd / "instr" / "occlusion.csv")
    bs_w, bs_neg = load_profiles(ROOT / args.base / "instr" / "occlusion.csv")

    if not cd_w.index.equals(bs_w.index):
        sys.exit("the two runs do not cover the same subjects; gate 1 of section 5 fails")
    n = len(cd_w)
    print(f"subjects paired: {n}, channels: {N_CH}")
    print(f"negative drop_pp entries: channel dropout {cd_neg}/{n*N_CH}, "
          f"baseline {bs_neg}/{n*N_CH} (clipped to zero when normalizing)")
    if max(cd_neg, bs_neg) > 0.25 * n * N_CH:
        print("  WARNING: more than a quarter of entries are negative. Occluding a channel "
              "often helps, so the profile is a weak description of reliance. Read cautiously.")

    cd_p, bs_p = normalized(cd_w), normalized(bs_w)

    # ---- the raw quantity, which is the finding; the normalized ones are secondary
    cd_mag = cd_w.to_numpy().sum(axis=1)
    bs_mag = bs_w.to_numpy().sum(axis=1)
    dm = cd_mag - bs_mag
    lom, him = bca_ci(dm)
    print(f"\n--- total single-electrode occlusion cost per subject (THE headline quantity) ---")
    print(f"  baseline {bs_mag.mean():.2f} pp -> channel dropout {cd_mag.mean():.2f} pp "
          f"({bs_mag.mean()/cd_mag.mean():.2f}x reduction)")
    print(f"  paired change {dm.mean():+.2f} pp  95% BCa [{lom:+.2f}, {him:+.2f}]  "
          f"p = {stats.wilcoxon(cd_mag, bs_mag).pvalue:.4g}  d = {dm.mean()/dm.std(ddof=1):+.2f}  "
          f"({int((dm > 0).sum())}/{int((dm < 0).sum())} +/-)")
    print(f"  worst single channel: baseline {bs_w.to_numpy().max(axis=1).mean():.2f} pp -> "
          f"chandrop {cd_w.to_numpy().max(axis=1).mean():.2f} pp")

    print(f"\nmean drop_pp per subject (raw scale, for context)")
    print(f"  channel dropout {cd_w.to_numpy().mean():.3f} pp     "
          f"baseline {bs_w.to_numpy().mean():.3f} pp")

    rows = [
        paired_row("concentration, normalized (lower = flatter)",
                   concentration(cd_p), concentration(bs_p)),
        paired_row("between-subject consistency, Spearman",
                   consistency(cd_p, "spearman"), consistency(bs_p, "spearman")),
    ]
    holm(rows)

    aux = paired_row("between-subject consistency, Pearson",
                     consistency(cd_p, "pearson"), consistency(bs_p, "pearson"))
    aux["p_holm"] = float("nan")
    raw = paired_row("concentration, raw pp (scale-confounded)",
                     concentration(np.clip(cd_w.to_numpy(), 0, None)),
                     concentration(np.clip(bs_w.to_numpy(), 0, None)))
    raw["p_holm"] = float("nan")

    print("\n--- family of two, Holm corrected within ---")
    for r in rows:
        show(r)
    print("--- reported beside the family, not corrected with it ---")
    show(aux)
    show(raw, unit=" pp")

    print("\nrandomization test on consistency (10,000 subject-level swaps, seed 42)")
    p_rand = randomization_p(cd_p, bs_p, "spearman")
    print(f"  Spearman difference in mean consistency: p = {p_rand:.4g}")
    agree = (p_rand < 0.05) == (rows[1]["p_holm"] < 0.05)
    print(f"  agrees with the Holm-adjusted Wilcoxon: {'yes' if agree else 'NO'}"
          f"{'' if agree else ' -- quote the randomization p-value, it is the honest one'}")

    # ---- pre-registered verdict --------------------------------------------
    # VALIDITY GUARD, added 2 September 2026 after the first run.
    # Normalization was meant to remove a modest scale difference between two arms of
    # comparable occlusion magnitude. When one arm's magnitude has collapsed, it instead
    # divides by nearly nothing and returns a profile of measurement noise: apparent
    # concentration rises and apparent consistency falls, both as artifacts. Check first.
    ratio = bs_mag.mean() / cd_mag.mean() if cd_mag.mean() > 0 else float("inf")
    rho_mag = stats.spearmanr(cd_mag, consistency(cd_p, "spearman"))
    normalization_valid = (ratio < 2.0) and not (rho_mag.pvalue < 0.05 and rho_mag.statistic > 0)
    print(f"\n--- validity of the normalized measures ---")
    print(f"  magnitude ratio between arms         : {ratio:.2f}x  (normalization assumes < 2x)")
    print(f"  corr(profile magnitude, consistency) : rho = {rho_mag.statistic:+.3f}, "
          f"p = {rho_mag.pvalue:.4g}  (a positive, significant value means the measure "
          "degrades as the signal shrinks)")
    if not normalization_valid:
        print("  [!] THE NORMALIZED MEASURES ARE NOT INTERPRETABLE. Both are computed on a "
              "profile that has collapsed toward noise. Report the raw magnitude result above "
              "as the finding, and report the transfer question as UNANSWERED, not as answered "
              "negatively. To confirm, shrink the baseline profiles to the augmented arm's "
              "scale, add matched noise, and check whether that alone reproduces the deficit.")

    flat_sig = rows[0]["p_holm"] < 0.05 and rows[0]["mean"] < 0
    cons_sig = rows[1]["p_holm"] < 0.05 and rows[1]["mean"] > 0 and p_rand < 0.05
    reversed_sig = ((rows[0]["p_holm"] < 0.05 and rows[0]["mean"] > 0) or
                    (rows[1]["p_holm"] < 0.05 and rows[1]["mean"] < 0))
    if cons_sig and flat_sig:
        letter, meaning = "B", ("both. Lead the write-up with consistency: flattening is the "
                                "mechanism, consistency is the consequence that matters.")
    elif cons_sig:
        letter, meaning = "C", ("consistency. Section 5.7's argument becomes measured rather "
                                "than asserted.")
    elif flat_sig:
        letter, meaning = "F", ("flattening only. The model stops depending on any single "
                                "electrode, with no evidence the profile transfers. Do not "
                                "extend this to a claim about generalization.")
    elif not normalization_valid:
        letter, meaning = "A", ("artifact. The normalized measures are invalid here (see the "
                                "validity guard above), so neither the concentration nor the "
                                "consistency result may be read. The raw magnitude comparison "
                                "is the finding; the transfer question is UNANSWERED.")
    elif reversed_sig:
        letter, meaning = "X", ("significant REVERSAL. One or both quantities moved against the "
                                "hypothesis with a significant result. This is not outcome N and "
                                "must not be reported as one. Check the validity guard first, "
                                "then report the reversal as the finding it is.")
    else:
        letter, meaning = "N", ("neither. Section 5.7's mechanism paragraph is an "
                                "interpretation the data does not reach and must be softened "
                                "to say so. Quote the intervals; do not report a direction "
                                "as a finding.")
    print(f"\nOUTCOME {letter}: {meaning}")

    out = pd.DataFrame({
        "conc_cd": concentration(cd_p), "conc_base": concentration(bs_p),
        "cons_cd": consistency(cd_p, "spearman"), "cons_base": consistency(bs_p, "spearman"),
    }, index=cd_w.index).round(5)
    out.to_csv(ROOT / "g3_occlusion_per_subject.csv")
    pd.DataFrame(cd_p, index=cd_w.index).round(5).to_csv(ROOT / "g3_profiles_chandrop.csv")
    pd.DataFrame(bs_p, index=bs_w.index).round(5).to_csv(ROOT / "g3_profiles_noaug.csv")
    print("\nwrote g3_occlusion_per_subject.csv, g3_profiles_chandrop.csv, g3_profiles_noaug.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
