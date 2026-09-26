#!/usr/bin/env python3
"""
kc23_run_loader.py
====================
One fail-closed loader for the per-fold outputs of an instrumented
`run_cnn_arch_loso.py` run, shared by the KC-D2, KC-D3 and KC-D4 aggregators
(26 September 2026). Every check either passes or raises; nothing is defaulted.

A run directory holds cnn_arch_subjectwise.csv, run_config.json and, when it was
instrumented, instr/{occlusion,attenuation,permutation,embed_probes}.csv.

Per-subject summaries (KC-D1's C17 definition, "per-subject summed drop", no
clipping of negative drops):
  occlusion_sum    sum over the channels of drop_pp (zero one channel)
  attenuation_sum  sum over the channels of drop_pp at alpha = 0.5
  permutation_sum  sum over the channels of the mean F1 drop over R = 5
                   permutations, in percentage points
A negative drop (the perturbed model scored higher) is kept as it is.
The reduction factor of an arm against the reference is
mean over subjects of the reference sum / mean over subjects of the arm sum.

The run's configuration is verified against what the caller expects
(augmentation, seed, gain SD, chandrop rate, architecture, normalization), so a
directory that holds the wrong arm cannot be aggregated under the right name.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_SUBJECTS_SIAT = 40
ATTENUATION_ALPHA = 0.5


def read_config(run_dir: Path) -> dict:
    p = run_dir / "run_config.json"
    if not p.exists():
        raise FileNotFoundError(f"missing input: {p}")
    return json.loads(p.read_text(encoding="utf-8")).get("args", {})


def check_config(run_dir: Path, *, augmentation=None, seed=None, gain_sd=None, chandrop_p=None,
                 arch="resnet_se", norm="per_subject", window_ms_token=None) -> dict:
    a = read_config(run_dir)
    want = {"augmentation": augmentation, "seed": seed, "arch": arch, "norm_mode": norm}
    for k, v in want.items():
        if v is not None and a.get(k) != v:
            raise ValueError(f"{run_dir.name}: run_config {k}={a.get(k)!r}, expected {v!r} (mislabelled or wrong arm)")
    if gain_sd is not None and abs(float(a.get("aug_gain_sd", -1)) - gain_sd) > 1e-9:
        raise ValueError(f"{run_dir.name}: aug_gain_sd={a.get('aug_gain_sd')!r}, expected {gain_sd}")
    if chandrop_p is not None and abs(float(a.get("aug_chandrop_p", -1)) - chandrop_p) > 1e-9:
        raise ValueError(f"{run_dir.name}: aug_chandrop_p={a.get('aug_chandrop_p')!r}, expected {chandrop_p}")
    if window_ms_token is not None and window_ms_token not in str(a.get("npz", "")):
        raise ValueError(f"{run_dir.name}: npz {a.get('npz')!r} does not contain {window_ms_token!r}")
    return a


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    return pd.read_csv(path)


def load_f1(run_dir: Path, n_subjects: int = N_SUBJECTS_SIAT) -> pd.Series:
    d = _read(run_dir / "cnn_arch_subjectwise.csv")
    if not {"subject", "f1_macro"} <= set(d.columns):
        raise ValueError(f"{run_dir.name}: cnn_arch_subjectwise.csv lacks subject/f1_macro")
    if d["subject"].duplicated().any() or len(d) != n_subjects or d["f1_macro"].isna().any():
        raise ValueError(f"{run_dir.name}: needs {n_subjects} unique non-NaN subjects, found {len(d)} rows")
    return d.set_index("subject")["f1_macro"].astype(float).sort_index()


def _sum_over_channels(df: pd.DataFrame, value: str, subjects: set, label: str, scale: float = 1.0) -> pd.Series:
    if df[value].isna().any():
        raise ValueError(f"{label}: NaN in {value}")
    s = df.groupby("subject")[value].sum() * scale
    n_ch = df.groupby("subject").size()
    if set(s.index) != subjects or n_ch.nunique() != 1:
        raise ValueError(f"{label}: subjects differ from the F1 subjects, or the channel count varies across subjects")
    return s.sort_index()


def load_instr(run_dir: Path, subjects: set, probes: bool = True) -> dict:
    ins = run_dir / "instr"
    occ = _read(ins / "occlusion.csv")
    att = _read(ins / "attenuation.csv")
    perm = _read(ins / "permutation.csv")
    label = run_dir.name
    att05 = att[np.isclose(att["alpha"], ATTENUATION_ALPHA)]
    out = {
        "occlusion_sum": _sum_over_channels(occ, "drop_pp", subjects, f"{label} occlusion"),
        "attenuation_sum": _sum_over_channels(att05, "drop_pp", subjects, f"{label} attenuation@0.5"),
        "permutation_sum": _sum_over_channels(perm, "f1_drop_mean", subjects, f"{label} permutation", scale=100.0),
    }
    if probes:
        ep = _read(ins / "embed_probes.csv")
        need = {"subject", "subject_probe_bacc", "held_out_class_silhouette", "held_out_class_probe_bacc"}
        if not need <= set(ep.columns):
            raise ValueError(f"{label}: embed_probes.csv lacks {sorted(need - set(ep.columns))}")
        if ep["subject"].duplicated().any() or set(ep["subject"]) != subjects:
            raise ValueError(f"{label}: embed_probes.csv subjects differ from the F1 subjects, or are duplicated")
        if ep[["subject_probe_bacc", "held_out_class_silhouette", "held_out_class_probe_bacc"]].isna().any().any():
            raise ValueError(f"{label}: NaN in an embedding probe")
        ep = ep.set_index("subject").sort_index()
        out["subject_probe_bacc"] = ep["subject_probe_bacc"].astype(float)
        out["class_silhouette"] = ep["held_out_class_silhouette"].astype(float)
        out["class_probe_bacc"] = ep["held_out_class_probe_bacc"].astype(float)
    return out


def load_run(run_dir: Path, *, instrumented: bool = True, probes: bool = True, n_subjects: int = N_SUBJECTS_SIAT,
             **config_expect) -> dict:
    """Validate the configuration, then return {'f1': Series, and, if instrumented, the per-subject summaries}."""
    run_dir = Path(run_dir)
    check_config(run_dir, **config_expect)
    f1 = load_f1(run_dir, n_subjects)
    out = {"f1": f1}
    if instrumented:
        out.update(load_instr(run_dir, set(f1.index), probes=probes))
    return out


def reduction_factor(ref_sum: pd.Series, arm_sum: pd.Series) -> float:
    """mean(reference summed drop) / mean(arm summed drop); +inf if the arm's mean is not positive."""
    den = float(arm_sum.mean())
    return float(ref_sum.mean() / den) if den > 0 else float("inf")
