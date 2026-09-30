"""Synthetic instrumented-run fixtures, in the REAL on-disk format of src/run_cnn_arch_loso.py, shared by the aggregator
tests (D2, D3, D4, D1 contrasts, C6). Real-format means: the same file names, column names and channel/alpha grids
that the instrumentation writes (checked against results/kc23_d1_r1_s42/ on 26 September 2026)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_CH = 9
ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)


def _val(x, subject, default=0.0):
    if x is None:
        return default
    return x(subject) if callable(x) else x


def make_run(root: Path, name: str, *, augmentation="none", seed=42, gain_sd=0.4, chandrop_p=0.2, arch="resnet_se",
             norm="per_subject", f1=0.8, occ=5.0, att=1.0, perm=0.1, probe=0.6, sil=0.2, cprobe=0.9,
             instrumented=True, n=40, subjects=None, npz="windows_x_w250_y.npz"):
    """One run directory. f1, occ, att, perm, probe, sil, cprobe may be scalars or callables of the subject id.
    occ/att are drop_pp PER CHANNEL, perm is the F1 drop (fraction) PER CHANNEL."""
    d = Path(root) / name
    (d / "instr").mkdir(parents=True, exist_ok=True)
    subjects = list(subjects) if subjects is not None else list(range(1, n + 1))
    (d / "run_config.json").write_text(json.dumps({"args": {
        "augmentation": augmentation, "seed": seed, "aug_gain_sd": gain_sd, "aug_chandrop_p": chandrop_p,
        "arch": arch, "norm_mode": norm, "npz": npz, "instrument": f"{name}/instr"}}), encoding="utf-8")
    pd.DataFrame({"subject": subjects, "arch": arch, "f1_macro": [_val(f1, s) for s in subjects],
                  "bal_acc": [_val(f1, s) for s in subjects]}).to_csv(d / "cnn_arch_subjectwise.csv", index=False)
    if not instrumented:
        return d
    pd.DataFrame([{"subject": s, "channel": c, "f1_full": _val(f1, s), "f1_occluded": 0.5,
                   "drop_pp": _val(occ, s)} for s in subjects for c in range(N_CH)]).to_csv(
        d / "instr" / "occlusion.csv", index=False)
    pd.DataFrame([{"subject": s, "channel": c, "alpha": a, "f1": 0.5,
                   "drop_pp": 0.0 if a == 1.0 else _val(att, s) * (1.0 - a)}
                  for s in subjects for c in range(N_CH) for a in ALPHAS]).to_csv(
        d / "instr" / "attenuation.csv", index=False)
    pd.DataFrame([{"subject": s, "channel": c, "r": 5, "f1_drop_mean": _val(perm, s), "f1_drop_sd": 0.01}
                  for s in subjects for c in range(N_CH)]).to_csv(d / "instr" / "permutation.csv", index=False)
    pd.DataFrame([{"subject": s, "n_val_subjects": 6, "n_probe_subjects": 7, "chance_subject_probe": 1 / 7,
                   "subject_probe_bacc": _val(probe, s), "held_out_class_silhouette": _val(sil, s),
                   "held_out_class_probe_bacc": _val(cprobe, s), "probe_cap": 100} for s in subjects]).to_csv(
        d / "instr" / "embed_probes.csv", index=False)
    return d
