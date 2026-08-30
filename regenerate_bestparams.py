#!/usr/bin/env python3
"""
regenerate_bestparams.py
=========================
EXPERIMENT_PLAN_CRITIQUE.md, E3 Step 0. `_bestparams.json` (consumed by
rescore_streaming_buffer_v2.py and run_causal_ensemble.py) was deleted.
Rebuild it from the `best_params` column of the authoritative per-subject-norm
LOSO run (results_loso_freq_persubj/*_subjectwise.csv) into
{model: {subject_int: {clf__param: value}}}.

Sanity check: SVM must be C=1 for all 40 subjects (matches the headline run).
"""
import ast
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
SRC = ROOT / "results_loso_freq_persubj"
STEM = "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext"

out = {}
for model in ["SVM", "RF"]:
    csv_path = SRC / f"{STEM}__{model}_nested_loso_subjectwise.csv"
    df = pd.read_csv(csv_path)
    d = {}
    for _, r in df.iterrows():
        params = ast.literal_eval(r["best_params"])
        d[int(r["heldout_subject"])] = params
    out[model] = d
    print(f"[{model}] {len(d)} subjects loaded from {csv_path.name}")

# sanity check
svm_c = {s: p.get("clf__C") for s, p in out["SVM"].items()}
bad = {s: c for s, c in svm_c.items() if c != 1}
assert len(out["SVM"]) == 40, f"expected 40 SVM subjects, got {len(out['SVM'])}"
if bad:
    raise AssertionError(f"SVM C != 1 for subjects: {bad}")
print("[sanity] SVM C=1 for all 40 subjects: OK")

with open(ROOT / "_bestparams.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"[save] {ROOT / '_bestparams.json'}")
