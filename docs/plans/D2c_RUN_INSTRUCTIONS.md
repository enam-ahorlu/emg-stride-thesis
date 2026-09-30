# Run instructions for D2c (Deep CORAL alignment logging)

The instructions handed to whoever runs this are everything below the line.

---

Run D2c for the MSc thesis.

**Read first:** `06_Code/docs/EXPERIMENT_PLAN_DEEPCORAL.md`, section **"D2c AMENDMENT, 19 September 2026"** at the end of the
file. Read D2 and D2b above it for context only.

**Working directory:** `06_Code/`, in the project `.venv`. Use the absolute path to the inner python.

**Inputs:**
```
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
SCRIPT=run_deep_coral_align_loso.py      (new; imports from run_deep_coral_cnn_loso.py, train_cnn_loso.py, cnn_architectures.py)
ANALYSIS=d2c_analysis.py                 (new)
```

**Order of operations. Do not reorder.**

1. Write `PREDICTION.md`, timestamped, into `results_deep_coral_align_lam0p1/` before any run starts, copying the prediction and
   the outcome table from the amendment.
2. Smoke test: `--heldout 1 --out _d2c_smoke` at lambda 0.1. Confirm the three CSVs appear (`deep_coral_subjectwise.csv`,
   `alignment_subjectwise.csv`, `training_log.csv`) and note the fold time. Delete nothing; the smoke directory is ignored.
3. Stage A: lambda 0.1 into `results_deep_coral_align_lam0p1`, lambda 100 into `results_deep_coral_align_lam100`, each with
   `--augmentation chandrop --epochs 40 --resume`, each its own job.
4. `python d2c_analysis.py`. It checks the reproduction gates first and stops on a miss.
5. Only on Outcome 1: lambda 0 into `results_deep_coral_align_lam0`, then `python d2c_analysis.py --with-lam0`.
6. Family v9 as the amendment states.

**Hard stops:** a failed gate; Outcome 4 or 5. Report and run nothing further. Outcomes 2 and 3 are reported for Enam's
decision on framing; do not edit any thesis file.

**Framing note.** Whatever this returns, it decides one sentence in the thesis's discussion of the through-line and one row of the
limitations. It does not touch Finding A: per-subject normalization's standing against Deep CORAL rests on D2b, not on this.
