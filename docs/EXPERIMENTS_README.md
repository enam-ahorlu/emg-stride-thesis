# Experiments — Status & Exact Commands Used

**All experiments below are DONE (completed 2026-07-11).** This file now records what was
actually run, with the real commands and result locations, so everything is reproducible.
For the full results tables and interpretation, see **`JULY_2026_EXPERIMENTS_SUMMARY.md`**.

Run these in the project's own `.venv` (Python 3.14 — VS Code auto-activates it when you open
this folder; it has torch+CUDA, sklearn, psutil) from the `06_Code/` folder.

Shared paths:
```
FEAT=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
```

---

## 1.1 — Causal / streaming normalisation (`run_streaming_norm_loso.py`) — DONE
```
python run_streaming_norm_loso.py --features $FEAT --meta $META \
    --configs transductive,calib25,calib50,calib100,running --models SVM,RF \
    --n-jobs 1 --rf-n-jobs 6 --resume
```
Results: `results_loso_freq_streaming/streaming_norm_summary_FULL.csv`. `transductive`
reproduced the headline (0.7767/0.7731). Model-dependent story on the causal variants —
see the summary doc for the full table.

## 2.2 — CORAL UDA baseline (`run_coral_loso.py`) — DONE
```
python run_coral_loso.py --features $FEAT --meta $META --models SVM,RF \
    --n-jobs 1 --rf-n-jobs 6 --resume
```
Results: `results_loso_freq_coral/`. Per-subject z-score beats CORAL for both models.

## 2.3 — Seed stability (`run_seed_stability.py`) — DONE
```
python run_seed_stability.py --features $FEAT --meta $META --seeds 42,7,123 --models classical --resume
python run_seed_stability.py --features $FEAT --meta $META --npz $NPZ --seeds 42,7,123 --models classical,cnn --resume
```
Results: `results_seed_stability/seed_stability_summary.csv`. All 3 models seed-stable
(SVM SD=0.0000, RF SD=0.0003, CNN SD=0.0038).

## 2.6 — CNN calibration fine-tune (`run_cnn_calibration_loso.py`) — DONE + investigated
```
python run_cnn_calibration_loso.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 15 --resume
```
Results: `results_cnn_calibration/`. **Surprising finding**: K=5 calibration hurt F1 by
−11.4pp vs. no calibration. This was audited and investigated with two follow-up runs
(see below and the summary doc for the full story):
```
python run_cnn_calibration_loso.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 3 --out results_cnn_calibration_ftepochs3 --resume
python run_cnn_calibration_loso.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 15 --seed 7 --out results_cnn_calibration_seed7 --resume
python analyze_cnn_calibration_followup.py   # per-subject audit, outputs -> report_figs/cnn_calibration_followup/
```
Root cause found: the fine-tune step had no early stopping (unlike base CNN training).
Fewer epochs (3 vs 15) eliminated the degradation and gave a *better* result overall
(+3.2pp at K=20 vs. the original protocol's +1.7pp).

---

## 2.1 — External validation (ENABL3S) — DONE
See `../EXTERNAL_VALIDATION_GUIDE.md` for the original design rationale (Design B, replication).
Actual steps run:
```
python adapt_external_dataset.py --root 5362627 --resume
python extract_features.py --npz features_out_ext/windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60.npz \
    --meta features_out_ext/windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_meta.csv \
    --out-dir features_out_ext --prefix freq --use raw --freq --fs 1000 --no-wavelet

FEXT=features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_ext.npz
MEXT=features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_meta.csv
NEXT=features_out_ext/windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60.npz
MEXT_RAW=features_out_ext/windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_meta.csv

python train_classical_loso.py --features $FEXT --meta $MEXT --out results_ext_persubj --models SVM,RF --norm-mode per_subject --n-jobs 1 --rf-n-jobs 6 --resume
python train_classical_loso.py --features $FEXT --meta $MEXT --out results_ext_global  --models SVM,RF --norm-mode global      --n-jobs 1 --rf-n-jobs 6 --resume
python train_cnn_loso.py       --npz $NEXT --meta $MEXT_RAW --out results_ext_cnn_persubj --norm-mode per_subject --resume
python train_cnn_loso.py       --npz $NEXT --meta $MEXT_RAW --out results_ext_cnn_global  --norm-mode global      --resume

# Subject-dependent baseline (run last, added after the LOSO comparison was already confirmed)
python train_classical_patched.py --features $FEXT --meta $MEXT --subjects all --splits 5 \
    --models SVM,RF --svm-scale --rf-n-jobs 1 --out results_ext_sd --save-preds --resume
```
STDUP was **not** sparse (7,312/45,525 windows) — no 3-class fallback needed.
**Both headline findings replicated, more strongly than on SIAT**: per-subject beats global
for all 3 models (SVM +10.3pp, RF +11.1pp, CNN +16.9pp), and SD≫LOSO generalization gap
(SVM 18.6pp, RF 17.6pp) — roughly double SIAT's gap. Full table in the summary doc.

---

## Crash-safe long runs (memory) — what actually worked

Real lessons learned running these on a 16GB-RAM machine with fluctuating background load:

1. **`GridSearchCV`'s own `n_jobs` must stay at 1.** Raising it above 1 deadlocks under
   repeated calls (once per LOSO fold) in a detached/background process on Windows — loky's
   reusable executor hangs on the 2nd+ `.fit()`. This bit every script that originally used
   `n_jobs=-1` here (`train_classical_loso.py`, `run_streaming_norm_loso.py`,
   `run_coral_loso.py`, `train_classical_patched.py`) — all fixed to hardcode/default this to 1.
2. **`RandomForestClassifier`'s own `n_jobs` (its internal tree-building parallelism) is a
   different, safe code path** — verified safe under repeated calls. Use `--rf-n-jobs 3–6` for
   real speed; using `-1` (all cores) risks memory spikes on a tight machine, so it's
   explicitly bounded rather than left unlimited.
3. **`gc.collect()` between CV folds is real hygiene, not optional**, for any RF grid that
   includes `max_depth=None` at high `n_estimators` — Windows doesn't always return freed
   numpy/sklearn memory to the OS promptly, so without this, memory creeps up fold-over-fold.
4. **Per-fold checkpointing beats per-subject/per-model checkpointing** when a single fold's
   own peak memory is close to the machine's ceiling — a memory-driven kill mid-fold should
   only cost the in-flight fold, never redo already-completed ones. `train_classical_loso.py`,
   `train_cnn_loso.py`, and (after a live fix) `train_classical_patched.py` all checkpoint at
   this granularity.
5. **`run_with_memory_guard.py`** (single long-running job) and **`run_multi_guard.py`**
   (several independent jobs run concurrently, e.g. SVM+RF together, scaling concurrency to
   free RAM) were both used throughout. Needs `pip install psutil` (already in the project venv).
   Example:
   ```
   python run_with_memory_guard.py --max-mem-percent 94 --min-free-gb 1.0 -- \
       python run_streaming_norm_loso.py --features $FEAT --meta $META \
       --configs transductive,calib25,calib50,calib100,running --models SVM,RF --resume
   ```
   (Put everything after the literal `--`. The guard adds `--resume` for you if you forget it
   — so any script it wraps must accept that flag, even as a no-op.)
6. **Root cause of most kills was NOT a bug in these scripts** — it's a genuine resource race
   between a fluctuating ~9–13GB baseline (Windows + the editor + whatever background
   apps are open) and each job's own peak memory need, both landing in the same 4–8GB band on
   a 16.9GB machine. The checkpointing above guarantees eventual completion regardless of how
   many times that race is lost; it doesn't make individual runs fast.

### Notes
- `train_classical_loso.py` accepts `--seed` (RF + RFE) and `--rf-n-jobs`.
- All scripts reuse the exact model definitions from `train_classical_loso.py` /
  `train_cnn_loso.py`, so comparisons are apples-to-apples.
- Full results, interpretation, and the complete list of fixes made along the way:
  **`JULY_2026_EXPERIMENTS_SUMMARY.md`**.
