# EXPERIMENT_PLAN.md — new experiments from the external-review pass (July 2026)

These experiments answer the actionable items from the five external reviews (the
"where they're right and it's cheap to act" set, plus the CNN/CORAL depth issue).
Every script mirrors the existing experiment conventions (per-subject or train-fold
normalisation, LOSO, `--resume` per-subject checkpointing, subjectwise + summary
CSVs). Run from `06_Code/` with the project venv (`.venv`), Python 3.10,
GPU used automatically where relevant (RTX 4050 confirmed).

Shared inputs:
- `FEAT = features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz`
- `META = features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv`
- `NPZ  = windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz`  (raw envelopes for the CNN)

Status legend:  [DONE] already run this pass · [RUN] to be run this pass.

---

## 1. LDA carried through LOSO — [RUN]  (cheap, ~5–10 min each)
**Why:** Critiques 1 & 5 — LDA was only ever an SD baseline; carry the "classical minimal"
baseline through LOSO to complete the classical-vs-deep comparison and retire the
"LDA not evaluated under LOSO" limitation.
```
python run_lda_loso.py --features $FEAT --meta $META --norm-mode per_subject --out results_lda_persubj --resume
python run_lda_loso.py --features $FEAT --meta $META --norm-mode global      --out results_lda_global  --resume
```
**Outputs:** `results_lda_{persubj,global}/lda_subjectwise.csv`, `lda_summary.csv`.
**Compare to:** SVM 0.777, RF 0.773, CNN 0.754 (per-subject); global SVM 0.708, RF 0.722.
**Wire into thesis:** add an LDA row to the LOSO model table (§4.2) and the normalisation
ablation (Table 4.7); replace the Discussion limitation "LDA was not evaluated under LOSO".
Smoke-checked: Sub01 0.628, Sub02 0.806, Sub03 0.670 (runs correctly).

## 2. Honest per-window inference latency — [DONE]
**Why:** Critique 2 — the CNN's 0.0116 ms/window was implausible (batched figure). Re-measured
single-window (batch=1), single-threaded (embedded-realistic), warm-up + timed reps.
```
python measure_inference_latency.py --features $FEAT --meta $META --npz $NPZ --reps 1500 --warmup 150
```
**Result** (`results_latency/inference_latency_measured.csv`), median ms/window:
- SVM **0.97 ms** (thesis said 1.06 — consistent)
- CNN **0.67 ms** CPU / 0.98 ms CUDA (thesis said 0.0116 ms — the figure under review; ~60× too low)
- RF **25.4 ms** single-threaded (thesis said 0.57 ms — that was **batched/amortised**; single-window is far slower; with `n_jobs=-1` a 1-sample predict balloons to ~71 ms from joblib dispatch, itself an artifact)
- All three p95 < 125 ms → real-time claim **holds**, but RF headroom is much smaller than stated.
**Wire into thesis:** correct Discussion §5.12 (Limitation 8) latency numbers; state the
single-window-vs-batched distinction explicitly; note RF is the slowest per single window.

## 3. STDUP class-balance sub-sampling control — [RUN]  (~30–40 min)
**Why:** Critiques 1 & 2 — is STDUP's high F1 biomechanical or a sample-size effect macro-F1 masks?
Downsamples STDUP **training** windows to the mean of the other classes (test untouched).
```
python run_stdup_subsample.py --features $FEAT --meta $META --models SVM,RF --conditions imbalanced,balanced --out results_stdup_subsample --resume
```
**Outputs:** `results_stdup_subsample/stdup_subsample_{subjectwise,summary}.csv` (per-class F1 × condition × model).
**Read:** if `f1_STDUP` holds under `balanced`, the advantage is biomechanical, not sample size.
Smoke-checked Sub01: STDUP 0.862→0.860 (SVM) despite cutting STDUP train 14348→3770 — early
support for the biomechanical reading.
**Wire into thesis:** new short Results paragraph (class-imbalance control) + firm up / appropriately
soften §5.3 hierarchy claim based on the full-run result.

## 4. Fairer deep baseline — compact 1D ResNet + SE attention — [RUN]  (GPU, ~1–2 h per arch)
**Why:** All five reviews — SimpleEMGCNN is a 2015-era design, so "classical beats deep under LOSO"
is confounded by architecture. `EMGResNet1D` (residual + squeeze-excite channel attention, ~0.56M
params) is a modern-but-compact fairer baseline. Identical input, per-subject norm, LOSO.
```
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch simple    --epochs 40 --out results_cnn_loso_simple_repro --resume   # sanity: should reproduce ~0.754
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_se --epochs 40 --out results_cnn_loso_resnet_se --resume     # fairer deep baseline
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet    --epochs 40 --out results_cnn_loso_resnet    --resume     # ablation: SE on/off
```
**Outputs:** `results_cnn_loso_*/cnn_arch_{subjectwise,summary}.csv`.
**Interpretation (either outcome strengthens the thesis):** if resnet_se still trails SVM/RF, the
"unadapted end-to-end model is exposed to amplitude shift" argument is now robust to architecture,
not an artifact; if it closes the gap, report it honestly and revise RQ1. Smoke-checked: trains
end-to-end (557,276 params).
**Wire into thesis:** §4.2.2 + the classical-vs-deep framing (RQ1) + retire the "shallow CNN" confound
in the CNN limitation.

## 5. Deep CORAL for the CNN — CORAL is no longer classical-only — [RUN]  (GPU, ~1–2 h)
**Why:** Critiques 1, 2, 5 — CORAL was applied to SVM/RF only. Deep CORAL (Sun & Saenko, 2016) adds a
covariance-alignment loss between labelled source and **unlabelled** target features during CNN
training (train-fold global norm base, so CORAL is the only adaptation — mirroring the classical setup).
```
python run_deep_coral_cnn_loso.py --npz $NPZ --meta $META --arch resnet_se --coral-lambda 1.0 --epochs 40 --out results_deep_coral_cnn_resnet_se --resume
# optional sensitivity: --coral-lambda 0.3 and 3.0
```
**Outputs:** `results_deep_coral_cnn_resnet_se/deep_coral_{subjectwise,summary}.csv`.
**Compare to:** CNN global 0.682 · CNN per-subject z-score 0.754 · classical CORAL SVM 0.724 / RF 0.747.
The key question: does a *learned* UDA on the CNN beat the *simple* per-subject normalisation? Smoke-checked: runs end-to-end with the CORAL loss.
**Wire into thesis:** extend §4.13 (CORAL) to the CNN; update the Discussion limitation that currently
says "CORAL was applied only to the SVM and RF".

## 6. Adaptive BatchNorm (AdaBN) for the CNN — the cheapest, most apt modern UDA — [RUN]  (GPU, ~1–2 h)
**Why:** AdaBN (Li et al., 2016/2018) is the deep-network analogue of per-subject normalisation, and the
domain-generalisation literature (DomainBed, EMGBench) shows this class of simple statistical-alignment
methods is hard to beat. Train the CNN on source (train-fold global norm); at test time REPLACE every
BatchNorm layer's running statistics with the held-out subject's own unlabelled statistics, then classify.
Parameter-free, label-free, transductive — the most direct deep counterpart to the thesis's winning
classical move, and cheaper than Deep CORAL.
```
python run_adabn_cnn_loso.py --npz $NPZ --meta $META --arch resnet_se --epochs 40 --out results_adabn_cnn_resnet_se --resume
# optional: --arch simple  (AdaBN on the original SimpleEMGCNN, like-for-like with the 0.754 / 0.682 numbers)
```
**Outputs:** `results_adabn_cnn_resnet_se/adabn_{subjectwise,summary}.csv` (records pre- and post-AdaBN F1 per
subject, so the adaptation effect is explicit).
**Compare to:** CNN global 0.682 · CNN per-subject z-score 0.754 · Deep CORAL (§5) · classical CORAL SVM 0.724 / RF 0.747.
Smoke-checked Sub01 (2-epoch): 0.512 → 0.564 (+5.3 pp) after swapping BN stats to the target — AdaBN adapts
the CNN with zero weights touched.
**Wire into thesis:** fold into the CNN-side adaptation comparison in §4.13 alongside Deep CORAL; it is the
more-modern, more-appropriate UDA the reviewers implied, and it connects to the causal-normalisation analysis
(§4.14), since a causal/streaming AdaBN is its deployable version.

---

## After the runs — wiring the numbers in
1. Add a small `compare_*` / stats step (mirror `stats_new_experiments.py`): paired Wilcoxon +
   Cohen's d + BCa CIs for LDA-vs-baselines, resnet_se-vs-SimpleEMGCNN, Deep-CORAL-vs-per-subject,
   and fold the new paired tests into the whole-thesis BH-FDR family (`stats_unified_fdr.py`).
2. Update REPRODUCE.md with these commands (a stub is already appended for the earlier airtight pass).
3. Update the thesis chapters at the section anchors noted under each experiment above.
Keep every new claim paired-tested, effect-sized, and multiple-comparison-corrected, exactly as the
existing §4.16 family is.
