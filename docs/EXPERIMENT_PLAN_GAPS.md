# EXPERIMENT_PLAN_GAPS.md — close the G1–G7 gaps (headline models under-validated)

**Why.** The robustness/deployability leg was built around the *original* models (SVM, RF, SimpleEMGCNN) and never re-run on the models that became the headline: **ResNet-SE** (best single, 0.782) and the **soft-vote ensemble** (best overall, 0.815). This plan extends every under-covered analysis to those models so the strongest claim is not the least-validated one. Same conventions as the other plans: run from `06_Code/` with the venv, `--resume`, subjectwise + summary CSVs. [RUN] = Claude Code.

**Shared SIAT inputs:** `FEAT`, `META`, `NPZ` as in EXPERIMENT_PLAN.md.
**ENABL3S inputs:** the feature/window/meta files produced by `adapt_external_dataset.py` (root `5362627/`); confirm their exact paths from that script (the same ones the existing `results_ext_cnn_*` used). ENABL3S = 10 subjects, 7 channels (56-D features).

---

## G5 — Error / per-class / subject-difficulty on the headline models — ✅ DONE (no run needed)
Computed from the already-saved ensemble probabilities. Results in `results_ensemble_v2/headline_error_analysis/`:
- **ResNet-SE**: per-class F1 DNS 0.666 / STDUP 0.954 / UPS 0.744 / WAK 0.701; DNS recall 66.9%, DNS→WAK 11.2%; subject-F1 range 0.55–0.90.
- **Soft ensemble (SVM+RF+ResNet-SE)**: DNS **0.726** / STDUP 0.967 / UPS 0.796 / WAK 0.761; DNS recall **71.2%**, **DNS→WAK 8.6%** (down from ~12.5% single-model); subject range 0.62–0.95.
- **Key finding to wire in:** the ensemble improves the hardest, safety-critical class (DNS) the most and cuts the fall-risk DNS→WAK confusion by ~4 pp. Use in the re-pivoted §4.3 (anatomy of the gap / safety) and §4.5 (ensembling).

## G1 — External validation (ENABL3S) of the headline models — [RUN, GPU]
Replicate the SIAT deep-tier findings on the independent dataset. For each, report per-subject vs global (and vs each other) exactly as the existing ENABL3S SVM/RF/CNN comparison does; lead with effect size (n = 10).
```
# ResNet-SE on ENABL3S (per-subject and global)
python run_cnn_arch_loso.py --npz <ENABL3S_NPZ> --meta <ENABL3S_META> --arch resnet_se --epochs 40 --norm-mode per_subject --out results_ext_resnet_se_persubj --resume
python run_cnn_arch_loso.py --npz <ENABL3S_NPZ> --meta <ENABL3S_META> --arch resnet_se --epochs 40 --norm-mode global      --out results_ext_resnet_se_global --resume
# Deep CORAL + AdaBN on ENABL3S (resnet_se backbone, global-norm base)
python run_deep_coral_cnn_loso.py --npz <ENABL3S_NPZ> --meta <ENABL3S_META> --arch resnet_se --coral-lambda 1.0 --epochs 40 --out results_ext_deepcoral --resume
python run_adabn_cnn_loso.py       --npz <ENABL3S_NPZ> --meta <ENABL3S_META> --arch resnet_se --epochs 40 --out results_ext_adabn --resume
# Soft ensemble on ENABL3S: regenerate SVM/RF/ResNet-SE probabilities on ENABL3S (per-subject norm), then combine
#   (reuse the patched trainers from EXPERIMENT_PLAN_ENSEMBLE.md, output to results_ext_ensemble_v2/proba/), then:
python ensemble_v2_combine.py --proba-dir results_ext_ensemble_v2/proba --subjects 10 --out results_ext_ensemble_v2
```
**Wire into:** §4.7 (robustness) external-validation subsection + the cross-dataset synthesis. **Question answered:** do "ResNet-SE ≈ classical", "per-subject norm beats Deep CORAL/AdaBN", and "soft ensemble is best" all replicate on ENABL3S? Report honestly either way.

## G2 — Inference latency of the headline models — [RUN, fast]
Extend `measure_inference_latency.py` to add **ResNet-SE** (batch=1, single-thread, CPU + CUDA) and the **soft ensemble** (sum of SVM + RF + ResNet-SE single-window inference + the probability-average op). Append rows to `results_latency/inference_latency_measured.csv`.
**Wire into:** §4.7 latency subsection; confirm the ensemble p95 stays < 125 ms (the ensemble pays the sum of member costs, so this is the real deployability check for the headline).

## G3 — Causal / streaming AdaBN on ResNet-SE — [RUN, GPU]
New `run_adabn_causal_loso.py`: as AdaBN, but estimate each held-out subject's BatchNorm statistics from a **causal calibration buffer** (first K = 25/50/100 windows) and from a **running estimator**, instead of the whole session. This is the deployable deep analogue of the classical causal-normalization study (§4.14), which is currently flagged "untested". Output `results_adabn_causal/`. Compare to: global-norm CNN, full (non-causal) AdaBN, per-subject norm.
**Wire into:** §4.7 deployability — completes the causal story on the deep side.

## G4 — Supervised calibration on ResNet-SE — [RUN, GPU]
Re-run the regularized 3-epoch calibration (and the 5-draw robustness check) on **ResNet-SE** instead of SimpleEMGCNN:
```
python run_cnn_calibration_loso.py --npz $NPZ --meta $META --arch resnet_se --ft-epochs 3 --out results_cnn_calibration_resnet_se --resume
python run_cnn_calibration_multidraw.py --npz $NPZ --meta $META --arch resnet_se --draws 5 --out results_cnn_calibration_resnet_se_multidraw --resume
```
**Wire into:** §4.7 calibration subsection — the calibration PoC should be on the architecture of record. Compare the K=0/5/10/20 lift to the SimpleEMGCNN version.

## G6 — STDUP class-balance control on CNN + ResNet-SE — [RUN, GPU]
Extend `run_stdup_subsample.py` (currently SVM/RF) to the CNN and ResNet-SE, same balanced-vs-imbalanced protocol. Output into `results_stdup_subsample/`.
**Wire into:** §4.3 class-imbalance control — makes the "STDUP advantage is biomechanical" claim hold for the deep models too, not just the classical ones.

## G7 — Generalization-gap table update — [RUN, GPU + analysis]
Run ResNet-SE under **subject-dependent** 5-fold CV (`run_cnn_arch_loso` in SD mode, or `train_cnn_subjectdep` with the resnet_se arch) → `results_sd_resnet_se`. Then recompute the generalization-gap table (Table 4.5) to (a) add a ResNet-SE row (SD − LOSO gap) and (b) reflect the 0.815 optimized endpoint. 
**Wire into:** §4.2 generalization gap; the "gap reduction" numbers should reference the current best, not 0.792.

---

## After the runs
For every new comparison compute paired Wilcoxon + Cohen's d + BCa CIs and fold into `stats_unified_fdr.py` (the family grows again). Then the numbers slot into the re-pivoted Results at the anchors above. Keep every claim paired-tested and corrected, exactly as the existing families are.
