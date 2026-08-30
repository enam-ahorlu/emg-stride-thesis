# EXPERIMENT_PLAN_ENSEMBLE.md — better ensemble (soft/weighted/stacking) + fold in ResNet-SE

**Why.** The thesis ensemble is **hard majority voting** of SVM+RF+SimpleEMGCNN = **0.7917**. Two questions:
1. *Was hard voting the best combiner?* Hard voting throws away confidence and breaks 3-way ties arbitrarily. Soft voting (average probabilities), weighted-soft, and stacking (a meta-learner) usually do better. An **oracle upper bound** computed from the saved hard predictions — "correct if ANY of the three models is correct" — is **0.889**, i.e. there is ~10 pp of complementarity a smarter combiner could tap. So a better method is plausibly worth it.
2. *Does folding in ResNet-SE help?* ResNet-SE alone (0.782) already ≈ the ensemble (0.792), so an ensemble that includes it is the obvious candidate to beat 0.792.

**Blocker.** Only **hard** predictions were saved (no probabilities), and ResNet-SE saved **no per-window predictions**. So soft/weighted/stacking and any ResNet-SE ensemble require regenerating per-window **class probabilities**. This plan does that, then runs the combiner (already written: `ensemble_v2_combine.py`).

Run from `06_Code/` with the project venv (`.venv`, GPU auto for CNNs). Status: [RUN] = to be run by Claude Code.

---

## Phase 1 — regenerate per-window probabilities (per-subject norm, LOSO, all 40 subjects) — [RUN]
Target output for every model: `results_ensemble_v2/proba/{MODEL}_sub{K:02d}.npz` with keys
`proba` (shape [n_windows, 4], columns in LABELS order = [DNS, STDUP, UPS, WAK]) and `y_true` ([n_windows]).
MODEL ∈ {SVM, RF, CNN, RESNET_SE}.

**1a. Classical (SVM, RF) — fast, CPU (~20–40 min total).** Patch `train_classical_loso.py`: it already saves `y_pred` (lines ~388, ~467 under `--save-preds`). Add a sibling `--save-proba` that, in the same per-fold loop, also saves `clf.predict_proba(X_test)` aligned to `y_true`. SVM must be built with `probability=True` (Platt scaling); RF already exposes `predict_proba`. Reuse the already-selected best params (SVM C=1 all subjects; RF from `results_loso_freq_persubj/*subjectwise.csv` `best_params`) — no GridSearch needed, so this is a cheap refit. Save to `results_ensemble_v2/proba/SVM_sub{K}.npz` and `RF_sub{K}.npz`. Ensure the class-column order matches LABELS (remap from `clf.classes_` if needed).
```
python train_classical_loso.py --features $FEAT --meta $META --model SVM --norm-mode per_subject --save-proba --proba-out results_ensemble_v2/proba --resume
python train_classical_loso.py --features $FEAT --meta $META --model RF  --norm-mode per_subject --save-proba --proba-out results_ensemble_v2/proba --resume
```

**1b. CNNs (SimpleEMGCNN + ResNet-SE) — GPU (~30–40 min + ~1–2 h).** Patch `run_cnn_arch_loso.py` to save per-window **softmax** for the held-out subject (apply `torch.softmax` to logits, move to CPU, save aligned to `y_true`) into `results_ensemble_v2/proba/{CNN|RESNET_SE}_sub{K}.npz`. Use **per-subject normalization** (matching the 0.754 / 0.782 headline runs), NOT the global-norm base used for the adaptation experiments.
```
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch simple    --epochs 40 --norm-mode per_subject --save-proba results_ensemble_v2/proba --model-tag CNN        --resume
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_se --epochs 40 --norm-mode per_subject --save-proba results_ensemble_v2/proba --model-tag RESNET_SE --resume
```
Sanity: the argmax of the saved probabilities must reproduce the known per-subject F1s (SVM 0.777, RF 0.773, CNN ~0.754, ResNet-SE 0.782). If not, the class-column order or normalization is off — fix before Phase 2.

## Phase 2 — compute the ensemble comparison — [RUN, fast, CPU]
```
python ensemble_v2_combine.py
```
Already written. It computes, for each model subset {SVM,RF,CNN}, {SVM,RF,ResNet-SE}, {SVM,RF,CNN,ResNet-SE}, and the pairwise sets: **hard vote, soft vote, weighted-soft (weighted by each model's LOSO F1), and stacking** (a LOSO-safe multinomial-logistic meta-learner over the concatenated probabilities). Outputs ranked `results_ensemble_v2/ensemble_v2_summary.csv` with paired Wilcoxon p vs the current `SVM+RF+CNN [hard]` = 0.7917. **The oracle bound is 0.889**, so treat anything up to there as reachable in principle.

## Phase 3 — wire the winner into the thesis — [RUN]
- If a method beats 0.7917 significantly, update **§4.10 + Table 4.10** (add soft/weighted/stacking rows and the ResNet-SE-inclusive ensemble), the **Abstract/Conclusion** "0.792" headline, **Table 4.18** (ensemble row), and **§5.8** (ensemble discussion). Report the paired Wilcoxon + d + BCa and fold the new comparison into the whole-thesis BH-FDR family (`stats_unified_fdr.py`, which becomes a 39+-test family).
- If nothing beats hard voting significantly, say so plainly: "hard majority voting is not improved upon by soft, weighted, or stacked combination at n=40, so it is retained for its simplicity" — that is itself a clean, honest result and answers the method question.
- **Add ensemble-method literature** (currently the thesis just does hard voting without citing the combiner literature). Real references to cite in §3.10/§4.10:
  - Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J. (1998). On combining classifiers. *IEEE TPAMI, 20*(3), 226–239. https://doi.org/10.1109/34.667881
  - Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1
  - Dietterich, T. G. (2000). Ensemble methods in machine learning. In *Multiple Classifier Systems* (LNCS 1857, pp. 1–15). https://doi.org/10.1007/3-540-45014-9_1
  - Kuncheva, L. I. (2004). *Combining Pattern Classifiers: Methods and Algorithms.* Wiley.

## Shared inputs
- `FEAT = features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz`
- `META = features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv`
- `NPZ  = windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz`
