# EXPERIMENT_PLAN_CHANDROP.md — promote channel-dropout ResNet-SE to the deep model of record

**Why.** The augmentation ablation showed channel dropout lifts ResNet-SE from 0.782 to **0.840** LOSO macro-F1 (+5.7 pp, 36/40 subjects, verified broad and mechanistically sound: robustness to per-subject electrode/channel variation, synergizing with the SE channel-attention block). This is now the **optimized deep model of record**, so every downstream analysis that was computed on the plain ResNet-SE must be redone on the channel-dropout ResNet-SE (call it **ResNet-SE+CD**) to keep the thesis frontier-consistent.

**Verified on disk:** `results_cnn_aug_resnet_se_chandrop` = 0.8395 (solo). **NOT verified:** the ~0.86 ensemble — the chandrop run did not save per-window probabilities, so Phase 1 must regenerate them. G7 (SD-vs-LOSO gap) is unaffected: ResNet-SE+CD's per-subject SD is still invalid (data starvation), so it remains LOSO-only.

Run from `06_Code/` with the venv, GPU, `--resume`. Do NOT fabricate numbers. Report honestly if the ensemble does NOT beat 0.815.

**Inputs.** SIAT: `NPZ = windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz`, `META = features_out/..._features_meta.csv`, `FEAT = features_out/..._features_ext.npz`. ENABL3S: the files `adapt_external_dataset.py` produces. SVM/RF probabilities already exist in `results_ensemble_v2/proba/`.

---

## Phase 1 — regenerate ResNet-SE+CD with saved probabilities (SIAT) — [RUN, GPU]
Re-run the winning config saving per-window softmax + hard preds, so the ensemble and per-class/confusion analyses can be computed.
```
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_se --augmentation chandrop \
    --norm-mode per_subject --epochs 40 --save-proba results_chandrop_resnet_se/proba --out results_chandrop_resnet_se --resume
```
**SANITY:** the summary must reproduce ~0.840; argmax of the saved probs must give the same. If not, stop and fix. Save probs as `results_chandrop_resnet_se/proba/RESNET_SE_CD_sub{K:02d}.npz` (keys `proba` [n,4] in LABELS order [DNS,STDUP,UPS,WAK], `y_true`).

## Phase 2 — ensemble, per-class, confusion, subject-difficulty (from probs) — [RUN, fast CPU]
1. **Ensemble:** run `ensemble_v2_combine.py` with SVM + RF + **ResNet-SE+CD** (point it at the new proba dir). This gives the real chandrop-inclusive hard/soft/weighted/stacking numbers vs the 0.815 reference. Report whether soft/weighted beats 0.815.
2. **Per-class + confusion + subject-difficulty** for ResNet-SE+CD and the new best ensemble (reuse the analysis in `results_ensemble_v2/headline_error_analysis`): per-class F1, DNS→WAK critical-error rate, subject-difficulty range and cross-model Pearson r (ResNet-SE+CD vs SVM, RF).

## Phase 3 — external validation on ENABL3S — [RUN, GPU]
Run ResNet-SE+CD on ENABL3S (per-subject and global), and the ensemble on ENABL3S, mirroring the base-ResNet-SE external runs.
```
python run_cnn_arch_loso.py --npz <ENABL3S_NPZ> --meta <ENABL3S_META> --arch resnet_se --augmentation chandrop --norm-mode per_subject --out results_ext_chandrop_resnet_se_persubj --resume
python run_cnn_arch_loso.py --npz <ENABL3S_NPZ> --meta <ENABL3S_META> --arch resnet_se --augmentation chandrop --norm-mode global      --out results_ext_chandrop_resnet_se_global  --resume
```
Report whether the +per-subject-over-global and ensemble findings still hold; note honestly if the deep model still trails classical at n=10.

## Phase 4 — supervised calibration on ResNet-SE+CD — [RUN, GPU]
```
python run_cnn_calibration_loso.py --npz $NPZ --meta $META --arch resnet_se --augmentation chandrop --ft-epochs 3 --out results_cnn_calibration_chandrop_resnet_se --resume
python run_cnn_calibration_multidraw.py --npz $NPZ --meta $META --arch resnet_se --augmentation chandrop --draws 5 --out results_cnn_calibration_chandrop_resnet_se_multidraw --resume
```

## Phase 5 — CNN-side domain adaptation on ResNet-SE+CD — [RUN, GPU]
Redo the "simple normalization vs learned adaptation" comparison on the new backbone so §4.6/§4.13 stay consistent: Deep CORAL and AdaBN on ResNet-SE+CD (global-norm base), plus causal AdaBN.
```
python run_deep_coral_cnn_loso.py --npz $NPZ --meta $META --arch resnet_se --augmentation chandrop --coral-lambda 1.0 --epochs 40 --out results_deep_coral_chandrop --resume
python run_adabn_cnn_loso.py       --npz $NPZ --meta $META --arch resnet_se --augmentation chandrop --epochs 40 --out results_adabn_chandrop --resume
python run_adabn_causal_loso.py    --npz $NPZ --meta $META --arch resnet_se --augmentation chandrop --out results_adabn_causal_chandrop --resume
```
The claim to re-check: per-subject-normalized ResNet-SE+CD (0.840) still beats Deep CORAL and AdaBN on the same backbone.

## Phase 6 — statistics — [RUN, CPU]
Paired Wilcoxon + Cohen's d + BCa for: ResNet-SE+CD vs base ResNet-SE, vs SVM, vs RF; the new ensemble vs 0.815; each augmentation vs the 0.782 none-baseline. Fold every new test into `stats_unified_fdr.py` (the family grows again) and report the updated survive/total count.

## Phase 7 — figures — [RUN, CPU]
Regenerate the frontier figures with ResNet-SE+CD in place of plain ResNet-SE: per-class, subject-difficulty, confusion, external, calibration, optimization journey (new endpoint), deployability; and add an **augmentation figure** (ResNet-SE: none / gaussian / chandrop / timemask / combined), which the thesis currently lacks. Match the existing `make_frontier_figs.py` style.

---

## After the runs — hand back for the write-up cascade
Leave clean summary CSVs + regenerated PNGs; do NOT edit the chapters (the write-up cascade — §4.8 augmentation as a headline finding, §4.9 ensemble, §4.10 journey, the model-comparison/per-class/subject-difficulty/confusion/external/calibration/deployability updates, and the figure swaps — will be done in one consistent pass). Report: (1) the new ensemble number and whether it beats 0.815; (2) whether external / DA / calibration findings still hold on ResNet-SE+CD; (3) any finding that did NOT replicate.
