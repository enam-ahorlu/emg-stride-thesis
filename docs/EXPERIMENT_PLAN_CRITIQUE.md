# EXPERIMENT_PLAN_CRITIQUE.md — the five experiments that answer the external critique

**Why.** Five external LLM reviews of the 85.8% thesis were triaged in `CRITIQUE_TRIAGE.md` (root, 39 items). Nobody disputed result validity; the substantive attacks are on framing, plus five places where a real run would convert a defensive sentence into an owned number. This plan is those five runs, in execution order. Everything else from the triage is handled in writing (see `FRAMING_CHECKLIST.md` in the project root).

**Explicitly out of scope, by decision (25 July 2026):** T17b (wider hyperparameter grid) and T19/T24 (500 ms window / MNF-MDF-drop LOSO ablation). These will be conceded or reframed in prose, not run. Do not start them.

**Conventions (same as the other plans).** Run from `06_Code/` with the venv Python (`& '.\.venv\Scripts\python.exe' ...`). GPU auto-used for CNN work. `--resume` everywhere. Every experiment writes `*_subjectwise.csv` + `*_summary.csv` into its own `results_<experiment>/`. Seed 42. `LABELS = ["DNS","STDUP","UPS","WAK"]` (alphabetical = encode order). GridSearchCV `n_jobs` stays 1 (Windows loky deadlock); RF parallelism via `--rf-n-jobs`. **Do not fabricate numbers. If a result contradicts the thesis, report it plainly — every one of these five is designed so that either outcome is usable.**

**Shared inputs.**
```
FEAT = features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz
META = features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
NPZ  = windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz          # key X_env, (N, 9, 500)
```
**Existing artifacts you will reuse (verified on disk):**
- `results_ensemble_v2/proba/{SVM,RF,CNN,RESNET_SE}_sub{K:02d}.npz` — transductive per-subject probabilities, base ResNet-SE (160 files).
- `results_ensemble_v2/proba_aug_chandrop/{SVM,RF,CNN,RESNET_SE}_sub{K:02d}.npz` — **the headline set**, `RESNET_SE` here is ResNet-SE+CD (160 files).
- `results_ensemble_v2_chandrop/ensemble_v2_summary.csv` — the headline table (SVM+RESNET_SE soft = **0.8579**, stacking 0.8604, SVM+RF+RESNET_SE soft 0.8467).
- `results_loso_freq_persubj/*_subjectwise.csv` — per-subject `best_params` (SVM C=1 for all 40; RF n_estimators 500/400/200).
- `results_loso_freq_streaming/` — the classical causal study (SVM calib100 0.745, RF 0.707) and `streaming_buffer_rescore_v2.csv`.
- `results_adabn_causal_chandrop/adabn_causal_summary.csv` — causal AdaBN on the CD backbone (calib100 **0.7679**, calib50 0.7634, calib25 0.7303, running 0.4764).

**Statistics for every experiment.** Paired Wilcoxon signed-rank + Cohen's d + deterministic BCa CI (RNG seeded from `hashlib.blake2b(data_bytes).digest()`, never Python `hash()`). Fold every new test into `stats_unified_fdr.py` and report the updated survive/total count for the whole-thesis family.

---

## E1 — Between-subject variance decomposition (triage T25) — [RUN, CPU, no training]

**The question.** *Why* does per-subject z-scoring work, and why does CORAL add so little on top? Right now the thesis shows the effect but not the mechanism, and two reviewers (C1 #17, C3) turn that into "it's just first-order alignment" as though it were a weakness. Decomposing the between-subject shift turns their critique into your own analysis. Triage calls this the single best way to raise the ceiling of the contribution.

**New script:** `analyze_between_subject_variance.py`. Output dir `results_variance_decomposition/`.

**Part A — variance components.** For each of the 72 features, estimate between-subject vs within-subject variance and the intraclass correlation `ICC = s2_between / (s2_between + s2_within)`. **Compute this within each movement class and then pool**, so differences in class composition across subjects cannot masquerade as subject effect. Report ICC distribution overall, per feature family (MAV, RMS, WL, ZC, WAMP, MNF, MDF, spectral power) and per channel (9). → `variance_components.csv`.
*Expected story:* amplitude-domain features (MAV/RMS/WL) carry most of the subject effect; frequency-domain features less. If that holds it explains mechanistically why a per-channel amplitude rescale is the right lever.

**Part B — the alignment ladder (the core result).** Apply a ladder of increasingly expressive alignment operators to each subject's features and measure how much between-subject discrepancy each removes:

| Rung | Operator | What it models |
|---|---|---|
| 0 | global z-score only | baseline shift |
| 1 | per-subject **mean centering** only | first-order location |
| 2 | per-subject **scale** only (divide by SD, no centering) | first-order scale |
| 3 | per-subject **mean + scale** (the thesis method) | diagonal first+second order |
| 4 | per-subject **full whitening / CORAL recolor** | full second order (covariance) |

For each rung compute three discrepancy measures: (i) mean pairwise **MMD** (RBF kernel, median heuristic bandwidth) between subjects, computed class-conditionally then averaged; (ii) mean per-feature **Wasserstein-1** between subject pairs; (iii) a **subject-identity probe** — logistic regression (or linear SVM) trained to predict subject ID from the features under 5-fold CV, balanced accuracy against the 1/40 = 0.025 chance floor. The probe is the most legible: the more subject identity survives, the more subject-specific nuisance remains. → `alignment_ladder.csv`.

**The number the thesis wants:** the fraction of between-subject discrepancy removed at each rung, e.g. "mean centering alone removes X%, adding scale takes it to Y%, and full covariance alignment only reaches Z% — which is why CORAL buys +1.6/+2.5 pp where per-subject z-scoring buys +6.9/+5.1."

**Part C — does distributional outlierness explain subject difficulty?** For each subject compute distance from the pooled training distribution (MMD to the other 39, and Mahalanobis distance of the subject mean vector) under global norm and under per-subject norm. Spearman-correlate against that subject's LOSO macro-F1 from `results_loso_freq_persubj` (SVM and RF) and against the ResNet-SE+CD per-subject F1. → `subject_distance_vs_f1.csv`.
*Payoff:* if distance predicts difficulty under global norm but stops predicting it after per-subject norm, that is a clean mechanistic account of the ~30 pp subject-difficulty range and of what normalization fixed.

**Figures:** ICC histogram by feature family; the alignment ladder as a bar chart of discrepancy removed; the distance-vs-F1 scatter with the Spearman rho. Match `make_frontier_figs_cd.py` style, save to `report_figs/new_experiments/`.

**Wires into:** a new short Results subsection under the normalization analysis, plus Discussion §5.10 (mechanism), and it directly defuses the "just first-order alignment" line.

---

## E2 — Feature-space visualization under different normalizations (triage T38) — [RUN, CPU, no training]

**The question.** Two reviewers asked for t-SNE/UMAP of the feature space; the triage confirmed it is genuinely absent (their "no confusion matrices" complaint was a false alarm — Fig 4.6 and 4.16 exist). Cheapest experiment on the list, and it is the visual companion to E1, so run it right after.

**New script:** `make_feature_space_viz.py`. Outputs to `report_figs/new_experiments/` + `results_variance_decomposition/embedding_metrics.csv`.

**Design.** Stratified subsample of 200 windows per subject (8,000 points), seed 42, balanced across classes as far as availability allows. Embed the Freq-72 features under three conditions — **global norm**, **per-subject z**, **CORAL-aligned** — with both t-SNE (perplexity 30) and UMAP. If `umap-learn` is not in the venv, `pip install umap-learn` into it; if that fails, ship t-SNE only and say so.

**Panels.** A 2 (method) × 3 (normalization) grid, each panel drawn twice: once colored by **subject** (40 colors), once by **class** (4 colors). Six-panel figure per coloring is fine.

**Do not leave it qualitative.** For every condition also report, in `embedding_metrics.csv`: silhouette score by subject and by class (computed in the original 72-D space, not the embedding), plus the subject-identity probe accuracy from E1 so the figure and the number agree.

*The claim it should support:* under global norm the data organize by subject; under per-subject z the subject structure dissolves and class structure surfaces. **If the picture does not show that, say so** — it would be a real finding about how much of the gain is not distributional and would need to be reconciled with E1.

**Wires into:** interpretability figure in Results (beside the normalization ablation) and Discussion §5.10; answers T38.

---

## E3 — Deployable (causal) score for the headline ensemble (triage T2) — [RUN, GPU + CPU]

**The question, and why it matters most defensively.** The critics assert the deployable number is "≈81%." That is *their extrapolation*: the causal fraction was measured on SVM/RF (0.745/0.707) and on causal AdaBN (0.768) but **never on the 85.8% ensemble**. Run it and you own the number instead of inheriting theirs. This is the single highest-value defensive run on the list.

**New script:** `run_causal_ensemble.py`. Output dir `results_causal_ensemble/`.

**Protocol.** Both ensemble members must be normalized causally, and scoring excludes the calibration buffer, exactly as the §4.14 study and `rescore_streaming_buffer_v2.py` already do. Configs: `calib25`, `calib50`, `calib100` (skip `running` — causal AdaBN showed it collapses to 0.476 on the deep side, and the classical running config is already reported).

**Step 0 — regenerate `_bestparams.json` (it was deleted).** Rebuild from the `best_params` column of `results_loso_freq_persubj/*_subjectwise.csv` into `{model: {subject_int: params_dict}}`. Sanity: SVM should be `C=1` for all 40 subjects.

**Step 1 — causal SVM probabilities.** Reuse `normalise_test_subject(..., mode="calib", k=K)` from `run_streaming_norm_loso.py` for the held-out subject and `per_subject_transductive` for the training subjects (unchanged — training subjects are offline data, their whole sessions are legitimately available). Fit `SVC(kernel="rbf", C=best, class_weight="balanced", probability=True)` and save `results_causal_ensemble/proba_calib{K}/SVM_sub{S:02d}.npz` with keys `proba` [n,4] in LABELS order, `y_true`, and **`is_buffer`** (boolean, True for the first K time-ordered windows).
**Honesty check that must be run first:** `probability=True` fits Platt scaling internally via an extra CV, so the probability-SVM is not byte-identical to the decision-function SVM behind the published 0.7767. Re-score the *transductive* config with `probability=True` and confirm it lands within ~0.005 of 0.7767. If it does not, fall back to `decision_function` → softmax and note the choice in the script docstring. Either way, state which was used.

**Step 2 — causal ResNet-SE+CD probabilities.** Train exactly as the headline model (`--arch resnet_se --augmentation chandrop`, 40 epochs, train-fold source data), but normalize the **held-out** subject's envelope windows with causal calib-K statistics: per-channel mean/std over axes (0,2) computed from the **first K time-ordered windows only** — the 3-D analogue of `causal_calib_stats`, matching `per_subject_zscore_3d`'s axis convention. Note this is *input* normalization, not the BN-stat swap of `run_adabn_causal_loso.py`; both are causal, they are different mechanisms, and the comparison between them is itself worth a sentence. Save probabilities as `RESNET_SE_sub{S:02d}.npz` into the same `proba_calib{K}/` dirs with the same keys.

**Step 3 — combine and score.** For each K:
```
python ensemble_v2_combine.py --proba-dir results_causal_ensemble/proba_calib100 --out results_causal_ensemble/calib100
```
then re-score with `is_buffer == False` rows only (buffer-excluded, faithful to the §4.14 convention). Report both buffer-included and buffer-excluded so the comparison to the existing streaming table is exact.

**Report.**

| Row | What |
|---|---|
| Transductive (upper bound) | SVM+ResNet-SE+CD soft = 0.858 |
| Causal calib-25 / 50 / 100 | the measured deployable ensemble, buffer-excluded |
| Attribution | causal SVM solo, causal ResNet-SE+CD solo at each K |
| Reference | causal AdaBN calib100 = 0.7679, classical SVM calib100 = 0.745/0.748 |

Paired Wilcoxon + d + BCa for causal-vs-transductive and causal-vs-global-norm-ensemble. **The deliverable is one sentence for the abstract:** "the transductive ensemble reaches 0.858 offline; under a strictly causal K-window calibration buffer the same locked pipeline retains X, so the deployable figure is Y pp lower." Whatever X is, it is now measured, not conceded.

**Wires into:** abstract + Results §4.14 + Discussion deployability, and it converts triage item T2 from a framing concession into a result.

---

## E4 — Within-subject baseline at matched label budget (triage T11) — [RUN, CPU]

**The question.** C3 #7: if you can collect a handful of *labelled* windows from the new subject, do you need cross-subject transfer at all? The thesis argues label-free per-subject normalization reduces the calibration burden; that argument is only complete once you show how many labels a purely subject-specific model would need to beat it. Triage calls this the most worthwhile new experiment.

**New script:** `run_within_subject_baseline.py`. Output dir `results_within_subject/`.

**Three regimes, all scored on exactly the same test windows** (the subject's session minus the first N labelled windows per class, so no regime gets to test on data another regime trained on):

- **A — subject-specific only.** Train on that subject's own first N labelled windows per class, no source data at all. N ∈ {5, 10, 20, 50, 100} per class. SVM primary (C=1, RBF, balanced), RF secondary.
- **B — cross-subject, zero labels (the thesis pipeline).** The existing per-subject-normalized LOSO model, re-scored on the same reduced test set so the comparison is like-for-like. Do *not* reuse the published 0.7767 directly — it is scored on the full session; re-score it on the truncated test set.
- **C — cross-subject + the same N labels.** Source training data plus that subject's N labelled windows pooled in. This is the classical analogue of the CNN calibration study (which already gives ResNet-SE+CD K=20 → 0.854), so quote that alongside.

**Draws.** Two variants, mirroring the existing calibration convention: the deterministic **first-N** windows (deployment-realistic, conservative) and **5 random draws** of N (robustness), reporting mean and spread across draws.

**The headline number:** the crossover N* where regime A overtakes regime B — "a purely subject-specific model needs at least N* labelled windows per class before it beats our label-free pipeline." Plot the three curves (F1 vs N labels/class, with per-subject spread) into `report_figs/new_experiments/within_subject_learning_curve.png`.

*Either outcome is usable.* If N* is large, the label-free pipeline is vindicated on exactly the axis the reviewer raised. If N* is small, that is an honest and interesting deployability finding: it says the practical recommendation is a short *labelled* calibration, and the contribution shifts (correctly) toward what happens when labels are unavailable.

**Wires into:** a new Results subsection on the calibration/deployability trade-off, Discussion §5.12, and it strengthens the T7 reframe (normalization reduces the requirement from labelled to unlabelled data, it does not eliminate calibration).

---

## E5 — RF probability calibration and re-vote (triage T15) — [RUN, CPU]

**The question.** C3 #4: the best ensemble drops the Random Forest. The thesis reads that as RF being too weak to contribute (0.773 diluting 0.840). The alternative mechanism is that RF's *probabilities* are poorly calibrated, and soft voting punishes miscalibration rather than weakness. Both are plausible; only a run distinguishes them.

**New script:** `run_rf_calibrated_ensemble.py`. Output dir `results_rf_calibrated/`.

**Step 1 — quantify miscalibration from what already exists.** From `results_ensemble_v2/proba_aug_chandrop/`, compute per model (SVM, RF, ResNet-SE+CD) the **expected calibration error** (15 equal-mass bins), **Brier score**, and mean maximum confidence, pooled and per class. Draw reliability diagrams. → `calibration_metrics.csv` + `report_figs/new_experiments/reliability_diagrams.png`. This alone is publishable as a sentence even if step 2 changes nothing.

**Step 2 — recalibrate RF inside the LOSO loop.** `CalibratedClassifierCV(RandomForestClassifier(...), method=..., cv=GroupKFold(5))` grouped by **training** subject, so calibration never sees the held-out subject. Run both `method="isotonic"` and `method="sigmoid"` (Platt); isotonic is the more flexible but needs data, and with ~30k training windows it is affordable. Save calibrated probabilities to `results_rf_calibrated/proba/RF_sub{S:02d}.npz`, plus the unchanged `SVM_*` and `RESNET_SE_*` copied or symlinked in so the combiner sees a complete set.

**Step 3 — re-vote.**
```
python ensemble_v2_combine.py --proba-dir results_rf_calibrated/proba --out results_rf_calibrated
```
The decisive comparison: does **SVM + RF_calibrated + ResNet-SE+CD** now reach or beat **SVM + ResNet-SE+CD = 0.8579**? Also re-check the stacking row (0.8604) since stacking is less sensitive to miscalibration and its near-tie with soft voting is itself evidence about the mechanism.

**Step 4 — optional symmetry.** SVC with `probability=True` is already Platt-scaled; the CNN softmax is not calibrated at all. If cheap, add temperature scaling for the ResNet-SE+CD probabilities (fit the temperature on the *training* subjects' validation split only) and report whether that moves the ensemble too.

**Verdict logic, written before the result so it cannot be rationalized after:**
- If calibrated RF re-enters the winning ensemble → the thesis sentence improves from "RF dilutes the vote" to "**uncalibrated** RF dilutes the vote," which is more mechanistic and directly answers C3. **But report the latency consequence honestly:** RF back in the ensemble takes the headline from ~4 ms/window to ~30 ms (RF single-window is ~26 ms). State the accuracy/latency trade-off and say which configuration the thesis recommends.
- If calibrated RF still does not help → the strength/complementarity reading stands, now backed by a direct test rather than an assumption, and T15 is closed empirically.

**Wires into:** Results §4.9 (ensemble composition), Discussion, and it retires triage items T15 and T16 together (the "counter-intuitive" wording gets replaced by whichever mechanism the data supports).

---

## Execution order and rough cost

| # | Experiment | Compute | Depends on |
|---|---|---|---|
| E1 | Variance decomposition | CPU, ~1 h | nothing (features only) |
| E2 | t-SNE / UMAP feature space | CPU, ~30 min | reuses E1's probe code |
| E3 | Causal ensemble | GPU (40 folds × 3 K) + CPU | `_bestparams.json` regenerated first |
| E4 | Within-subject baseline | CPU, several hours (5 N × 40 subjects × 2 draw modes) | nothing |
| E5 | RF calibration + re-vote | CPU, moderate (isotonic × 40 folds) | existing proba dirs |

E1, E2, E4 and E5 are CPU-only and can interleave with E3's GPU folds.

---

## After the runs — hand back for the write-up

Leave clean `*_subjectwise.csv` + `*_summary.csv` per experiment and the regenerated PNGs in `report_figs/new_experiments/`. **Do not edit the thesis chapters** — the write-up cascade is done in one consistent pass afterwards, together with the framing items in `FRAMING_CHECKLIST.md`.

Report back, per experiment: (1) the headline number; (2) whether it supports or contradicts the current thesis claim; (3) the paired test + effect size + BCa CI; (4) the updated whole-thesis FDR survive/total count; (5) anything that failed to run or that you had to change in the design, and why. Add each new script and its outputs to `REPRODUCE.md`.
