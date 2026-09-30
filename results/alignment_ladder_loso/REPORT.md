# E-C1 — Downstream LOSO F1 for every rung of the alignment ladder

## Command run

```
& '.\.venv\Scripts\python.exe' -u run_alignment_ladder_loso.py --rungs 3,0,1,2,4 --resume
```

New script: `run_alignment_ladder_loso.py`. The five rung transforms
(`rung0_global_z` … `rung4_full_whiten_recolor`) are imported from
`analyze_between_subject_variance.py`, not reimplemented; each `Xr = rung(X,
subjects)` is built once over the full 26347×72 matrix, exactly as the
published ladder analysis does. Per rung: standard LOSO over the 40 subjects
with full nested GridSearchCV mirroring `train_classical_loso.py` — inner
5-fold `GroupKFold`, grid `clf__C ∈ {1,5,10}`, `clf__gamma='scale'`,
`SVC(kernel='rbf', class_weight='balanced')`, `scoring='f1_macro'`,
`refit=True`, GridSearchCV `n_jobs=1`. **Re-tuned per rung** — `_bestparams.json`
was not reused. No scaler in the pipeline: the rung transform is the
normalization, which is what makes rung 3 identical to the published
`--norm-mode per_subject` run. Checkpoint per subject, `--resume`. Compute was
run as subject-partitioned chains for throughput; the command above is the
final assembly pass (all 200 folds already on disk → summary + gates + join
only).

Outputs in `results_alignment_ladder_loso/`: `ladder_loso_{0..4}_SVM_subjectwise.csv`,
`alignment_ladder_loso_summary.csv`, `alignment_ladder_loso_stats.csv`, and
`alignment_ladder_full.csv` (the new Table 4.16). The published
`results_variance_decomposition/alignment_ladder.csv` was not modified.

## Validation gates — BOTH PASS

| gate | rung | this run | published | check |
|---|---|---|---|---|
| 1 (mandatory) | 3 `mean_scale` = `per_subject_zscore` | **0.776700** | 0.776700 (`results_loso_freq_persubj` SVM headline) | \|diff\| = 0.000000 ≤ 0.002 → **PASS** (bit-identical) |
| 2 | 0 `global_z` | **0.709404** | 0.708 (global baseline, StandardScaler on 39 train subjects in-Pipeline) | gap **+0.0014**, well under the ~0.005 the plan flagged as expected |

Rung 3 reproduces the published per-subject z-score SVM F1 exactly (C=1 for all
40 subjects, matching `_bestparams.json`), so the harness reproduces the old
number before any new rung is trusted. Rung 0's +0.0014 gap is in the predicted
direction and magnitude: `rung0_global_z` standardizes over all 40 pooled
subjects (a mild optimistic leak), whereas the published baseline fits the
scaler on the 39 training subjects only. Both definitions are stated here so
the thesis can report the value and the gap.

## Summary table — the new Table 4.16

`alignment_ladder_full.csv` (joined with the published probe/silhouette/MMD ladder):

| rung | name | MMD removed % | W1 removed % | subject-probe bal. acc. | silhouette by class | **LOSO F1 (SVM)** |
|---|---|---:|---:|---:|---:|---:|
| 0 | `global_z` | 0.0 | 0.0 | 0.777 | 0.0080 | 0.7094 |
| 1 | `mean_center` | 42.4 | 35.5 | 0.043 | 0.0190 | 0.7482 |
| 2 | `scale_only` | −26.7 | −35.3 | 0.909 | 0.0078 | 0.7186 |
| 3 | `mean_scale` | 54.4 | 62.3 | 0.024 | 0.0230 | **0.7767** (peak) |
| 4 | `full_whiten_recolor` | 73.0 | 75.2 | 0.012 | −0.0061 | **0.6752** |

Paired tests (`alignment_ladder_loso_stats.csv`, n = 40 subjects):

| comparison | mean ΔF1 | Wilcoxon p | Cohen's d (paired) |
|---|---:|---:|---:|
| rung 4 vs rung 3 | **−0.1015** | 3.6 × 10⁻¹² | **−2.34** |
| rung 3 vs rung 0 | +0.0673 | 3.1 × 10⁻¹⁰ | +1.48 |

## Does the result support or contradict the thesis as written?

**It supports the over-alignment account of Sections 4.13.1–4.13.2, and
strengthens it.** The plan pre-registered the falsifiable shape: F1 should rise
from rung 0 to rung 3 and then fall at rung 4, tracking the class-separability
silhouette rather than the monotonically-falling subject-identity probe. That
is exactly what happened — F1 climbs 0.709 → 0.777 as diagonal alignment is
added, then drops sharply to 0.675 when alignment is pushed to full covariance
whitening, a 0.10-point fall with a paired effect size of d = −2.34 (Wilcoxon
p ≈ 4 × 10⁻¹²). The F1 curve peaks at rung 3 with the silhouette, not with the
probe, so the "over-alignment destroys class structure and this costs
downstream accuracy" claim is now measured directly rather than inferred.

Crucially, this closes the associative-not-causal objection the PRC reviewers
raised. Section 4.13.2 currently reads "the classifier built on it performs
worse, at 72.4% against 77.7%", where 72.4% is CORAL's LOSO F1 from a different
transform in a different script (`results_loso_freq_coral`, verified 0.7236)
and 77.7% is the rung-3 probe/silhouette operator. Alignment strength, subject
identity, class separability and downstream F1 are now all read off the *same*
manipulated variable on the *same* 40-subject LOSO protocol. The direct rung-4
number is **0.6752**, which is *lower* than the CORAL proxy the thesis quoted
(0.7236), so the true over-alignment penalty is larger than the text implies.
**Recommended edit:** replace "72.4% (CORAL)" in Section 4.13.2 with the direct
figure — rung-4 LOSO SVM F1 = 0.675, identical protocol — and cite Table 4.16
as the single-axis dose-response.

Two secondary findings worth a sentence each in the thesis. (1) `mean_center`
(rung 1) does most of the alignment work on its own — it collapses the subject
probe from 0.777 to 0.043 and already lifts F1 to 0.748 — while (2) `scale_only`
(rung 2) is actively counterproductive for alignment: it *raises* the subject
probe to 0.909 and yields *negative* MMD/W1 removed (−27% / −35%), yet still
nudges F1 up to 0.719. Per-subject re-centering, not per-subject re-scaling, is
the operative half of the z-score, and scaling without centering can make
subjects more separable while remaining mildly useful for the classifier. This
is consistent with, and adds granularity to, the thesis's account of why the
simple per-subject z-score works.
