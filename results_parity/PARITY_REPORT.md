# Channel-dropout parity programme: consolidated outcome

Executor: Claude Code. No thesis file was edited. Section 4.17 was not touched. The deep model of record (channel-dropout resnet_se, 0.840) is unchanged. Stage P-4 was not started.

| Stage | Outcome | One line |
|---|---|---|
| P-0 literature | done | 4 sources verified; Pereira et al. is ICASSP 2024, not a preprint; Srivastava Table 10 read in full. See p0_literature.md. |
| P-1 mechanism to outcome | S | Separated at the boundary. Occlusion cost keeps falling with rate while F1 plateaus / falls: more invariance keeps buyin |
| P-2 between-subject transfer | X | Ranks are not scale-free in practice. The shrink control reproduces 142% of the baseline-to-CD agreement deficit on simu |
| P-3 external replication | R | Replicates. Primary contrast positive, Holm p < 0.05, d >= 0.5. The augmentation gain is not SIAT-specific; goes into se |
| P-5 published alternative | M | Matches. The primary contrast is within 2.0 pp and not Holm-significant, and the secondary is clearly positive. A third  |
| P-6 perturbation ladder | V | Variance is what matters. Neither contrast reaches 2.0 pp with Holm significance: the mean shift does nothing material a |
| P-6 extension | DIVERGE | DIVERGE. The curves separate materially at SD 0.50 (+2.81 pp). Where they diverge says how the form matters; report the  |
| P-7 high-dose decomposition | M | The mean shift carries it. What predominantly breaks channel dropout at high dose is the collapse in expected activation |

## New paired Wilcoxon tests for the whole-thesis Benjamini-Hochberg family (section 6.3)

Section 4.17 is not edited here. recompute_unified_fdr_v2.py runs the family in one pass.

| test | raw p |
|---|---|
| P-1a cost p0.1->p0.2 | 4.73e-05 |
| P-1a cost p0.2->p0.3 | 0.1029 |
| P-1a cost p0.3->p0.5 | 0.06415 |
| P-2 per-subject rank consistency, CD vs baseline | 1.859e-06 |
| P-3 external CD vs no aug, per-subject norm | 0.001953 |
| P-3 external CD vs no aug, global norm | 0.2754 |
| P-5 subset vs channel dropout | 0.3403 |
| P-5 subset vs no augmentation | 1e-10 |
| P-6 C1 channel dropout vs mean-preserving chandrop | 0.3011 |
| P-6 C2 mean-preserving chandrop vs gain jitter | 0.01303 |
| P-6 ext matched-variance SD 0.30 (FDR family) | 0.4049 |
| P-6 ext matched-variance SD 0.40 (cross-check, SD 0.40 already in core) | 0.02885 |
| P-6 ext matched-variance SD 0.50 (FDR family) | 2.275e-07 |
| P-7 D1 mean-preserving chandrop vs channel dropout p0.5 | 4.099e-05 |
| P-7 D2 gain jitter sd0.50 vs mean-preserving chandrop | 0.002639 |

Recommended additive count: 14 (15 listed; the SD 0.40 extension row duplicates information already in the P-6 core arms and is a cross-check, not an additive test).

Reported separately (not paired Wilcoxon, per section 6.3, as with the G3 randomization test): P-1a Spearman(rate, cost) and Page trend and their permutation context; P-1b and P-1c Spearman correlations with their 10,000-draw permutation nulls; P-2 subject-level randomization test and the two shuffle-null validity checks.

## Files

results_parity/: p0_literature.md, p1_*, p2_*, p3_*, p5_*, p6_*, p6_extension_*, parity_outcome.json, this report.
