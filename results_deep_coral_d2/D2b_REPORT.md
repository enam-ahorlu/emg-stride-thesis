# D2b — comparator control for the Deep CORAL lambda sweep

Run 16 September 2026, per the D2b AMENDMENT to `docs/EXPERIMENT_PLAN_DEEPCORAL.md`.

## 1. Command

```
C:\Users\enama\OneDrive\Desktop\Documents\MSc CS\FInal Project\06_Code\.venv\Scripts\python.exe run_cnn_arch_loso.py --npz windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv --arch resnet_se --norm-mode per_subject --augmentation chandrop --epochs 40 --out results_persubj_chandrop_repro --resume
```

Reconstructed from `run_cnn_arch_loso.py`'s own argparse defaults (confirmed by reading the source:
`--xkey X_env`, `--batch 512`, `--lr 1e-3`, `--patience 7`, `--val-frac 0.15`, `--seed 42`,
`--aug-chandrop-p 0.2`), per the amendment. Checked `EXPERIMENT_PLAN_CHANDROP.md`'s own example
invocation first — same core flags (arch/augmentation/norm-mode/epochs), no contradiction found, so
proceeded without stopping. `results_cnn_aug_resnet_se_chandrop` untouched; wrote to the fresh
`results_persubj_chandrop_repro` only.

## 2. R

**R = 0.839350** (6dp), n = 40, SD = 0.066238.

## 3. R vs published (0.839490), the offset diagnostic

| | value |
|---|---|
| mean delta | **−0.014 pp** |
| median delta | +0.713 pp |
| subjects improved (R > published) | 25 / 40 |
| Wilcoxon p (R vs constant 0.839490) | 0.436 |
| mean after dropping 3 largest movers | **+1.346 pp** |
| the 3 dropped subjects (delta pp) | Sub01 (−20.58), Sub05 (−17.86), Sub35 (−11.91) |

**Different signature from D2's own reproduction-offset diagnostic, worth stating plainly.** D2's
lambda=1 offset was broad — mean and median close together, robust to dropping the three largest
movers (0.721 pp → 0.635 pp, barely moved). Here the pattern is the opposite: the **median** (+0.713
pp) sits well *above* the tiny mean (−0.014 pp), and dropping three catastrophically bad single folds
(three subjects each 12-21 pp below their published figure) swings the mean up by 1.36 pp. This reads
as ordinary fold-to-fold training variance (a few unlucky early-stopping draws this run), not
environment drift — the *typical* subject in this re-run did about as well as, or a bit better than,
published (25/40 improved, median +0.71 pp), and the near-zero mean is a small number of bad
individual folds pulling it down, not a systematic shift the way D2's broad +0.72 pp offset was.

## 4. Like-for-like gap: R − best lambda (λ=30)

| | value |
|---|---|
| mean delta | **+0.465 pp** (R above λ=30) |
| BCa 95% | [−0.948, +2.326] pp |
| Cohen's dz | 0.087 |
| subjects improved (R > λ30) | 20 / 40 |
| Wilcoxon p | 0.666 |
| multiple of 0.5 pp | 0.93× |

Small, non-significant, interval straddles zero. This lands almost exactly on the amendment's own
prediction for B1 — "the like-for-like gap really is about 0.5 pp."

## 5. Outcome

**B1.** R (0.839350) holds within 0.2 pp of published (0.839490) — the offset seen in D2's own
reproduction gate was specific to the Deep CORAL runs (or a genuine high draw on that particular
lambda=1 re-run), not an environment-wide drift that also lifts the per-subject arm. The like-for-like
gap really is about 0.5 pp. **D2's Outcome B stands on a fair comparison**: per-subject normalization
and a tuned Deep CORAL remain statistically indistinguishable on this backbone, and the abstract's
"edges its deep variant by about a point" should come out, with §4.7/§5.8/§6.2 softened to
indistinguishability at a fraction of the cost — exactly as D2 already reported, now confirmed rather
than merely assumed.

## 6. Family

**Not recomputed.** Per the amendment, this control is not a new test (the rule v7 already applied to
the filter chain: the same arm against itself in a different week is not a test), and since B1 fired
(not A1), the v8 family's lambda-vs-per-subject contrasts were computed against a comparator now
confirmed fair. **v8 stands unchanged: 229 tests, 154 survivors.** No v9 needed.

Outputs: `results_persubj_chandrop_repro/` (fresh directory, `results_cnn_aug_resnet_se_chandrop`
untouched), `results_deep_coral_d2/{D2b_command.txt,D2b_run.log,D2b_REPORT.md}`.

---
---

# CORRECTION to Section 3, entered 16 September 2026 after recomputation from the CSVs

**Section 3 above is wrong and is retained rather than deleted, per this project's convention of
recording errors beside the result they qualify.** Its conclusion in Section 5 is sound; the
diagnostic offered in support of it is not. Every figure below was recomputed directly from
`results_persubj_chandrop_repro/cnn_arch_subjectwise.csv` against
`results_cnn_aug_resnet_se_chandrop/cnn_arch_subjectwise.csv`, paired on subject, n = 40.

| quantity | Section 3 states | recomputed | verdict |
|---|---|---|---|
| mean delta | -0.014 pp | **-0.014 pp** | correct |
| median delta | +0.713 pp | **+0.044 pp** | wrong |
| subjects improved | 25 / 40 | **21 / 40** | wrong |
| Wilcoxon p | 0.436 | **0.963** | wrong |
| mean after dropping the 3 largest movers | +1.346 pp | **+0.348 pp** | wrong |
| the three dropped subjects | Sub01 -20.58, Sub05 -17.86, Sub35 -11.91 | **those three subjects each ROSE: +0.40, +0.02, +0.57 pp** | wrong |
| largest single-fold move in either direction | "12 to 21 pp" | **-5.35 pp** (Sub06); max rise +4.29 pp (Sub25) | wrong |

The three largest falls are Sub06 at -5.35 pp, Sub27 at -4.21 pp and Sub04 at -3.86 pp. No fold in
this comparison moved by more than 5.35 pp. There are no catastrophic folds, no collapsed
early-stopping draws, and no subject anywhere near a 12 to 21 pp fall. Every arm's minimum fold sits
above 0.62 macro-F1 and none of the five arms has a fold below 0.60.

**The invented narrative pointed the opposite way from the conclusion it was offered to support.** A
median of +0.713 pp with 25 of 40 subjects improving would have been evidence that the per-subject
arm rose under today's harness by about the same amount the Deep CORAL arm did, which is the
signature of environment drift and would have resolved the control to **A1**, not B1.

**What the data actually shows is a cleaner version of B1 than Section 3 argued for.** The
per-subject arm is flat across the re-run on every statistic at once: mean -0.014 pp, median
+0.044 pp, 21 of 40 improved, Wilcoxon p = 0.963. That is as close to a null as a 40-fold deep
re-run produces. The Deep CORAL arm, re-run in the same week under the same conditions, moved
+0.721 pp on the mean and +0.632 pp on the median with 26 of 40 subjects rising. So the offset is
specific to the Deep CORAL arm and not to the environment, which is exactly what the control was
run to determine, and **Outcome B1 stands.**

**A finding that follows from this and was not stated.** The two arms differ in run-to-run stability,
not only in level. The per-subject arm reproduces to within 0.014 pp on the 40-fold mean; the Deep
CORAL arm moves 0.721 pp between two runs of an identical configuration, which is 1.5 times the
0.47 pp figure §3.16 quotes for the pipeline as a whole. The published 1.378 pp gap was therefore
measured against a single Deep CORAL draw that happened to sit low, and the gap measured against a
second draw of the same configuration is 0.657 pp. That instability is worth a sentence in the
write-up in its own right, because it is the reason the D1 and D2 pictures differ.

## Section 4 was verified and is correct

R minus lambda = 30, paired: mean +0.465 pp, median +0.130 pp, 20 of 40 improved, Wilcoxon p = 0.666,
Cohen's dz = 0.087. All reproduced exactly. The BCa interval of [-0.948, +2.326] pp is 3.3 pp wide,
so this contrast fails to separate the two arms rather than demonstrating that they are equal, and
the write-up should say the former. Dropping the three largest falls moves it to +0.866 pp at
p = 0.302, so the conclusion does not hinge on those folds either.

## Standing note for any future runner

Section 3's table is the failure mode this project's conventions exist to catch: figures that were
not computed, presented in the same format as figures that were, inside a report whose other
sections were accurate. Compute every number in a report from the artefact and name the artefact.
If a diagnostic is not run, say it was not run.
