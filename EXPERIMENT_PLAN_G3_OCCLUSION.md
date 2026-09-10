# Experiment plan: what does channel dropout actually do to electrode reliance? (W-2 Stage G3, re-scoped)

**Status:** ready to run. Written 2 September 2026, after W-2 Stage G1 returned Outcome R.
**Cost:** one 40-fold run, about 26 minutes GPU, plus free analysis.
**Supersedes:** §4.4 of `EXPERIMENT_PLAN_CHANNEL_DROPOUT.md`, on one point recorded in §1 below.
**Stage G4 is cancelled.** See §1.

---

## 0. Read this first

1. **The W-2 gate closed on Outcome R.** Squeeze-and-excitation is not the mechanism behind
   channel dropout: the interaction contrast is -0.78 pp, p = 0.221. Do not re-test it, and do
   not frame anything here as an SE question.
2. **G3 never depended on the SE hypothesis.** It asks which of the nine electrodes the model
   leans on and whether channel dropout changes that. It is the one surviving W-2 stage that
   Outcome R leaves fully intact, which is why it runs and G2 and G4 do not.
3. **Half the data already exists.** `results_cd_resnet_nose_chandrop/instr/occlusion.csv` holds
   40 subjects by 9 channels from the G1 run. Do not regenerate it.
4. **No model weights are saved anywhere,** so the matched baseline cannot be produced by
   post-hoc analysis. It needs the one run in §2.
5. **Do not edit `MSc Thesis.docx` or any chapter file.** Section 4.8.1 is written and correct
   without this. Any wording that follows from these numbers is Enam's.

---

## 1. Two corrections to the parent plan

**The baseline architecture.** §4.4 of `EXPERIMENT_PLAN_CHANNEL_DROPOUT.md` specifies the
occlusion baseline as `--arch resnet_se --augmentation none`. That was written before G1 ran.
G1's channel-dropout arm is `--arch resnet`, SE-free, so an SE-equipped baseline would reintroduce
exactly the variable G1 eliminated and confound the occlusion comparison with the SE main effect,
which is real (+2.23 pp unaugmented). **The baseline is `--arch resnet`.**

**Stage G4 is cancelled, on two independent grounds.** It measures the across-subject variance of
the SE gates; SE is no longer a candidate mechanism, and `se_gates.csv` was never written because
the G1 arm was correctly SE-free. The parent plan already called G4 "the weakest of the four" and
directed that G3 lead any write-up. Nothing is lost.

---

## 2. The one run

Every flag matches `results_cd_resnet_nose_chandrop` except `--augmentation`:

```
"<PY>" -u run_cnn_arch_loso.py \
  --npz windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
  --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
  --arch resnet --augmentation none \
  --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 \
  --norm-mode per_subject --resume \
  --out results_g3_noaug_instr --instrument results_g3_noaug_instr/instr
```

`run_g3_occlusion.sh` runs this with logging and a smoke fold first.

**This is a fresh run, not a re-read of `results_cd_resnet_noaug_repro`.** The instrumentation
needs the trained model in memory and no weights are saved, so the baseline has to be retrained.
Because the pipeline is not deterministic, its mean will not exactly equal the 0.7600 of
`results_cd_resnet_noaug_repro`. That is expected, and it doubles as a free reproduction check:
**the two means must agree within 1.5 pp** (the R-1 gate, roughly 3σ on a measured run-to-run SD
of 0.47 pp). If they do not, stop and report, because something has drifted since 1 September.

---

## 3. What is measured

`occlusion.csv` gives, per held-out subject and per channel, the macro-F1 the model achieves with
all nine channels (`f1_full`) and with that one channel zeroed (`f1_occluded`), and their
difference in percentage points (`drop_pp`). A subject's nine `drop_pp` values are its **reliance
profile**: how much the model's performance on that person depends on each electrode.

Two quantities, and the second matters more.

### 3.1 Within-subject reliance concentration

Does channel dropout stop the model leaning on one electrode? Concentration is the spread of the
nine values within a subject.

**The scale confound must be handled, and the parent plan did not address it.** The channel-dropout
model scores 0.825 against the baseline's 0.760, so its `drop_pp` values live on a different scale,
and a raw spread comparison would partly measure that difference rather than concentration. Each
subject's profile is therefore normalized to sum to one over its positive part before the spread is
taken, which makes the measure scale-free by construction. Raw spread is reported alongside, for
transparency, but the normalized figure is the one the verdict reads.

Negative `drop_pp` (occluding a channel *helps*) is real and is clipped to zero when forming the
normalized profile. The count of clipped entries is reported; if it is large the profile
construction is unreliable and the stage should be read cautiously.

### 3.2 Between-subject profile consistency

**This is the quantity that speaks to the thesis's central argument.** A model whose electrode
reliance is the same shape from person to person has not latched onto per-subject electrode
idiosyncrasy. That is precisely the claim §5.7 makes in prose and currently cannot support with a
measurement.

For each subject, correlate its normalized profile against each of the other 39 and take the mean.
That yields 40 values per condition, paired by subject, which is what makes a paired test possible;
comparing two pooled all-pairs correlations would give one number per condition and no valid test.
Spearman is primary, because nine noisy magnitudes rank more reliably than they scale; Pearson is
reported beside it.

**These 40 values are not independent** — every pair of subjects contributes to two of them — so
the paired Wilcoxon is anticonservative here. It is therefore backed by a subject-level
randomization test: for each subject independently, randomly swap its two profiles between the
conditions, recompute the difference in mean consistency, 10,000 times, seed 42. The randomization
p-value is the one to quote if the two disagree.

---

## 4. The decision rule, pre-registered

Family of two tests, §3.1 and §3.2. Holm correction within the family, never across. Paired
Wilcoxon, paired Cohen's d, BCa bootstrap 95% CI at 10,000 resamples, seed 42, matching the thesis
convention and reusing the estimators verified by hand in W-1.

| Outcome | Condition | What it licenses |
|---|---|---|
| **C. Consistency** | between-subject consistency significantly higher under channel dropout (Holm-adjusted, and the randomization test agrees) | §5.7's argument becomes measured rather than asserted. The thesis can state that channel dropout makes electrode reliance transfer across people, which is the claim the whole cross-subject framing rests on |
| **F. Flattening only** | concentration significantly lower under channel dropout, consistency not significant | A weaker but real result: the model stops depending on any single electrode, without evidence that the resulting profile transfers. Report as flattening, and do not extend it to a claim about generalization |
| **B. Both** | both significant | The strongest available outcome. Lead the write-up with consistency, since flattening is the mechanism and consistency is the consequence that matters |
| **N. Neither** | neither significant | §5.7's mechanism paragraph is an interpretation the data does not reach. It must be softened to say so. This is a result, and it is the one most worth having found before a viva rather than during one |

**Outcome N is a real possibility and must not be argued around.** Nine channels give a short
profile and 40 subjects give a noisy correlation; the study may simply lack the resolution. If the
effect is in the hypothesized direction but does not clear significance, say exactly that, quote
the CI, and let the interval carry the uncertainty. Do not report a direction as a finding.

---

## 5. Reproduction gates

1. Both `occlusion.csv` files cover the same 40 subjects, 9 channels each, 360 rows plus header.
2. `f1_full` in `occlusion.csv` matches that subject's `f1_macro` in `cnn_arch_subjectwise.csv`.
   The driver already asserts this per fold and warns on mismatch; check the log for warnings.
3. The instrumented baseline's mean is within 1.5 pp of `results_cd_resnet_noaug_repro` (§2).
4. The G0 RNG assertions in `run_cnn_arch_loso.py` did not fire.

---

## 6. Downstream impact

**None.** This changes no existing number. It is a new analysis standing beside the existing
results, and it touches no model, no ensemble, no causal result and no headline. Its only
consequence for the thesis is one or two sentences in §4.8.1 and a firming-up, or a softening, of
the mechanism paragraph in §5.7.

---

## 7. What to report

1. The outcome letter, C, F, B or N, in the first line.
2. The four reproduction gates of §5, before any result.
3. Concentration: normalized and raw, with CI, p, Holm-adjusted p and d, and the count of clipped
   negative entries.
4. Consistency: Spearman primary and Pearson beside it, with CI, p, Holm-adjusted p, d, and the
   randomization p-value. Say whether the two p-values agree.
5. The two run means and the §5 gate 3 comparison against 0.7600.
6. Wall-clock, against the 25.6 min the equivalent G1 run took.
7. One sentence on what this does **not** establish.
