# Experiment plan: where the residual problem lives (P-8, P-9, P-10)

> **P-8 RUN 3 September 2026. OUTCOME I (idiosyncratic but inconsequential).** No GPU.
> - **Column-to-channel mapping (Freq-72, feature-major):** MAV(9) RMS(9) WL(9) ZC(9) WAMP(9)
>   MNF(9) MDF(9) SpectralPower(9); column j -> family j//9, channel j%9. Channel c's 8 features
>   are columns {c, c+9, c+18, c+27, c+36, c+45, c+54, c+63}. Channel 0..8 = TFL, RF, VM, SMB,
>   Upper TA, Lower TA, Lateral GC, Medial GC, SOL.
> - **40x9 within-subject per-channel LDA macro-F1 matrix:** overall mean 0.5150, every cell well
>   above chance (0% within 0.05 of 0.25), so not outcome X. Before/after-normalization contrast
>   not run (LDA invariant to the diagonal map; vacuous).
> - **Consensus ranking (high to low informativeness):** Lower TA 0.570, Upper TA 0.548,
>   Lateral GC 0.525, SMB 0.522, SOL 0.519, TFL 0.504, Medial GC 0.501, VM 0.477, RF 0.469.
> - **Test A:** mean pairwise Spearman +0.167 over 780 pairs, reliably above the within-subject
>   shuffle null (p = 1e-4) but far below the +1.000 ceiling. Orderings are weakly shared.
> - **Test B1** (consensus agreement vs un-augmented ResNet-SE LOSO F1): rho +0.149, perm p 0.355,
>   Holm p 0.355. Sign positive as the account predicts, not significant.
> - **Test B2** (consensus agreement vs channel-dropout gain): rho -0.226, perm p 0.164,
>   Holm p 0.329. Sign negative as predicted, not significant.
> - **Test C** (consensus informativeness vs un-augmented model mean occlusion profile, n=9):
>   rho +0.250, perm p 0.512. Positive as predicted, not significant.
> - **Reading:** every directional check (A, B1, B2, C) is consistent with the second-locus
>   hypothesis, but nothing survives correction at n=40. A consistent-but-underpowered negative,
>   not a refutation. The 2.6 decision point (does a second locus become a numbered finding) is
>   NOT triggered: outcome is I, not L. Section 4.8.2 keeps its current scope. P-9 remains
>   optional; P-8 did not answer the locus question affirmatively, and it did not kill it either.
> - P-8 contributes no paired Wilcoxon tests (all Spearman-with-permutation-null, reported outside
>   the BH family). Section 4.17 not touched. No thesis file edited. Numbers:
>   `results_locus/p8_verdict.md`. Pre-registration below is unedited. P-9 and P-10 not started
>   (awaiting Enam's word).

> **P-9 RUN 3 September 2026 (four arms, both backbones per the amended §3.4).**
> **BACKBONE OUTCOME S (split vindicated). TRANSFER OUTCOME U on both backbones.**
> - **§3.2 gate:** attenuation alpha = 0 reproduces same-run occlusion row for row on all four
>   arms (max|diff| 0.0). RNG-inertness assertions around instrument_fold held every fold.
> - **§3.5a reproduction check (A1, A2 only; §4.8.2 keeps its 84.2 -> 15.0 headline, P-9 does not
>   restate it):** A1 summed occlusion cost 86.88 pp vs original 84.21 (+2.67 pp, within the 3.0 pp
>   gate); A2 13.40 pp vs 15.03 (-1.63 pp, within gate). 40-fold F1: A1 0.7611 vs 0.7684, A2 0.8247
>   vs 0.8251. **A1/A2 reproduction spread (the yardstick): 2.67 pp on summed cost; the reduction
>   factor moved 5.60x -> 6.48x on a pure same-config re-run, i.e. 0.88x.**
> - **A3/A4 first instrumented measurement on resnet_se (new numbers, never reproductions):**
>   A3 (resnet_se, none) summed occlusion cost 81.77 pp, F1 0.7746; A4 (resnet_se, chandrop 0.2,
>   the model of record) 13.36 pp, F1 0.8352. resnet_se reduction 6.12x [95% 4.94, 7.85].
> - **Backbone question = S.** resnet 6.48x [5.11, 8.54] vs resnet_se 6.12x [4.94, 7.85]:
>   |difference| 0.36x against the 0.88x that a re-run alone moved it, bootstrap CIs overlap, and
>   the per-subject paired backbone difference in absolute CD-induced reduction is -5.07 pp
>   (95% BCa [-13.76, +2.62], p = 0.35), not distinguishable from zero. The occlusion mechanism
>   transfers across backbones, measured not assumed. **§4.8.2 may quote either figure provided it
>   names the backbone; the G1 warrant is retrospectively supported on a mechanism quantity.** The
>   B1 manifest work still needs to fix the three passages that state the number without naming the
>   backbone, but the number itself does not have to move.
> - **Transfer question = U on both backbones.** Augmented-arm censoring is 58% (resnet) / 56%
>   (resnet_se) at the 2.0 pp criterion and still 37% / 38% at the pre-committed 1.0 pp fallback,
>   above the 30% limit: the robust models tolerate near-total single-channel attenuation for over
>   a third of channels, so the attenuation measure cannot resolve transfer either. A third
>   principled measure has now failed on the §4.8.2 transfer question (after P-2 rank agreement and
>   P-1 occlusion coupling). §4.8.2 keeps its wording; §5.13 records the third attempt. Per §3.7,
>   do not commission a fourth.
> - **Model vs data (P-8 Test A benchmark +0.167):** un-augmented threshold-ordering agreement is
>   +0.150 (resnet) and +0.099 (resnet_se), both bracketing +0.167 -> the model tracks the
>   redundancy the data offers rather than over-committing. (P-2's occlusion-ordering agreement on
>   the resnet arm family was +0.305, on a measure that saturates.)
> - **Per-subject coupling: retired, underpowered by design.** P-8 needs about 151 subjects at
>   rho = -0.226 and 351 at +0.149; the binding constraint is cohort size. Not run.
> - P-9 adds no paired Wilcoxon tests to the BH family (permutation-null correlations and
>   shuffle-null comparisons, reported outside it as P-1 and P-2 were). §4.17 not edited. No thesis
>   file edited. Numbers: `results_locus/p9_verdict.md`. Pre-registration below is unedited.
>
> **P-10 RUN 3 September 2026. OUTCOME B (boundary found), SD 1.00 excluded as outcome X for that
> arm.** Mean-preserving channel dropout upward sweep on resnet_se, per-subject norm, 250 ms.
> Multiplier gates passed for SD 0.60 / 0.80 / 1.00 (realized mean 1.00, SD on target;
> p' = 0.2647 / 0.3902 / 0.5000).
> - **Curve (mean LOSO macro-F1):** SD 0.40 0.8350, SD 0.50 0.8379 (peak), SD 0.60 0.8320,
>   SD 0.80 0.7916, SD 1.00 0.7313. vs no-aug 0.7822: +5.28, +5.57, +4.98, +0.94, -5.09.
> - **SD 1.00 (p' = 0.5) collapses 5.09 pp below no augmentation, outcome X for that arm**, a bound
>   on the family (half the channels zeroed per sample, the rest scaled by 2), excluded from the
>   ceiling reading.
> - **On the arms that trained: peak at SD 0.50, then a Holm-significant fall of 4.63 pp at
>   SD 0.80** (Holm p 3.9e-09, d 1.18; adjacent SD 0.60 to 0.80 fall 4.05 pp, p 2.7e-09, d 1.03).
>   SD 0.50 to 0.60 is flat (+0.58 pp, p 0.28). The curve is flat through the operating range and
>   turns down past the peak: **a genuine over-invariance boundary near SD 0.80**, at a dose the
>   unnormalized rate sweep never reached because its own mask artifact (P-7) masked it.
> - **Reading:** the Section 4.13.2 over-alignment parallel is supported on the mean-preserving
>   family, and P-1's structural reading (outcome S) holds after all, relocated to a higher dose.
>   Caveat (section 6): two or three usable points past the peak is not a dose-response curve, and
>   the boundary sits where the perturbation approaches the p' = 0.5 degeneracy, so "over-alignment
>   destroys class structure" versus "the perturbation becomes too destructive to train against"
>   is not fully separable with these points.
> - **Section 4.6 decision point (for Enam):** P-1's verdict text must be rewritten whichever way
>   P-10 fell. Draft replacement wording is in `results_locus/p10_verdict.md` and the report; the
>   thesis edit itself is Enam's.
> - New paired Wilcoxon tests for the BH family: post-peak contrasts on the trained arms
>   (SD 0.50 vs 0.60 raw p 0.2826; SD 0.50 vs 0.80 raw p 1.944e-09; SD 0.50 vs 1.00 raw p
>   3.638e-12). §4.17 not edited. No thesis file edited. Numbers: `results_locus/p10_verdict.md`,
>   `p10_curve.csv`, `p10_ceiling_curve.png`. Pre-registration below is unedited.

**Status:** ready to run. Written 3 September 2026.
**Executor:** Claude Code, on Enam's machine.
**Cost: P-8 needs NO GPU** and is pure re-analysis of files on disk. P-9 is 2 runs, about 80 minutes. P-10 is
3 runs, about 2.5 hours.
**Run P-8 first and report before starting either of the others.** P-8 may answer the locus question on its own,
in which case P-9 becomes optional rather than necessary.
**Owner decision points:** two, at 2.6 and 4.6.

---

## 0. Read this first

1. **The Python interpreter is `06_Code/.venv/Scripts/python.exe`.** The `jobs_*.txt` files are stale.
2. **No CNN checkpoints exist anywhere.** `results_*/` holds `cnn_arch_subjectwise.csv`, `cnn_arch_summary.csv`
   and sometimes `instr/`, and nothing else. Any probe of a trained model requires re-running the arm. This is why
   P-9 costs two runs rather than being inference-only.
3. **Re-running an arm gives a different model.** Run-to-run SD is 0.47 to 0.59 pp at the 40-fold mean and much
   more per fold. **P-9's new profiles must never be cross-compared with the existing `occlusion.csv` numbers in
   Section 4.8.2.** Both arms are re-run together and everything in P-9 is internal to that pair.
4. **Write a cell for the null, the reversal and the broken measure.** Six pre-registered grids in this project
   have had holes. P-9 in particular re-runs a question two measures have already failed on, and its grid says so.

---

## 1. Why this exists

Section 4.13.1 established that the between-subject shift is a **location** shift in the marginals, and that
per-subject z-scoring removes it. Two implementation facts sharpen what that leaves behind:

- For the classical pipeline, `per_subject_zscore` standardises **`axis=0`, per feature column**. Every one of the
  72 Freq-72 columns is centred and scaled within each subject.
- For the deep pipeline, `per_subject_zscore_3d` standardises **per channel**, over windows and time, giving
  shape `(1, C, 1)`. Every electrode is centred and scaled within each subject.

So after normalization, per-channel and per-feature location and scale differences between people **do not
exist**. And channel dropout still adds +5.7 pp on top of that.

**Whatever channel dropout treats, it therefore cannot be a per-channel moment.** The hypothesis this plan tests
is that what remains is **which electrodes carry the discriminative signal, and that this differs between
people**. Normalization aligns each channel's marginal; it has no mechanism for aligning a channel's
informativeness, because informativeness is a property of the channel-to-label relationship rather than of the
marginal. If that hypothesis holds, the thesis has a **second, distinct locus of the cross-subject problem**, and
the two interventions become complementary halves of one account rather than two separate wins. Their additivity
is the existing evidence for that; P-8 is the direct test.

---

## 2. Stage P-8: is channel informativeness subject-specific? NO GPU.

### 2.1 Inputs, all on disk

`features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz` and its `_meta.csv`
(columns include `subject`, `y_int`). Freq-72 is 8 features per channel by 9 channels; confirm the column-to-
channel mapping from `extract_features.py` and **report it**, because everything below depends on it.

Also `results_g3_noaug_instr/instr/occlusion.csv` for test C.

### 2.2 The matrix

Apply per-subject z-scoring exactly as `per_subject_zscore` does. Then for each subject and each of the 9
channels, fit a 4-class LDA on **that channel's 8 features only**, using stratified 5-fold cross-validation
**within that subject**, and record macro-F1. That is 40 x 9 x 5 = 1800 small fits and takes seconds.

The result is a 40 by 9 matrix of within-subject per-channel discriminative power.

**One thing not to do.** Do not compute this matrix before and after normalization and compare. LDA is invariant
to invertible linear maps and per-column standardisation is a diagonal one, so the two would be identical by
construction and the comparison is vacuous. Compute on the normalized features, which is the operating condition,
and say why the before-and-after contrast was not run.

### 2.3 Test A: is the ordering subject-specific?

Mean pairwise Spearman of channel rankings across all 780 subject pairs, against a permutation null that shuffles
channel labels within each subject, 10,000 draws, seed 42. Report the agreement, the null distribution and the
consensus ranking with the muscle names from Table 3.1 (TFL, RF, VM, SMB, Upper TA, Lower TA, Lateral GC,
Medial GC, SOL).

### 2.4 Test B: does deviation from consensus predict difficulty and benefit? **This is the primary test.**

For each subject, compute a **consensus agreement score**: the Spearman between that subject's channel ranking
and the consensus ranking of the other 39, leave-one-out so the subject never contributes to its own consensus.
Then correlate that score against:

- **B1**, the subject's LOSO macro-F1 under the un-augmented ResNet-SE (`results_cnn_arch` no-augmentation arm,
  0.7822 mean). The account predicts a **positive** correlation: subjects whose channel ordering is unusual are
  harder.
- **B2**, the subject's channel-dropout gain. The account predicts a **negative** correlation: subjects whose
  ordering is unusual gain more from an augmentation that stops the model committing to any ordering.

Spearman with a 10,000-draw permutation null at seed 42, Holm across the two.

**This is the direct analogue of Section 4.5**, where distance from the pooled training distribution predicts
difficulty under global normalization. It is also the question P-1b and P-1c tried to answer and could not,
because they measured it on the augmented model's occlusion profile, which saturates. **This measures it on the
features, where nothing saturates.** Say that explicitly in the verdict.

### 2.5 Test C: does the model learn what the data offers?

Correlate the consensus informativeness ranking against the un-augmented model's mean occlusion profile from
`results_g3_noaug_instr/instr/occlusion.csv`. If the model's reliance tracks the data's informativeness, the
account is coherent end to end. Report as a single Spearman with a permutation null, beside the family.

### 2.6 Pre-registered grid, and the decision point

Holm across B1 and B2. Test A and Test C are reported beside the family.

| Outcome | Condition | Reading |
|---|---|---|
| **L. A second locus** | A shows agreement above the null but well below ceiling, **and** B1 or B2 survives correction with the predicted sign | Channel informativeness is partly shared and partly subject-specific, and the subject-specific part predicts difficulty or benefit. **This is the result the hypothesis predicts** and it gives channel dropout its own "where the problem lives" statement. |
| **U. Universal ordering** | A shows agreement at or near ceiling and neither B test survives | Every subject carries the signal in the same electrodes. The hypothesis is wrong, channel dropout is not treating an informativeness mismatch, and the account in Section 4.8.2 must be narrowed to "it prevents single-channel commitment" without claiming that commitment is subject-specific. **A clean negative and a useful one.** |
| **I. Idiosyncratic but inconsequential** | A shows low agreement but neither B test survives | Orderings differ between people and that difference does not predict who is hard or who benefits. Report both, claim neither. |
| **R. Reversal** | A B test survives with the sign opposite to the account | Report and escalate. Do not force a reading. |
| **X. Measure failed** | Per-channel LDA macro-F1 is at chance for most channels, or the consensus score has no spread | The metric cannot resolve informativeness at this granularity. Report and stop. |

**Decision point, for Enam:** if the outcome is L, whether that becomes a numbered finding, a Section 4.8.2
subsection, or a paragraph in Section 5 is his call. Report and stop.

---

## 3. Stage P-9: graded attenuation, on both backbones. 4 RUNS, about 3.5 to 4.5 hours.

### 3.1 Why

Full occlusion is a one-bit probe. On the augmented model it saturates: cost falls from 84.2 pp to 15.0 pp,
a quarter of entries go negative, and every statistic built on it loses resolution. That single flaw broke
P-1b, P-1c and P-2. Attenuating a channel progressively and finding where the model finally responds keeps
resolution on a robust model, because the dial is turned until there is a response.

### 3.2 The change

Extend `instrument_fold` in `run_cnn_arch_loso.py` to sweep an attenuation factor per channel,
alpha in {1.0, 0.75, 0.5, 0.25, 0.0}, multiplying that channel by alpha and recording macro-F1 at each. Write
`attenuation.csv` with columns `subject, channel, alpha, f1, drop_pp`.

**alpha = 0 is exactly the existing occlusion, so it is a built-in gate**: the alpha = 0 slice must reproduce the
same-run `occlusion.csv` row for row.

The existing instrumentation already carries RNG-inertness assertions immediately around it
(`assert torch.equal(torch.get_rng_state(), _rt)`). Keep them and confirm they still pass. Instrumentation is
inference-only and runs after the fold's row is written, so the extension costs little beyond the training.

### 3.3 The derived quantity

Per subject and channel, the **reliance threshold**: the alpha at which macro-F1 has fallen 2.0 pp below the
unattenuated fold value, by linear interpolation between bracketing alphas. Lower threshold means the model
tolerates more attenuation, so less reliance. Bounded in [0, 1] and defined for robust and fragile models alike.

### 3.4 Arms: four, on both backbones. Amended 3 September after Enam's decision.

The mechanism programme is split across two backbones. `resnet` (SE-free) carries G1, G3, W-3 and W-4; `resnet_se`
carries G2, P-5, P-6 and P-7. The split is coherent, elimination work on the simpler backbone and dose work on the
model of record, **but it rests on G1's null interaction of -0.78 pp at p = 0.221, whose interval runs
[-2.05, +0.55]**. Four experiments were sited on that null. Worse, **Section 4.8.2's headline 84.2 pp to 15.0 pp
is measured on the SE-free backbone while the deep model of record is `resnet_se`+CD at 0.8395**, and three
passages in the thesis state that number without naming the backbone.

Running all four arms settles it by measurement instead of by inference.

| Arm | backbone | augmentation | prior instrumented run |
|---|---|---|---|
| A1 | `resnet` | none | `results_g3_noaug_instr` |
| A2 | `resnet` | chandrop p = 0.2 | `results_cd_resnet_nose_chandrop` |
| A3 | `resnet_se` | none | **none** |
| A4 | `resnet_se` | chandrop p = 0.2 | **none. This is the model of record.** |

`--norm-mode per_subject`, 250 ms, seed 42, everything else default, all four with the extended instrumentation.

### 3.5 What the four arms buy, and why each is needed

1. **The headline on the model of record.** A3 and A4 give the occlusion and attenuation profile of the network
   the thesis actually ships. Report as a **new measurement**, never as a reproduction, because no prior
   instrumented `resnet_se` run exists.
2. **A reproduction check.** A1 and A2 are a second measurement of the published 84.2 pp and 15.0 pp.
3. **The noise floor, and this is the point of pairing them.** The A1/A2 reproduction spread **is the yardstick**
   for judging whether any `resnet` to `resnet_se` difference is real. There is no other estimate of how
   reproducible the occlusion statistic is, and without one a cross-backbone difference cannot be interpreted.
   **State the A1/A2 spread before interpreting any cross-backbone comparison.**
4. **The cross-backbone test, which is the G1 warrant made empirical.** Compare A1-vs-A2 against A3-vs-A4: the
   summed occlusion cost per arm, the reduction factor with its interval, and the per-subject paired difference
   between backbones. If the reduction on `resnet_se` matches the SE-free figure to within the A1/A2 spread, the
   split is vindicated on a mechanism quantity rather than on a null about accuracy, and the thesis can say so.
   If it differs by more, that is a finding and Section 4.8.2's headline belongs to `resnet_se`.
5. **Transfer**, the P-2 question, computed **within each backbone**, with the shrink control P-2 used and
   **P-8's Test A agreement of +0.167 as the benchmark**: near it, the model tracks what the data offers; well
   above it, the model over-commits relative to the redundancy available.
6. **Per-subject coupling: report as underpowered by design and do not run it as if it could succeed.** P-8
   measured the same effect on a non-saturating quantity and found it needs about 151 subjects at rho = -0.226
   and 351 at rho = +0.149. The binding constraint is cohort size, not measurement.

### 3.5a The reproduction trap. Settle this BEFORE the run, not after seeing the numbers.

P-9's A1 and A2 re-run the two `resnet` arms that produced **Section 4.8.2's headline result, the 84.2 pp to
15.0 pp occlusion cost and its 5.6-fold reduction**. A3 and A4 are new and have nothing to reproduce. Run-to-run variation is 0.47 to 0.59 pp at the 40-fold mean and far
more per fold, so the new alpha = 0 slice **will not reproduce those figures exactly**.

That is not a problem unless it is handled badly, and this project has twice been damaged by exactly this
situation: the thesis carried both 0.787 and 0.772 for one quantity until Tier 1a, and G2 had to record a
reproduction gate when its fresh p = 0.2 arm came back 0.19 pp below the published one.

**Fixed in advance, whatever the numbers turn out to be:**

1. **Section 4.8.2 continues to quote the original 84.2 pp and 15.0 pp.** P-9 does not restate the headline.
2. The verdict reports P-9's own alpha = 0 occlusion costs **as a reproduction check**, alongside the originals,
   with the difference stated plainly. This is the treatment R-1 gave the CNN reproducibility question and it is
   the house pattern.
3. **A reproduction gate of 3.0 pp on the summed cost per arm.** Inside it, note the spread and carry on. Outside
   it, stop and report, because a gap that large is not run-to-run noise and something else has moved.
4. **Every P-9 statistic is computed within the new pair only.** Nothing in P-9 is compared against a number from
   the original run except in the reproduction check itself.

Write the reproduction check into the verdict **before** interpreting transfer, so the comparison is on record
regardless of which way it falls.

### 3.6 The validity guard, which is not optional

**Censoring.** If a channel never reaches a 2.0 pp drop even at alpha = 0, its threshold is undefined. Report the
**censoring rate per arm**. If more than 30% of (subject, channel) pairs are censored on the augmented arm, the
measure is censored rather than saturated and the comparison must either use a smaller drop criterion, which must
be chosen and stated before looking at the outcome, or be reported as outcome X.

### 3.7 Grid

Two questions, two letters. Report both.

**Backbone question (new, primary for the write-up):**

| Outcome | Condition | Reading |
|---|---|---|
| **S. Split vindicated** | The `resnet_se` occlusion reduction matches the SE-free one to within the A1/A2 reproduction spread | The mechanism findings transfer across backbones, measured rather than assumed. **Section 4.8.2 may keep quoting either figure provided it names the backbone**, and the G1 warrant is retrospectively supported on a mechanism quantity. |
| **D. Backbone matters** | The two reductions differ by more than the A1/A2 spread | **Section 4.8.2's headline belongs to `resnet_se`**, the number of record changes to the A3/A4 figure, and the SE-free result becomes the companion. Report both, say which model each is about, and say plainly that the G1 null did not license the transfer. |
| **X. No yardstick** | A1/A2 fail the 3.0 pp reproduction gate | The occlusion statistic is less reproducible than assumed, so no cross-backbone comparison may be read. That is itself worth reporting and it bounds every occlusion claim in the thesis. |

**Transfer question, per backbone:**

| Outcome | Condition | Reading |
|---|---|---|
| **T. Transfers** | Augmented threshold agreement above null and not below the un-augmented arm's | **Closes the Section 4.8.2 question two measures could not.** |
| **N. Does not transfer** | Above null and significantly below the un-augmented arm's | Flatter and more idiosyncratic. Also closes it, negatively. |
| **U. Still unresolved** | Agreement indistinguishable from null, or censoring above 30% | **A third measure has failed.** Report it as a finding about the measurement problem, and **do not commission a fourth**. Three principled attempts is thorough; four reads as fishing. |
| **X. Gate failed** | alpha = 0 does not reproduce same-run occlusion, or inertness fails | Stop. Instrumentation problem, not science. |

**If the two backbones disagree on the transfer letter, report both and force neither.** That would itself be
evidence bearing on the backbone question.

---

## 4. Stage P-10: does a genuine ceiling exist? 3 RUNS, about 2.5 hours.

### 4.1 Why, and what P-7 did to P-1

P-1 concluded that more invariance keeps buying robustness and **starts costing** accuracy, and was written as the
over-alignment analogue of Section 4.13.2. P-7 then showed the cost at p = 0.5 is the unnormalized mask, not the
invariance: the mean-preserving arm at the same variance sits at 0.8379, which is p = 0.2's value.

Remove the artifact and the sequence is 83.1, 83.8, 83.7, 83.8. **That is diminishing returns, not
over-alignment.** Section 4.13.2's force is that going further actively destroys class structure; nothing here
destroys anything, it merely stops helping. **P-1 cannot be written as an over-alignment analogue as it stands**,
and this stage decides whether it can be written as one at all.

### 4.2 The arms, and why not gain jitter

**Gain jitter cannot sweep upward.** Its multiplier is Uniform(1 - a, 1 + a) with a = sd * sqrt(3), so it goes
negative above sd = 1/sqrt(3) = 0.577, flipping channel sign. That is a qualitatively different perturbation and
would confound the curve. Do not extend the gain-jitter arm past 0.55.

Mean-preserving channel dropout has multiplier 0 or 1/(1 - p'), which is non-negative at every rate, so the
upward sweep runs on it. With p' = SD^2 / (1 + SD^2):

| target SD | p' | status |
|---|---|---|
| 0.40 | 0.1379 | **exists**, 0.8350 |
| 0.50 | 0.2000 | **exists**, 0.8379 |
| 0.60 | 0.2647 | **run** |
| 0.80 | 0.3902 | **run** |
| 1.00 | 0.5000 | **run** |

`resnet_se`, per-subject normalization, 250 ms, everything else default. Run the multiplier gate on each and
report realized mean and SD.

### 4.3 Grid

Trainability floor as before: an arm more than 3.0 pp below the no-augmentation baseline of 0.7822 is outcome X
for that arm.

| Outcome | Condition | Reading |
|---|---|---|
| **P. Plateau, no ceiling** | F1 flat or rising through SD 1.00, no fall exceeding 2.0 pp | **No over-invariance boundary exists in this range.** P-1 is written as diminishing returns and the Section 4.13.2 parallel is explicitly not supported, which is what G2 originally concluded. Honest, and it retires a claim rather than making one. |
| **B. Boundary found** | A peak followed by a fall exceeding 2.0 pp with a Holm-significant paired contrast | A genuine over-invariance boundary exists, at a dose the unnormalized form never reached because its own artifact masked it. **Report where.** The Section 4.13.2 parallel is then supported on the mean-preserving family, and P-1's structural reading holds after all. |
| **N. Non-monotone noise** | Movement within the run-to-run band with no coherent shape | Report the curve and claim no shape, which is what G2 did with the rate sweep. |
| **X. Collapse** | An arm fails the trainability floor | At p' = 0.5 half the channels are zeroed each sample and the rest doubled; failure there is plausible and is a bound on the family, not a bug. Report it as such. |

### 4.6 Decision point

Whichever way P-10 falls, **P-1's verdict text has to be rewritten**, because it currently claims a boundary that
P-7 has already explained away. Draft the replacement wording and report it. Do not edit any thesis file.

---

## 5. Scope, for all three stages

No thesis file is edited. Section 4.17 is not touched; report new paired tests with raw p-values and the family
is recomputed in one pass afterwards. The deep model of record stays the channel-dropout `resnet_se` at 0.840
whatever any stage finds. Correlations against permutation nulls are reported outside the Benjamini-Hochberg
family, as P-1 and P-2 correctly did.

Outputs into `results_locus/`: one verdict markdown and one outcome JSON per stage, plus
`p8_channel_informativeness.csv` (the 40 by 9 matrix and the consensus scores), `p9_attenuation_thresholds.csv` (all four arms, backbone-labelled)
and `p10_curve.csv`. Scripts named `p8_informativeness_stats.py`, `p9_attenuation_stats.py`,
`p10_ceiling_stats.py`. Add a status header to the top of this file per stage, leaving the pre-registration
below it unedited.

## 6. What to report back

1. **P-8 first, then stop.** The column-to-channel mapping, the consensus ranking in muscle names, tests A, B1,
   B2 and C with permutation nulls, and the outcome letter.
2. Then, on Enam's word, P-9 and P-10 with their gates, censoring rates, curves and outcome letters.
3. New paired tests and raw p-values for the family, and confirmation that no thesis file was edited.
