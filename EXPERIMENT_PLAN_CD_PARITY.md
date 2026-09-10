# Experiment plan: the channel-dropout parity programme (P-1 to P-4)

> **RUN 3 September 2026. P-0, P-1, P-2, P-3, P-5, P-6 executed. P-4 NOT run (recommended against).**
> - **P-0 literature:** four sources verified. Srivastava et al. 2014 read in full (Table 10
>   confirmed: MNIST 1.08 to 0.95, CIFAR-10 12.6 to 12.5). **Pereira et al. 2024 is ICASSP 2024,
>   a peer-reviewed conference paper, not a preprint;** cite as such. ModDrop and Zhang et al. 2026
>   verified by abstract and record only (MDPI full text 403). Caution: Zhang et al. uses "channel
>   dropout" for a test-time fault. Adjacent work found (MAST 2024 sensor-wise masking; EEG
>   channel-dropout augmentation). Nothing written to the thesis. Draft sentences and placements:
>   `results_parity/p0_literature.md`.
> - **P-1 OUTCOME S (separated at the boundary).** Occlusion cost falls monotonically with rate
>   (21.37, 12.10, 10.26, 7.42 pp; Page L p=3.0e-10, Spearman rho=-1.00, both survive Holm) while
>   F1 plateaus then falls (+0.65, -0.03, -1.68 pp). Per-subject correlations do not survive Holm
>   and carry the sign opposite the mechanism account but not significantly: P-1b rho=+0.305
>   perm p=0.056; P-1c rho=-0.282 perm p=0.080. Over-alignment analogue of 4.13.2, structural not
>   mechanistic. `results_parity/p1_verdict.md`.
> - **P-2 OUTCOME X (ranks not scale-free in practice).** Rank agreement baseline +0.305 vs
>   channel dropout +0.142; paired Wilcoxon p=1.9e-6; CD arm IS separable from its shuffle null
>   (p=1e-4), but the shrink control reproduces 142 percent of the deficit on simulated data with
>   no real ordering change. Report as a measurement finding, not a transfer answer; 5.13 records a
>   second measure failing on the transfer question. `results_parity/p2_verdict.md`.
> - **P-3 OUTCOME R (replicates).** ENABL3S, per-subject norm: channel dropout vs no aug +7.17 pp,
>   d=1.72, Holm p=0.0039, 10/0 subjects, 10 paired. Secondary (global norm) +2.88 pp, d=0.58,
>   Holm p=0.28. Goes into 4.12 beside the normalization replication. `results_parity/p3_verdict.md`.
> - **P-5 OUTCOME M (matches).** subset vs channel dropout -0.35 pp, Holm p=0.34, d=-0.13;
>   subset vs no aug +5.38 pp, d=1.15. The published alternative (Pereira subset training, 36-subset
>   vocabulary) does the same work. Model of record unchanged. `results_parity/p5_verdict.md`.
> - **P-6 OUTCOME V (variance is what matters).** Arms on resnet_se: no aug 0.7822, channel dropout
>   0.8395, mean-preserving chandrop 0.8350, gain jitter sd0.40 0.8477. C1 (activation shift)
>   +0.45 pp Holm p=0.30; C2 (form) -1.28 pp Holm p=0.026 but below the 2.0 pp floor. Operative
>   property is injected per-channel multiplicative variance. **Extension: DIVERGE at high SD.**
>   Gain-jitter and channel-dropout curves superimpose at multiplicative SD 0.30 (+0.59 pp p=0.40)
>   and 0.40 (+1.01 pp Holm p=0.058), then separate at SD 0.50 (+2.81 pp, Holm p=6.8e-7, d=0.95):
>   heavy Bernoulli zeroing is materially worse than continuous jitter of the same variance, which
>   localises P-1's boundary to the zeroing form at high rate. `results_parity/p6_verdict.md`,
>   `p6_extension_verdict.md`.
> - **Code:** `subset` and `mpchandrop` modes added to `augment_batch` behind guards; inertness
>   PASS on all six existing modes (byte-identical output and RNG state, both import sites) and on
>   resnet/resnet_se init (`p5p6_inertness.py`). mpchandrop p' derived from requested SD
>   (0.40 -> 0.1379), multiplier gate PASS (`p6_multiplier_gate.py`).
> - **FDR family:** 12 new paired Wilcoxon tests recommended (13 listed; the SD 0.40 extension
>   contrast is a cross-check). Raw p-values in `results_parity/PARITY_REPORT.md`. **Section 4.17
>   NOT edited.** P-1 correlations and the P-2 randomization/shuffle-null checks are reported
>   separately, not folded in, per section 6.3.
> - **P-4 recommendation: not worth the GPU hours.** W-5 already shows the gain flat from 231k to
>   535k params; the likely P-4 outcome is N, which 4.8.1 can already state, and a 59k skip-free
>   net has real odds of failing the trainability gate (outcome X, no information). Channel dropout
>   is already ahead of normalization on the specificity row. Not started.
> - Nothing downstream regenerated. Deep model of record stays channel-dropout resnet_se at 0.840.
>   No thesis file edited. Section 6.1 framing decision is Enam's. Pre-registration below is unedited.

**Status:** ready to run. Written 2 September 2026.
**Executor:** Claude Code, on Enam's machine.
**Cost: stages 1 to 3 need NO GPU.** They are paired tests and correlations on result files that already exist and were never analysed. Budget 2 to 3 hours of analysis and writing. Stage 4 is 4 GPU runs, about 2 to 3.5 hours, and is **conditional**: do not start it until stages 1 to 3 are reported.
**Owner decision points:** three, at 1.6, 5.5 and 6.1. Do not resolve any of them yourself.

---

## 0. Read this first

1. **The Python interpreter is `06_Code/.venv/Scripts/python.exe`.** The `jobs_*.txt` files point at a venv removed in the August restructure and are stale.
2. **Nothing in stages 1 to 3 trains anything.** If you find yourself launching a 40-fold run before stage 4, stop; you have misread the plan.
3. **Two backbones are in play and must never be pooled.** `results_g3_noaug_instr` and `results_cd_resnet_nose_chandrop` are `--arch resnet`, SE-free. The four `results_cd_rate_p*` arms are `--arch resnet_se`. Occlusion costs are not comparable across the two. Every analysis below states which family it uses.
4. **Write a cell for the null, the reversal and the broken measure.** Five pre-registered grids in this project have now had holes: W-3, G3, G2, W-1 and the fall-through in W-1's scorer. Stage 2 in particular is a re-run of a measure that has already broken once. Its grid has an explicit UNANSWERED cell and that cell is a legitimate result, not a failure.

---

## 1. Why this programme exists

### 1.1 The claim it protects

The thesis is an investigation of how the data-centric interventions that closed the cross-subject gap actually work. Two interventions closed it: per-subject normalization and channel dropout. Normalization has been investigated to a standard channel dropout has not, and the asymmetry is now visible in the document.

**The point of this programme is methodological parity, not matching results.** A negative or ambiguous answer for channel dropout is an acceptable outcome and should be written up as one. What is not acceptable is one intervention investigated four ways and the other asserted.

### 1.2 The parity map, which is what this programme closes

| Element of the investigation | Per-subject normalization | Channel dropout, before this programme |
|---|---|---|
| Effect established under LOSO | §4.6, four conditions, three models | §4.8, four augmentations, Table 4.10 |
| Mechanism measured | §4.13.1, location versus spread, subject probe 0.777 to 0.043 | §4.8.2, occlusion cost 84.2 to 15.0 pp |
| **Mechanism linked to outcome** | §4.13.2 alignment ladder: the operator that removes subject structure is the one that raises F1, and going further removes class structure faster | **absent.** G2 swept the rate and looked only at F1 |
| **Boundary, where more becomes worse** | §4.13.2 over-alignment, class silhouette turns negative | **absent.** G2 read flat on F1 alone |
| Head to head against alternatives | CORAL, Deep CORAL, AdaBN, plus a regularizer sweep | Gaussian noise, time masking, gain jitter |
| Specificity, what it is not | mostly implicit | SE, skips, removal, depth, capacity: five eliminations |
| **External replication** | §4.12, ENABL3S, three models, +10 to +17 pp | **absent** |
| Per-subject predictor of who benefits | §4.5, distance from the pooled distribution predicts difficulty under global norm and stops predicting after normalization | **absent** |
| Transfer between subjects | n/a | **asked and unanswered**, measure collapsed |

Stages 1 to 3 close the four bold rows without a single GPU-hour. Stage 4 addresses the one remaining architectural question.

### 1.3 What is at stake in the framing

If stage 1 finds a coupling and stage 3 replicates, channel dropout has an effect, a measured mechanism, a dose-response linking the two, a comparison against alternatives, and independent replication. That is the same evidentiary shape the normalization finding has, and the thesis could then describe itself as investigating both successful interventions rather than one.

**A note on what counts as success here, because it governs every grid below.** The object of study is the class of channel-level perturbation, not channel dropout's score. That per-subject normalization turned out to beat every alternative tried was a coincidence of how the results fell, and the finding that made it into the headline was the over-alignment result, which is a statement about *what matters*, not about *which method won*. The same standard applies here. If a family member outperforms channel dropout, that is a result to be characterized, since what distinguishes it is evidence about the operative property. If nothing does, that is equally informative. A null is a result. **No grid in this plan should be read as hoping for a particular member to win, and no write-up produced from it should defend one.** **That is Enam's call and nobody else's**, and it is decision point 6.1. If either stage comes back null, the honest framing is that channel dropout's effect is established and measured while the path from mechanism to benefit is not, which is still a substantially stronger position than the document holds today.

---

## 1.4 Stage P-0: the literature grounding. NO GPU, and do this first.

**The largest parity gap is not experimental.** Section 2.4.4, the thesis's entire treatment of data augmentation, cites one paper (Hu et al., 2019) and says nothing about dropout in any form. Channel dropout, which is responsible for +5.7 pp and for the deep model of record, has **no literature grounding anywhere in the thesis.** The normalization side by contrast carries CORAL, Deep CORAL, AdaBN and the whole of Sections 2.2 and 2.4.3. An examiner who works on augmentation will notice this before they notice anything in Chapter 4.

Four sources were verified on 3 September 2026 and are the starting set. **Verify each independently before citing it. Do not cite anything you cannot retrieve in full**, and if a source can only be partially retrieved, cite it only for what its title and abstract state. That rule exists because a source in this thesis was once cited beyond what could be verified, and it is recorded in the handoff's correction log.

| Source | Why it matters | Where it belongs |
|---|---|---|
| **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR, 15, 1929-1958.** Section 10, Table 10, reports that multiplying activations by Gaussian noise performs comparably or slightly better than Bernoulli dropout (MNIST 0.95% against 1.08%, CIFAR-10 12.5% against 12.6%). | **This is W-4's result one level up.** W-4 found gain jitter statistically indistinguishable from channel dropout at matched variance. The foundational dropout paper reports the same relationship between multiplicative noise and Bernoulli masking at the unit level. W-4 is therefore a replication of a known property in a new setting rather than an isolated ablation. | §2.4.4, and cited at §4.8.2's gain-jitter paragraph and §5.7 |
| **Neverova, N., Wolf, C., Taylor, G., & Nebout, F. (2016). ModDrop: adaptive multi-modal gesture recognition. IEEE TPAMI, 38(8), 1692-1706.** Drops entire modalities during training so the network tolerates missing signals at test time. | The conceptual parent of channel dropout: structured dropout at the level of an input stream rather than a unit, motivated by sensor availability. | §2.4.4 |
| **Pereira, J., Chalatsis, D., Hodossy, B., & Farina, D. (2024). Tackling electrode shift in gesture recognition with HD-EMG electrode subsets. arXiv:2401.02773.** Trains on collections of input channel subsets as augmentation, reporting +6.9% intersession accuracy under electrode displacement with 16 times fewer input channels. | **The direct sEMG precedent for the intervention.** Same idea, different framing (subset selection rather than random masking) and a different target (intersession electrode shift rather than cross-subject transfer). Farina is already in the reference list. **Confirm its peer-reviewed status; the retrieved copy carries an IEEE copyright notice but reads as a preprint.** If it is a preprint, say so in the citation. | §2.4.4 and §5.7 |
| **Zhang, C., Zhou, D., Fang, Y., Gao, D., & Ju, Z. (2026). Assessing the resilience of sEMG classifiers to sensor malfunction and signal saturation. Sensors, 26(8), 2386.** Calls itself the first systematic robustness map of a conventional sEMG pipeline under single-sensor failure, using test-time channel removal on classical pipelines across nine subjects. | **The direct sEMG precedent for the measurement.** It establishes that single-channel-loss profiling is an accepted way to characterize an sEMG pipeline, which means §4.8.2's occlusion protocol is not idiosyncratic. Its scope is complementary and does not overlap: it measures classical pipelines at test time and applies no training-time augmentation, whereas §4.8.2 measures a deep pipeline over 40 subjects under LOSO and reports an augmentation that collapses the cost 5.6-fold. | §2.4.4 and §4.8.2 |

### 1.4.1 What to do

1. Retrieve and read all four. Record for each whether it was retrieved in full.
2. Search for anything else in this space and report it, in particular training-time channel or electrode dropout in sEMG or EEG, and any prior report of an occlusion or channel-ablation profile on a deep sEMG model. Open access, and preferably within five years, except for foundational work such as Srivastava.
3. **Do not write any of it into the thesis.** Report where each belongs and draft the sentences separately for Enam to review. Section 2.4.4 is a literature section in a chapter that has been voice-audited, and new prose there goes through the same pass as everything else.

### 1.4.2 The one uncomfortable implication, which must be stated and not hidden

If Srivastava et al. already showed at the unit level that multiplicative noise matches Bernoulli masking, an examiner may ask whether W-4 was predictable. **The honest answer is that it was worth testing anyway and the write-up should say so**: Section 5.7 advanced a domain-specific mechanism, that channel dropout simulates electrode liftoff and re-siting, which is a claim about *removal* being the operative event. That claim predicts removal is special. W-4 tested it and it is not. That the result agrees with a known property of dropout at a different level strengthens it rather than diminishing it, and reporting the agreement is better than leaving the reader to notice it.

---

## 2. Stage P-1: does the mechanism explain the outcome? NO GPU.

The alignment-ladder analogue. §4.13.2 works because it plots mechanism quantities against an outcome quantity across a graded series. G2 built exactly such a series for channel dropout, recorded the mechanism quantity for every arm, and then read only the outcome.

### 2.1 Inputs, all present

| Arm | Backbone | Rate | Occlusion file | Per-subject F1 |
|---|---|---|---|---|
| `results_cd_rate_p0.1` | resnet_se | 0.1 | `instr/occlusion.csv` | `cnn_arch_subjectwise.csv` |
| `results_cd_rate_p0.2` | resnet_se | 0.2 | same | same |
| `results_cd_rate_p0.3` | resnet_se | 0.3 | same | same |
| `results_cd_rate_p0.5` | resnet_se | 0.5 | same | same |
| `results_g3_noaug_instr` | resnet | none | same | same |
| `results_cd_resnet_nose_chandrop` | resnet | 0.2 | same | same |

Each occlusion file holds 360 rows, 40 subjects by 9 channels, column `drop_pp`. Confirm that before anything else.

### 2.2 P-1a, the dose curve. Primary.

On the **resnet_se family only** (the four rate arms), compute per arm the mean total single-electrode occlusion cost and the mean LOSO macro-F1, and plot one against the other across rates 0.1, 0.2, 0.3, 0.5.

The question is whether the mechanism quantity keeps moving while the outcome quantity stops. G2 established that F1 is flat from 0.1 to 0.3 and falls 1.71 pp at 0.5. **If occlusion cost continues to fall monotonically across that same range, then past p = 0.2 the augmentation is still buying robustness and no longer buying accuracy, which is structurally the over-alignment result of §4.13.2 in a different currency.** That is the finding this stage is looking for. Test the monotonicity of occlusion cost across rate with a Page trend test or Spearman on the four arm means, and report the per-subject paired contrasts between adjacent rates.

**The absence of a p = 0 arm on resnet_se is a real limitation.** No instrumented no-augmentation resnet_se run exists. Say so; do not substitute the SE-free `results_g3_noaug_instr` as the origin of this curve.

### 2.3 P-1b, per-subject coupling. Primary.

On the **resnet family** (`results_g3_noaug_instr` against `results_cd_resnet_nose_chandrop`), for each of the 40 subjects compute the change in total occlusion cost and the change in macro-F1. Correlate them, Spearman, with a permutation null of 10,000 subject-label shuffles at seed 42.

This asks whether the subjects whose electrode reliance falls most are the subjects who gain most. A positive coupling is the individual-level version of the causal claim.

### 2.4 P-1c, does baseline reliance predict benefit? Primary.

Same pair. Correlate each subject's **un-augmented** total occlusion cost against that subject's channel-dropout gain, Spearman with the same permutation null.

This is the direct analogue of §4.5, where distance from the pooled training distribution predicts difficulty under global normalization and stops predicting after per-subject normalization. If subjects who lean hardest on single electrodes are the ones the augmentation helps most, the mechanism account gains its cleanest support.

### 2.5 Pre-registered grid for P-1

Holm correction across the three correlation tests of P-1b and P-1c and the P-1a trend test, family of four.

| Outcome | Condition | Reading |
|---|---|---|
| **L. Linked** | P-1b or P-1c significant after correction with the sign the mechanism account predicts, and P-1a monotone | The mechanism explains the benefit. Write §4.8.2 as an explanation. |
| **S. Separated at the boundary** | P-1a monotone in occlusion cost while F1 plateaus or falls | More invariance keeps buying robustness and stops buying accuracy. This is the over-alignment analogue and should be written as one, with the parallel to §4.13.2 stated as structural rather than mechanistic. |
| **C. Co-occurring only** | No correlation survives correction and P-1a shows no trend | Robustness and accuracy co-occur but do not track each other. **§4.8.2 must then say they co-occur, not that one explains the other.** This is an informative negative and is written up. |
| **R. Reversal** | A correlation survives with the sign opposite to the account | Report the numbers, do not force a reading, escalate. |
| **U. Underpowered** | The permutation null shows the correlation test cannot resolve an effect of the size at issue, or fewer than 35 subjects pair cleanly | The measure cannot answer it. Say so, as §4.8.2 already does for transfer. |

---

## 3. Stage P-2: does electrode reliance transfer between subjects? NO GPU.

§4.8.2 asks this, breaks its measure on it, and reports it unanswered. §5.13 names the fix: a measure that is scale-free by construction, such as the rank agreement of channel orderings against a permutation null. This stage builds that measure.

### 3.1 The measure

Per subject, rank the nine channels by `drop_pp`. Ranks are invariant to any positive rescaling, so the 5.6-fold magnitude collapse that destroyed the normalized measure cannot bias them the same way. Between-subject agreement is the mean pairwise Spearman correlation of rankings across all 780 subject pairs, computed separately for the un-augmented arm and the channel-dropout arm, on the **resnet family**. Compare the two with a paired test over subjects and against a permutation null that shuffles channel labels within each subject, 10,000 draws at seed 42.

### 3.2 The validity guard, which is not optional

The augmented arm's profile is near-noise: 93 of 360 of its `drop_pp` entries are negative against 14 of 360 on the baseline. Ranks of near-zero noisy values are themselves noise. **Two checks decide whether the measure has any power before its answer may be read.**

1. **Null separation.** Is the augmented arm's rank agreement distinguishable from its own permutation null at all? If it is not, the measure has no resolution on that arm and the question stays unanswered.
2. **Shrink control.** Take the baseline profiles, shrink them to the augmented arm's scale and add matched noise, exactly as the simulation in §4.8.2 does, then compute rank agreement on the simulated profiles. If rank agreement collapses under simulated shrink the way the normalized measure did, **ranks are not scale-free in practice and this stage reports that**, which is a finding about the measurement problem and worth stating.

### 3.3 Pre-registered grid for P-2

| Outcome | Condition | Reading |
|---|---|---|
| **T. Transfers** | Augmented rank agreement above its null and not below the baseline arm's | Reliance is flatter and still shared across people. Closes the §4.8.2 question positively. |
| **N. Does not transfer** | Augmented agreement above its null but significantly below the baseline's | Flatter and more idiosyncratic. A real negative, and it closes the question. |
| **U. Still unanswered** | Augmented agreement indistinguishable from its null, or the shrink control reproduces the deficit | The measure has no power here either. §4.8.2 keeps its current wording and §5.13 records that a second measure has now failed on it, which is itself worth one sentence. |
| **X. Ranks are not scale-free** | Shrink control collapses agreement on simulated data | Report as a measurement finding. Do not report a transfer answer. |

---

## 4. Stage P-3: external replication of the augmentation gain. NO GPU.

The normalization finding replicates on ENABL3S. The augmentation gain has never been tested there, and it does not need a new run: **all four arms already exist.**

### 4.1 Inputs, all present, ten subjects, matching ids

| Arm | Directory |
|---|---|
| ResNet-SE, per-subject norm, no augmentation | `results_ext_resnet_se_persubj` |
| ResNet-SE + channel dropout, per-subject norm | `results_ext_chandrop_resnet_se_persubj` |
| ResNet-SE, global norm, no augmentation | `results_ext_resnet_se_global` |
| ResNet-SE + channel dropout, global norm | `results_ext_chandrop_resnet_se_global` |

### 4.2 The tests

**Primary:** channel dropout against no augmentation under per-subject normalization, paired across the ten subjects, BCa 95% interval, paired Wilcoxon, paired Cohen's d.
**Secondary:** the same contrast under global normalization.

Holm across the family of two. **At n = 10 the effect size is the primary evidence and the p-value the secondary**, which is the convention §4.12 already uses and states for the RF replication. Follow it and say so.

Confirm the subject id sets match across arms before pairing, and report the number paired.

### 4.3 Pre-registered grid for P-3

| Outcome | Condition | Reading |
|---|---|---|
| **R. Replicates** | Primary contrast positive, Holm p < 0.05, d ≥ 0.5 | The augmentation gain is not SIAT-specific. Goes into §4.12 beside the normalization replication. |
| **W. Weakly replicates** | Positive with d ≥ 0.5 but Holm p ≥ 0.05 | Consistent in direction and size at n = 10 where power is low. Report as such, leaning on the effect size, exactly as the RF replication is reported. |
| **F. Fails to replicate** | Point estimate at or below zero, or d < 0.2 | A genuine negative and an important one. The augmentation would then be shown to be dataset-specific in a way normalization is not, and **that belongs in §5.12 as a limitation**, not buried. |
| **X. Not comparable** | Subject ids do not pair, or the arms differ in any flag other than augmentation | Stop and report. Do not force a pairing. |

---

## 5. Stage P-4: is there a capacity threshold? GPU, CONDITIONAL, do not start unbidden.

### 5.1 Why this and not kernel width

W-5 eliminated depth, and eliminated capacity from 231,697 up to 535,396 parameters. SimpleEMGCNN sits at roughly 59,000. **The untested band is 59k to 231k**, and a threshold inside it is now the most economical explanation of the whole architecture-dependence, ahead of kernel width and the stem. The kernel widths are in any case similar between the two backbones, 9/7/5 on SimpleEMGCNN against a 9 stem and 7 in the blocks, so kernel width was never the strong candidate the thesis implies.

### 5.2 The arms

On `resnet_nores` geometry, `blocks_per_stage=2`, widths scaled down, per-subject normalization, 250 ms, matching W-5 exactly:

| Arm | Target parameters | Runs |
|---|---|---|
| NARROW-120K | about 120,000 | no-aug and chandrop p = 0.2 |
| NARROW-59K | about 59,000, matched to SimpleEMGCNN | no-aug and chandrop p = 0.2 |

Compute exact counts with `count_params` and land within 5% of target; report the counts before running. Four 40-fold runs. Use `--resume`.

### 5.3 Gates

Reuse W-5's: parameter counts reported first, inertness assertions already passed and re-asserted, one smoke test per geometry at `--heldout 1 --epochs 3`, and the trainability check. **The trainability gate matters far more here**, because a 59k residual network may genuinely fail to train on this task, which would be a confound and not a finding. If either no-augmentation baseline sits more than 3.0 pp below W-5's BASE at 0.7718, that arm's gain is not interpretable and must be reported as outcome X for that arm.

### 5.4 Pre-registered grid for P-4

Interaction contrasts against W-5's BASE, Holm across the family of two.

| Outcome | Condition | Reading |
|---|---|---|
| **TH. Threshold found** | The gain falls by at least 2.0 pp with a Holm-significant interaction at one or both narrow arms | Capacity is the operative property, with a threshold between the arms. §4.8.1 closes properly. |
| **N. No threshold** | The gain survives at 59k parameters | Capacity is eliminated across the whole range down to SimpleEMGCNN's own size, and what separates the two backbones is neither depth, capacity, SE nor skips. §4.8.1 should then say the architecture-dependence is real and unexplained, and §5.13 should carry the stem and the training-time details as what is left. |
| **X. Did not train** | Trainability gate fails on an arm | Exclude that arm and say the design lost the cell. |

### 5.5 Decision point: whether to run stage 4 at all

**Report stages 1 to 3 and stop.** Stage 4 costs GPU hours on a thesis whose objectives are met, and its most likely outcome, N, leaves §4.8.1 saying the dependence is unexplained, which the section can already say after W-5 at no cost. Enam decides.

---

## 5A. Stage P-5: a published alternative, not a self-control. GPU, ONE RUN, ~50 minutes.

### 5A.1 Why it exists

This is the sharpest remaining parity gap and it is not one of the four Enam listed. Normalization is benchmarked
against three published, named methods by other authors, each with a citation: CORAL, Deep CORAL and AdaBN.
Channel dropout is benchmarked against Gaussian noise, time masking, their combination, and gain jitter. The
first three are generic and uncited; the fourth is a control this thesis designed. **Channel dropout has never
been compared with a published alternative strategy aimed at the same goal**, so the comparison table in Section
4.8 tests it against variants of itself.

P-0 supplies the comparator. Pereira, Chalatsis, Hodossy and Farina (2024) train on a collection of input channel
subsets as augmentation and report +6.9% intersession accuracy under electrode displacement. Same goal,
channel-level robustness; different mechanism, a fixed vocabulary of subsets rather than random per-sample
masking. That is channel dropout's CORAL, and testing against it costs one run.

### 5A.2 The arm

Add a `subset` mode to `augment_batch` in `train_cnn_loso.py` and `run_cnn_arch_loso.py`, alongside the existing
`none`, `gaussian`, `chandrop`, `timemask`, `combined` and `gainjitter`. **Follow the `gainjitter` precedent
exactly**: guarded, defaulted off, and proven inert with byte-identical RNG-state and initial-parameter
assertions on all existing modes before any run. That precedent is recorded in the handoff and it is what keeps
every published number comparable.

**Definition, fixed here and not after seeing results.** Enumerate a fixed collection of channel subsets that each
retain 7 of the 9 channels, which is the expected number surviving channel dropout at p = 0.2 (9 x 0.8 = 7.2).
Draw one subset uniformly per training sample and zero the two channels it omits. This matches channel dropout on
expected channels retained and differs from it only in that the masking pattern comes from a fixed vocabulary
rather than being drawn independently per channel. **Matching the retention rate is the whole point of the
design**, in the same way W-4 matched multiplicative variance. Report the enumeration used and its size.

Run on `resnet_se`, per-subject normalization, 250 ms, every other flag at its default, into
`results_p5_subset_resnet_se`. One 40-fold run.

### 5A.3 What it is compared against, both already on disk

| Contrast | Against | Question |
|---|---|---|
| **Primary:** subset training vs channel dropout | `results_cnn_aug_resnet_se_chandrop` (0.840) | Does the published alternative beat, match or trail the incumbent? |
| **Secondary:** subset training vs no augmentation | the no-augmentation ResNet-SE arm (0.782) | Does it work at all on this task? |

Paired per subject across the 40, BCa 95% intervals, paired Wilcoxon, paired Cohen's d, Holm across the family of
two.

### 5A.4 Pre-registered grid for P-5

The 2.0 pp floor and its justification are carried over from W-5.

| Outcome | Condition | Reading |
|---|---|---|
| **M. Matches** | Primary contrast within 2.0 pp and not Holm-significant, secondary clearly positive | **The most likely outcome and a valuable one.** A third channel-structured perturbation, this one from the literature, does the same work. W-4's generalization stops resting on a control the thesis invented and starts resting on a published method too. Section 4.8.2's claim strengthens materially. |
| **B. Subset better** | Primary Holm-significant, at least 2.0 pp in favour of subset training | **This is a finding, not a problem, and must not be written defensively.** The object of study is the class of channel-level perturbation, not the standing of one member of it. A member that does the job better is informative precisely because it differs from channel dropout in a stated way, and that difference is evidence about what the class is doing. Report what differs and what it implies. **Separately and for engineering reasons only, the deep model of record does not change**: it was fixed a priori, the thesis is not a search for the best augmentation, and switching cascades through eight downstream result sets. Those two things are independent and the write-up should keep them apart. |
| **W. Subset worse** | Primary Holm-significant, at least 2.0 pp against subset training | Random per-sample masking beats a fixed subset vocabulary, which is a positive specificity result and the first one channel dropout has. Write it up as such. |
| **F. Subset fails** | Secondary contrast null or negative | Not every channel-level augmentation works here, which bounds W-4's generalization usefully. Report it. |
| **X. Did not train** | The arm fails a trainability check against the no-augmentation baseline at 3.0 pp | Exclude and report. |

Note that **every one of these five outcomes is publishable**, which is the test of a well-posed comparison.

### 5A.5 Scope limits

Read Section 6.2 before starting. The deep model of record does not change whatever this finds, and no downstream
artifact is regenerated. P-5 contributes two paired tests to the FDR family; report their raw p-values and do not
touch Section 4.17.

---

## 5B. Stage P-6: the perturbation ladder. GPU, 2 runs core, 2 optional.

### 5B.1 Why this is the most valuable experiment in the plan

W-4 concluded that any channel-structured multiplicative perturbation of sufficient variance does the work.
**That conclusion rests on a confound, found on 3 September by reading the implementation.**

`augment_batch` multiplies by `Bernoulli(1 - p)` with **no 1/(1 - p) rescaling**. At p = 0.2 that multiplier has
mean 0.8 and SD 0.40. `gainjitter` multiplies by a uniform variable with mean exactly 1.0 and SD 0.40. **W-4
matched the variance and left the mean unmatched.** Channel dropout injects per-channel variance *and* shifts
expected activation down by 20%; gain jitter injects the variance alone. The two are therefore not a clean
isolation of "removal versus variation", and the mean-preserving one scored 0.79 pp higher.

Two candidate operative quantities are still entangled: the **per-channel multiplicative variance** and the
**expected-activation shift**. Separating them is the direct analogue of what Section 4.13.1 did for
normalization, where the alignment ladder decomposed the operator by which moment it acts on and found that
location, not spread, is what matters. **This stage decomposes the perturbation by which moment of the multiplier
it acts on.** If it lands, the thesis gains a mechanistic statement about the class rather than a ranking of its
members, which is the same kind of finding as the over-alignment result and for the same reason.

### 5B.2 The core design, two runs

All arms on `resnet_se`, per-subject normalization, 250 ms, every other flag default. Two arms already exist.

| Arm | multiplier mean | multiplier SD | form | status |
|---|---|---|---|---|
| no augmentation | 1.0 | 0 | none | **exists**, 0.782 |
| channel dropout p = 0.2 | 0.8 | 0.40 | Bernoulli | **exists**, 0.840 |
| gain jitter sd = 0.40 | 1.0 | 0.40 | uniform, continuous | **new run** |
| mean-preserving channel dropout | 1.0 | 0.40 | Bernoulli | **new run and new mode** |

The mean-preserving arm multiplies by `Bernoulli(1 - p') / (1 - p')`, which is inverted dropout. That multiplier
has mean 1 and SD `sqrt(p' / (1 - p'))`, so **matching SD = 0.40 requires p' = 0.138**, not 0.2. Derive it in the
code from the requested SD rather than hard-coding, and print the realized mean and SD of the sampled multiplier
on the first batch as a gate.

**The two contrasts this buys, each isolating one thing:**

| Contrast | Matched on | Isolates |
|---|---|---|
| channel dropout p = 0.2 vs mean-preserving channel dropout | SD, form | **the expected-activation shift** |
| mean-preserving channel dropout vs gain jitter sd = 0.40 | mean, SD | **the form: does actual zeroing matter, or only the moments?** |

Both paired per subject over the 40, BCa intervals, paired Wilcoxon, paired Cohen's d, Holm across the family of
two. Report each arm against no augmentation as well, uncorrected and beside the family.

### 5B.3 The optional extension, two more runs

Gain jitter at sd = 0.30 and sd = 0.50, which match the multiplicative SD of channel dropout at p = 0.1 and
p = 0.5 respectively, since `sqrt(p(1-p))` gives 0.30 and 0.50. With the existing four-rate sweep that yields two
curves over one shared variance axis. **If they superimpose, the injected variance is the operative quantity and
the form of the perturbation is irrelevant. If they diverge, where they diverge says how the form matters.**
Run this only if the core two arms show the class behaving coherently.

### 5B.4 Note on the backbone, and on a superseded recommendation

The four-rate sweep is on `resnet_se` and W-4's gain jitter is on `resnet`, so the existing gain-jitter number
cannot enter this ladder. The handoff's Tier 4 item 8 recommends against running gain jitter on the SE backbone,
on the grounds that the two are statistically indistinguishable and switching the deep member would cascade.
**That recommendation was made against a different question.** It is still right that nothing here changes the
model of record. It is no longer right that the run has no value, because the ladder cannot be built without it.
Record that supersession in the handoff.

### 5B.5 Pre-registered grid for P-6

The 2.0 pp floor is carried over from W-5.

| Outcome | Condition | Reading |
|---|---|---|
| **V. Variance is what matters** | Neither contrast reaches 2.0 pp with Holm significance | The mean shift does nothing and the form does nothing. What the class does is inject per-channel multiplicative variance, full stop. **This is the cleanest possible version of the finding** and Section 4.8.2 should state it as the operative property. |
| **A. The activation shift matters** | The first contrast is significant and at least 2.0 pp | Reducing expected activation is doing part of the work, separately from the variance. Channel dropout is then not merely a degenerate case of gain jitter and W-4's conclusion narrows. Which direction the effect runs matters and must be reported. |
| **F. The form matters** | The second contrast is significant and at least 2.0 pp | Actual zeroing differs from continuous jitter at matched moments. That is a specificity result and the first one channel dropout has. |
| **AF. Both** | Both significant | Report both and do not force a single operative property. |
| **R. Reversal** | Any arm falls below no augmentation | Something is wrong with the arm, not with the class. Check the multiplier gate in 5B.2 before reading anything. |
| **X. Did not train** | Any arm more than 3.0 pp below the no-augmentation baseline | Exclude, report the lost cell. |

Every cell is publishable, including V, which is a null on both contrasts.

### 5B.6 Scope

The mode addition follows the `gainjitter` precedent: guarded, defaulted off, inertness assertions on all existing
modes before any run. Nothing downstream is regenerated, the deep model of record does not change whatever this
finds, and Section 4.17 is not edited. P-6 contributes two paired tests to the family, four with the extension.

---

## 6. Reporting, and what must not happen

### 6.1 Decision point: framing

Do not restructure any framing, chapter ordering or claim of standing. If stages 1 to 3 come back strong, say so and stop. **Whether the thesis presents itself as investigating two interventions rather than one is Enam's decision**, it touches the Abstract, Chapter 5 and Chapter 6, and it is explicitly out of scope for the executor.

### 6.2 Do not touch

The deep model of record stays the channel-dropout `resnet_se` at 0.840. The headline ensemble stays at 0.858. No downstream artifact is regenerated by any stage here.

### 6.3 The FDR family

Every paired test added by stages 1 to 4 joins the whole-thesis Benjamini-Hochberg family, which currently stands at 145 tests and 109 survivors under the scope stated in §4.17. **Do not update §4.17 yourself.** Report how many new paired tests each stage contributes and their raw p-values; the family is recomputed with `recompute_unified_fdr_v2.py` in one pass afterwards. Note that correlations against a permutation null are not paired Wilcoxon tests and should be reported separately rather than folded into that family, the way §4.8.2 already treats the G3 randomization test.

### 6.4 Outputs

`results_parity/` holding `p1_coupling_tests.csv`, `p1_arm_summary.csv`, `p2_rank_agreement.csv`, `p3_external_cd_tests.csv`, one verdict markdown per stage with its outcome letter, and `parity_outcome.json`. Scripts named `p1_coupling_stats.py`, `p2_rank_transfer_stats.py`, `p3_external_cd_stats.py`, following the shape of `w3_residual_stats.py`. Add a status header to the top of this file recording each stage's outcome, leaving the pre-registration below it unedited, which is the house pattern.

### 6.5 House rules

No em dashes anywhere, including in chat. Decimals in tables, percentages in prose, with effect sizes, p-values and correlations decimal everywhere. Do not be over-defensive in any prose you draft. State what was found.

---

## 7. What to report back

1. Confirmation that every input file exists and pairs, with row and subject counts.
2. Stage P-1: the dose curve, the two per-subject correlations with their permutation nulls, and the outcome letter.
3. Stage P-2: the rank-agreement figures, **both validity checks**, and the outcome letter.
4. Stage P-3: the primary and secondary contrasts with intervals and effect sizes, the number of subjects paired, and the outcome letter.
5. The number of new paired tests for the FDR family, with raw p-values, and no edit to §4.17.
6. An explicit statement that no thesis file was edited and no downstream artifact touched.
7. Your recommendation on whether stage 4 is worth its GPU hours, given what stages 1 to 3 found.
