# Experiment plan: the channel-dropout mechanism (W-2)

> ## STATUS, 2 September 2026: G1 RAN. OUTCOME R. THE GATE IS CLOSED.
>
> `results_cd_resnet_nose_chandrop` against `results_cd_resnet_noaug_repro`: channel dropout is
> worth **+6.51 pp** on the SE-free residual network against **+5.73 pp** with the SE block, an
> interaction contrast of **-0.78 pp, p = 0.221**. Squeeze-and-excitation is not the mechanism.
> §1.4's Outcome R applies: **G2 and G4 do not run.**
>
> | Stage | Disposition |
> |---|---|
> | G0 instrumentation | done; `--instrument` writes `occlusion.csv`, RNG assertions in place |
> | G1 interaction test | **done. Outcome R.** |
> | G2 dropout-rate sweep | **REACTIVATED by Enam, 2 September 2026, and running.** The gate
> closed it as an SE-mechanism follow-on, which it never was. Read instead as the dose-response
> curve, it is the analogue of the alignment ladder that §4.13 gives per-subject normalization,
> and §4.1 of this plan already framed it that way: a mechanism probe that would "mirror the
> over-alignment result of §4.13.2 and give the thesis 'too much invariance hurts' from two
> unrelated mechanisms." §4.1 and §4.3 below are unaltered and govern it. Scripts:
> `run_g2_rate_sweep.sh`, `g2_rate_stats.py` |
> | G2b noise floor | **superseded by R-1**, which measured run-to-run SD = 0.47 pp (n = 3). §4.3's threshold stands at 3.0 pp |
> | G3 occlusion sensitivity | **re-scoped and running separately.** It never depended on the SE hypothesis. See `EXPERIMENT_PLAN_G3_OCCLUSION.md`, which corrects §4.4's baseline architecture from `resnet_se` to `resnet` |
> | W-4 gain-jitter control | **new, 2 September 2026.** Channel dropout has never been tested
> against a channel-structured rival; Gaussian noise and time masking are strawmen by §5.7's own
> argument. See `EXPERIMENT_PLAN_GAIN_JITTER.md` |
> | G4 SE gate variance | **cancelled.** SE is no longer a candidate mechanism and `se_gates.csv` was never written, the G1 arm being correctly SE-free |
>
> The thesis consequence has been taken: §4.8's squeeze-and-excitation clause is removed and
> §4.8.1 now reports what was measured. `verify_section_4_8_1.py` regenerates every figure in it.
>
> The remaining mechanism question, that skip connections, depth and a ninefold capacity gap are
> still confounded, is `EXPERIMENT_PLAN_RESIDUAL_ABLATION.md` (W-3).
>
> **Everything below is the plan as written on 31 August, kept unedited as the pre-registration.**


**Status:** ready to run. Written 31 August 2026.
**Execution:** on Enam's machine. Everything runs locally; nothing needs the network.
**Structure:** one mandatory stage with a conditional gate, then three optional stages that only run if the gate opens.
**Owner decision points:** two, in §1.4 and §4.3. Do not resolve either. Report and stop.

---

## 0. Read this first

Self-contained. You do not need the conversation that produced it. If you are also running W-1, read §10 first, because the two interact.

Five things that will otherwise cost you a round trip each:

1. **The interpreter is `06_Code/.venv/Scripts/python.exe`.** Every `jobs_*.txt` in `06_Code/` points at `MSc Python Project/.venv/...`, a folder removed in the August restructure. Stale. Do not copy it.
2. **There are two unrelated parameters called `p` in this codebase.** `--aug-chandrop-p` is the channel-dropout augmentation rate and is what this experiment sweeps. The `p=0.1` in `ResBlock.__init__` in `cnn_architectures.py` is architectural dropout inside the block and **must not be touched**. Confusing them invalidates everything.
3. **No model weights are saved anywhere.** `run_cnn_arch_loso.py` keeps `best_state` in memory for early-stopping restore and discards it. Stages G3 and G4 therefore cannot be done as post-hoc analysis on existing runs. They require instrumentation, added once in G0, after which they cost no extra compute.
4. **`--n-jobs 1` on anything classical.** Not relevant to most of this plan, which is GPU-bound and serial, but it holds if you touch a classical driver.
5. **`--resume` on every run, always.**

---

## 1. The experiment

### 1.1 The claim under test

§4.8 of the thesis states, of channel dropout on ResNet-SE:

> "Channel dropout forces the network not to rely on any single electrode, which is exactly the failure mode of cross-subject sEMG, where electrode placement and per-channel envelope amplitude vary from person to person; **it also complements the squeeze-and-excitation block, whose channel reweighting is now trained to cope with missing channels.**"

The bolded clause is a mechanistic claim and **nothing in the thesis evidences it.** This is the same class of vulnerability the PRC criticism landed on for causality: a mechanism asserted in one sentence with no experiment behind it.

### 1.2 What is known, and the one cell that is missing

| Architecture | No augmentation | Channel dropout | Δ |
|---|---|---|---|
| SimpleEMGCNN | 0.7602 | 0.7567 | **−0.35 pp** |
| EMGResNet1D, no SE | 0.7563 | **missing** | **unknown** |
| EMGResNet1D + SE | 0.7822 | 0.8395 | **+5.73 pp** |

The architecture ablation already establishes that SE helps without channel dropout (0.7563 to 0.7822). What it never establishes is whether channel dropout *needs* SE. That single missing cell separates "channel dropout requires channel attention" from "channel dropout requires only a residual architecture", and the thesis currently asserts the first without testing it.

### 1.3 Why this is worth doing even if it changes nothing

Every outcome improves the thesis. If the mechanism holds, an asserted sentence becomes a measured one. If it does not, a wrong sentence gets corrected before a viva. There is no outcome in which the current text is the right text.

### 1.4 The conditional gate

**Stage G1 is mandatory. Stages G2 to G4 run only if G1 opens the gate.**

G1 is a single 40-fold run: `EMGResNet1D` with `use_se=False` plus channel dropout at p = 0.2. The quantity of interest is the **interaction contrast**, computed per subject and then tested paired across the 40:

```
interaction = (ResNet-SE+CD − ResNet-SE) − (ResNet+CD − ResNet)
```

A positive interaction means channel dropout buys more on the SE architecture than on the plain one, which is what §4.8 asserts.

| Outcome | Condition on the ResNet+CD gain | Meaning | Gate |
|---|---|---|---|
| **S. SE-dependent** | gain < 2.0 pp **and** interaction contrast significant and positive | §4.8's claim holds. Channel dropout needs channel attention. | **OPEN.** Proceed to G2 to G4. |
| **M. Mixed** | gain 2.0 to 4.0 pp | Both residual depth and SE contribute. §4.8 must be rewritten to apportion. | **OPEN**, but rewrite §4.8 first so G2 to G4 are framed correctly. |
| **R. Residual-dependent** | gain ≥ 4.0 pp **and** interaction contrast not significant | **SE is not the mechanism.** §4.8's claim is wrong and must be corrected. | **CLOSED. Stop and report to Enam.** |

Under outcome R, do not proceed to G2 to G4 on your own initiative. The mechanism story changes shape and the follow-on experiments would be answering the wrong question. Report and stop.

**A capacity caveat you must record whatever the outcome.** `EMGResNet1D` with `use_se=False` drops two `Linear` layers per block, six blocks in total, so it has fewer parameters than the SE variant. The comparison is therefore not strictly capacity-matched. Print and report the parameter count of both variants; the class docstring claims about 0.3M at defaults while the thesis reports 557,276, so do not trust either figure without printing it. This does not invalidate the interaction contrast, which is a within-architecture difference of differences, but it belongs in the write-up.

---

## 2. Stage G0 — instrument the driver (no compute)

Do this before G1 so that every subsequent run yields G3 and G4 data as a by-product rather than needing its own pass.

Add to `run_cnn_arch_loso.py`, behind a new `--instrument DIR` flag so that every existing caller is unaffected when the flag is absent. After a fold finishes training and before the model is discarded, on the **held-out subject's windows only**:

**Channel-occlusion sensitivity.** For each of the 9 EMG channels in turn, zero that channel across all the held-out subject's windows, re-score, and record the drop in macro-F1 against the unoccluded score. Write one row per (subject, channel) to `{DIR}/occlusion.csv` with columns `subject, channel, f1_full, f1_occluded, drop_pp`.

**SE gate activations.** Only when the architecture has SE. **These gate over convolutional feature-map channels, not over the 9 electrodes.** `SEBlock1d` computes `s = sigmoid(fc2(relu(fc1(avgpool(x)))))` and returns `x * s`, and the feature widths are 32, 64 and 128, so with `blocks_per_stage = 2` there are six SE blocks of differing width. Capture `s` with a forward hook on each block's `fc2`, applying the sigmoid yourself; do not try to recover `s` by dividing the block's output by its input, which is numerically unsafe wherever the input is near zero. Average over the held-out subject's windows and write one row per (subject, block, feature_channel) to `{DIR}/se_gates.csv` with columns `subject, block, block_width, feature_channel, gate_mean, gate_sd`.

**Verify the instrumentation is inert.** **Revised 1 September 2026 after R-1: bit-identical `f1_macro` is no longer an available test**, because R-1 established that two CNN process runs never reproduce bit-for-bit on this machine. Assert instead that torch, CUDA and NumPy RNG state is byte-identical before and after the instrumentation block on every instrumented fold, and confirm the hooks return `None`. That targets the actual corruption path, which is RNG advancement leaking into the next fold, rather than a property the hardware cannot deliver. Also diff the `--instrument`-absent code path to confirm it is unchanged.

---

## 3. Stage G1 — the interaction test (mandatory)

```
"<PY>" -u run_cnn_arch_loso.py --npz windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv --arch resnet --augmentation chandrop --aug-chandrop-p 0.2 --norm-mode per_subject --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --out results_cd_resnet_nose_chandrop --instrument results_cd_resnet_nose_chandrop/instr --resume
```

Confirm `--arch resnet` is the SE-free variant by checking `cnn_architectures.py`; the constructor exposes `use_se`, and `resnet` should set it False. If `--arch resnet` does not disable SE, stop and report rather than guessing.

Smoke test one fold with `--heldout 1` first.

Then run the statistics of §6 and apply the §1.4 gate. **Report the outcome letter before doing anything else.**

---

## 4. Stages G2 to G4 — only if the gate opened

### 4.1 G2, the dropout-rate sweep

Three new runs, p ∈ {0.1, 0.3, 0.5}. p = 0.2 already exists as the 0.840 model of record, but it was **not** instrumented, so re-run it under `--instrument` into a separate directory.

**Gate that re-run at ±1.5 pp, not ±0.002. Revised 1 September 2026 after R-1.** R-1 measured a run-to-run SD of 0.47 pp on this architecture. A ±0.002 tolerance is 0.4σ, so it would fail about **67% of the time by chance alone** and tell you nothing. ±1.5 pp is roughly 3σ and passes 99.8% of the time when nothing is wrong, which is what a gate is for.

**Then use the fresh p = 0.2 as the sweep's comparator, not the published 0.8395.** This is the same era-mixing problem G1 has: comparing three new runs against one old one lets any code drift between eras masquerade as a dropout-rate effect. Every arm of the sweep must come from the same code state.

```
for p in 0.1 0.3 0.5:
  "<PY>" -u run_cnn_arch_loso.py --npz windows_..._w250_....npz --meta features_out/freq_windows_..._w250_..._features_meta.csv --arch resnet_se --augmentation chandrop --aug-chandrop-p $p --norm-mode per_subject --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --out results_cd_rate_p$p --instrument results_cd_rate_p$p/instr --resume
```

**What this stage is for, and what it is not for.** It is a mechanism probe testing whether too much enforced channel invariance begins to destroy class structure, which would mirror the over-alignment result of §4.13.2 and give the thesis "too much invariance hurts" from two unrelated mechanisms. **It is not a hyperparameter search.** p = 0.2 was fixed a priori and is the operating point. See §4.3 before letting any other p win.

The shape to look for is monotone improvement then decline. If p = 0.5 is materially worse than p = 0.2, the over-regularization parallel holds and is reportable.

### 4.2 G2b, run-to-run variability and seed replication

**Revised 1 September 2026 after W-1. The original threshold in this section sat below the noise floor and would have produced false positives.**

W-1 generated a fresh ResNet-SE+CD run at 250 ms under global normalization and got **0.7723**, where the thesis publishes **0.7874** for the same quantity (`f1_pre_adabn_mean` in `results_adabn_chandrop`). Those runs share a protocol: `run_adabn_cnn_loso.py` imports `train_fold` directly from `run_cnn_arch_loso.py`, applies the same `compute_train_norm` / `apply_norm` train-fold global z-score, and uses the same epochs, patience, val-frac and seed. **So the CNN pipeline is not run-to-run reproducible to better than about 1.5 pp**, almost certainly because cuDNN determinism is never enabled (no `torch.backends.cudnn.deterministic`, no `benchmark = False`).

The 0.0038 seed SD in `results_seed_stability/` does **not** bound this. It was measured on SimpleEMGCNN, across seeds, and it is roughly a quarter of the run-to-run spread just observed on ResNet-SE.

**So measure the noise floor before interpreting the sweep.** Re-run p = 0.2 twice more at seed 42, changing nothing at all. Three runs at identical settings give a direct estimate of run-to-run SD on this architecture. That is 2.6 hours and it is no longer optional; without it the sweep cannot be read.

Then re-run p = 0.2 and the highest-scoring p under **seed 7** for the cross-seed check, as originally planned. Two more runs.

### 4.3 The second decision point: can a different p win?

**Pre-registered. Revised 1 September 2026 to sit above the measured noise floor, before any sweep result was seen.**

A p other than 0.2 displaces the operating point only if **all four** hold:

- Holm-corrected paired Wilcoxon p < 0.05 against p = 0.2 across the 40 subjects, and
- **|Δ| ≥ 3.0 pp**, or at least twice the run-to-run SD measured in G2b, whichever is larger, and
- the ordering reproduces under seed 7, and
- the margin exceeds the spread between the G2b repeats of p = 0.2 against itself.

The original threshold was 1.0 pp, justified against a 0.38 pp seed SD. W-1 showed that figure does not apply to this architecture, and a 1.0 pp rule would have let ordinary run-to-run noise displace the deep model of record and cascade into the ensemble, the causal chain and the headline. **If G2b shows a run-to-run SD above 1.5 pp, the sweep can support a qualitative shape claim only, not a ranking.** That is still a good result: the over-regularization parallel needs p = 0.5 to be clearly worse, not p = 0.3 to be marginally better.

If all four hold, **stop and escalate to Enam.** Do not adopt the new p and do not touch any downstream artifact. §9 explains why.

If any one fails, **p = 0.2 stands** and the sweep is reported as a mechanism result only. This is the expected and desired case.

### 4.4 G3, channel-occlusion sensitivity

No new compute. Analyse `occlusion.csv` from the instrumented runs, comparing the CD model against a no-augmentation ResNet-SE baseline, which needs one instrumented run of its own:

```
"<PY>" -u run_cnn_arch_loso.py --npz ... --arch resnet_se --augmentation none --norm-mode per_subject --seed 42 --out results_cd_baseline_noaug --instrument results_cd_baseline_noaug/instr --resume
```

Two quantities, and the second matters more:

- **Within-subject reliance concentration.** The spread of `drop_pp` across the 9 channels for a given subject. The hypothesis is that channel dropout flattens it, meaning no single electrode dominates.
- **Between-subject profile consistency.** Correlate each subject's 9-channel importance profile against every other subject's, and take the mean pairwise correlation. **This is the one that speaks to cross-subject generalization**, because a model whose channel reliance is consistent across people is one that has not latched onto per-subject electrode idiosyncrasy. Report both, and lead with this one.

### 4.5 G4, SE gate variance across subjects

No new compute. **Note carefully what this measures and what it does not.** The SE gates weight convolutional feature maps, not the 9 EMG electrodes, so this stage cannot say which electrode the model relies on. That question belongs entirely to G3. What G4 asks is whether channel dropout makes the *learned* channel attention more consistent from one subject to the next.

From `se_gates.csv`, for each SE block and feature channel, compute the variance of `gate_mean` across the 40 subjects, then summarize per block. Compare the channel-dropout model against the no-augmentation baseline. If channel dropout reduces the across-subject variance of the gates, the mechanism is drawn directly rather than argued: the attention has settled on a weighting that does not need re-tuning per person.

Report per block and never pooled, since the six blocks differ in width and sit at different depths, and there is no reason to expect them to behave alike. Normalize before comparing across blocks of different width.

**This stage is the weakest of the four and should be reported as suggestive.** A gate is an internal activation, not an outcome, and a change in its variance is consistent with several stories. Lead the write-up with G3, which measures something the model actually does.

---

## 5. What already exists and must be reused, not re-run

| Artifact | Path | Value |
|---|---|---|
| SimpleEMGCNN, no aug | `results_cnn_loso_simple_repro/` | 0.7602 |
| SimpleEMGCNN + CD | `results_cnn_loso_aug_chandrop/` | 0.7567 |
| ResNet, no SE, no aug | `results_cnn_loso_resnet/` | 0.7563 |
| ResNet-SE, no aug | `results_cnn_loso_resnet_se/` | 0.7822 |
| ResNet-SE + CD, p=0.2 | `results_cnn_aug_resnet_se_chandrop/` | 0.8395 |
| CNN seed stability | `results_seed_stability/seed_stability_summary.csv` | SD 0.0038 |

Per-subject F1 for the paired tests comes from each directory's `cnn_arch_subjectwise.csv`.

---

## 6. Statistics

Thesis convention, as §3.6 sets it and Chapter 4 uses it throughout.

- Paired Wilcoxon signed-rank across the 40 subjects.
- Paired Cohen's d.
- BCa bootstrap 95% CI, 10,000 resamples, seed 42.
- Holm correction **within** families, never across. Family 1 is the G1 interaction analysis. Family 2 is the four rate-sweep comparisons against p = 0.2. Family 3 is the G3 and G4 mechanism comparisons.
- Decimals in tables, percentages in prose. Effect sizes, p-values and correlations stay decimal everywhere; a blanket regex will corrupt them.

**The interaction contrast is a difference of differences and must be computed per subject before testing**, not as a difference of two group means. Each subject contributes one value, and that vector is what the Wilcoxon runs on.

Reuse `window_ablation_stats.py` as the pattern for BCa and Holm; both are already implemented and verified there.

---

## 7. Reproduction gates

Before any new number is reported, confirm the reused arms read back at their published values from their existing CSVs. Do not re-run them.

| Arm | Expected | Tolerance |
|---|---|---|
| SimpleEMGCNN, no aug | 0.7602 | ±0.002 |
| ResNet, no SE, no aug | 0.7563 | ±0.002 |
| ResNet-SE, no aug | 0.7822 | ±0.002 |
| ResNet-SE + CD, p=0.2 | 0.8395 | ±0.002 |

Plus, in G2, the instrumented re-run of p = 0.2 must reproduce 0.8395 **within 1.5 pp** (see §4.1; ±0.002 is below the noise floor and is not a usable gate on this architecture). Inertness of the instrumentation is established separately by the RNG-state assertion in G0, not by reproducing a number.

**These four gate values are read from existing CSVs, not re-run.** That is why ±0.002 is right for them: reading a stored number is exact. Only a *fresh* run needs the wider tolerance.

Second gate on every new run: exactly 40 rows, one per subject, no duplicates. Append-mode checkpointing double-appends on a bad resume and it is silent.

**If a gate fails, stop and report. Do not adjust the expected value to match what you found.**

---

## 8. Cost

GPU-bound and serial; the GPU does not benefit from concurrency here. Per 40-fold configuration is estimated at **1.3 hours**, and that is an estimate, not a measurement, because no CNN run in this project has ever recorded timing. If G1 takes materially longer, say so rather than letting the rest run past the estimate.

| Stage | Runs | Hours |
|---|---|---|
| G0 instrumentation | 0 | 0, plus the inertness check |
| **G1 interaction test (mandatory)** | 1 | **1.3** |
| G2 rate sweep, p = 0.1, 0.3, 0.5 | 3 | 3.9 |
| G2 instrumented p = 0.2 re-run | 1 | 1.3 |
| G2b noise floor, 2 repeats of p=0.2 at seed 42 | 2 | 2.6 |
| G2b seed replication | 2 | 2.6 |
| G3/G4 no-augmentation baseline | 1 | 1.3 |
| G3, G4 analysis | 0 | under 1 |
| **Total if the gate opens** | 10 | **about 13.0** |

---

## 9. Downstream impact: what this can and cannot change

**Read this before running G2.**

G1, G3 and G4 change **no existing number in the thesis.** G1 adds a new cell to the architecture comparison. G3 and G4 are new analyses that stand beside the existing results. None of them touches the ensemble, the causal work, the external validation or the headline.

**All of the risk sits in one branch: G2 finding a materially better p.** The channel-dropout ResNet-SE at p = 0.2 is the deep model of record, and eight downstream result sets consume it:

1. §4.8 Table 4.10 and the model-of-record designation
2. §4.2.2 architecture comparison
3. §4.9 the soft-vote ensemble, the 0.858 headline, which needs the member probabilities regenerated
4. §4.9.1 the Random Forest calibration comparison
5. §4.12 ENABL3S external validation, run on the chandrop backbone
6. §4.13 AdaBN and Deep CORAL, both on the chandrop backbone
7. §4.14 the causal ensemble and the 0.817 deployable figure
8. §4.15 supervised subject calibration

and through them the Abstract, §5.11 and Chapter 6.

This is why §4.3 requires three independent conditions before a new p can displace the operating point, and why even then it escalates rather than proceeding. **A 0.5 pp improvement is not worth regenerating the causal chain, and inside the measured seed noise it is not an improvement at all.**

---

## 10. Interaction with W-1, the window-length ablation

If W-1 returns outcome C and Enam re-baselines to 400 ms, every run in this plan must be redone at 400 ms, because architecture and augmentation effects are not guaranteed to transfer across window lengths.

G1 is 1.3 hours and worth running regardless; the rework risk is small. **G2 to G4 are about 9 hours and should wait until W-1's outcome is known.** If W-1 has not been run, say so and let Enam sequence the two rather than deciding for him.

---

## 11. What to report

1. The G1 outcome letter, S, M or R, in the first line, with the gain, the interaction contrast, corrected p, d and CI.
2. The parameter counts of both ResNet variants.
3. Gate results from §7, pass or fail, with numbers.
4. Whether the instrumentation was verified inert.
5. If the gate opened: the rate-sweep table, the §4.3 verdict on whether any p displaces 0.2, and the G3 and G4 mechanism results with the between-subject profile consistency led first.
6. Actual wall-clock against the §8 estimates.

**Do not edit `MSc Thesis.docx` or any chapter file, whatever the outcome.** §4.8's sentence needs revising under every outcome, but that is a writing decision Enam makes once he has the numbers.
