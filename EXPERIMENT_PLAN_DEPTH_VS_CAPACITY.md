# Experiment plan: separating depth from capacity (W-5)

> **RUN 2 September 2026. OUTCOME N (neither).** Channel-dropout gain is flat across
> geometry: BASE +4.55 pp, SHALLOW-MATCHED +4.93 pp, NARROW-MATCHED +4.33 pp; both
> interaction contrasts null (BASE−SHALLOW −0.38 pp Holm p=1; BASE−NARROW +0.22 pp
> Holm p=1). Both reduced arms trained (no-aug baselines −1.97 and −0.04 pp vs BASE,
> inside the 3.0 pp X-threshold). Widths chosen: SHALLOW-MATCHED bps=1 (48,96,192)
> = 522,196 (−2.47%); NARROW-MATCHED bps=2 (21,42,84) = 231,697 (−0.75%). Inertness
> passed on resnet/resnet_se/resnet_nores. §1.5 did not trigger. §6 decision (whether
> and where to write this up) is Enam's; nothing downstream or in the thesis was
> touched. Numbers: `results_w5/w5_verdict.md`. Pre-registration below is unedited.

**Status:** ready to run. Written 2 September 2026.
**Execution:** on Enam's machine. Everything runs locally.
**Cost:** four 40-fold runs plus two smoke tests, about **2 to 3.5 hours of GPU**, on the W-3 timings of 25 to 50 minutes per 40-fold run.
**Owner decision points:** two, in sections 1.5 and 6. Do not resolve either yourself. Report and stop.
**This is Tier 4, which is optional.** If the calendar will not take the whole plan, run none of it. Section 5.13 of the thesis already records this as open, and a half-answered version is worse than a stated limitation. See section 8.

---

## 0. Read this first

Five things that will otherwise cost a round trip each.

1. **The Python interpreter is `06_Code/.venv/Scripts/python.exe`.** Every `jobs_*.txt` in `06_Code/` points at a `MSc Python Project/.venv` path that was removed in the August restructure. Those files are stale.
2. **`--n-jobs 1` on every classical job.** Not applicable here since this is all CNN work, but the rule stands if you touch anything classical.
3. **This experiment changes nothing downstream and must not.** The deep model of record stays the channel-dropout `resnet_se` at 0.840, and the headline ensemble stays at 0.858. Whatever this finds is a mechanism result on the SE-free, skip-free backbone. **Do not promote any configuration built here into any downstream artifact.** Switching the deep member cascades through eight result sets.
4. **A code change is required, and it must be inert.** See section 3. The project's precedent is `use_residual` in `cnn_architectures.py` and the `gainjitter` mode in the two drivers: both were added behind guards and both passed byte-identical RNG-inertness assertions before any new run. Follow that precedent exactly, or every existing published number becomes suspect.
5. **Write a cell for the reversal and one for the measure breaking.** Four pre-registered grids in this project have already failed because they assumed effects would land in the expected direction: W-3 had no cell for "not necessary but contributory", G3 had none for a significant reversal, G2 applied a displacement threshold to a shape question, and W-1's own grid left a hole that the SVM's 400-versus-250 result fell into. Section 5 below has cells N, R and X for exactly this reason. **Do not delete them because they look unlikely.**

---

## 1. The experiment

### 1.1 The confound it closes

Section 4.8.1 decomposes the architecture-dependence of channel dropout. The augmentation is worth +0.30 pp on SimpleEMGCNN, +4.55 pp on the skip-free residual network, and +6.51 pp with skip connections added. The step from SimpleEMGCNN to the skip-free residual network is +4.24 pp, roughly two thirds of the total, and Section 4.8.1 says in print that **what no experiment reported there separates is depth from capacity.** Those two remain entangled with each other, and with the kernel width and stem that also distinguish the backbones.

This is the last confound in that section and the only one that is cheap to close.

### 1.2 The question, stated so it can be answered

Does channel dropout's effectiveness depend on the **depth** of the backbone, on its **capacity**, or on both?

The experiment answers it by moving one and holding the other, on the SE-free, skip-free backbone that W-3 already established as a clean baseline.

### 1.3 The arms

All arms use `resnet_nores` geometry, that is `use_se=False` and `use_residual=False`, on the 250 ms windows under per-subject normalization, which is what W-3 and W-4 used.

| Arm | Depth | Capacity | Status |
|---|---|---|---|
| **BASE** `blocks_per_stage=2`, `widths=(32,64,128)` | 2 | 535,396 params | **Already run. Reuse, do not re-run.** `results_w3_nores_noaug` (0.7718) and `results_w3_nores_chandrop` (0.8172) |
| **SHALLOW-MATCHED** `blocks_per_stage=1`, widths widened to hold parameters | 1 | approximately 535k, matched to BASE | **build and run, both augmentation states** |
| **NARROW-MATCHED** `blocks_per_stage=2`, widths narrowed to cut parameters | 2 | approximately the parameter count of a `blocks_per_stage=1, widths=(32,64,128)` net | **build and run, both augmentation states** |

Four new 40-fold runs: SHALLOW-MATCHED with `--augmentation none` and with `--augmentation chandrop --aug-chandrop-p 0.2`, and the same pair for NARROW-MATCHED.

**Choosing the widths is your job and must be reported.** Block parameters scale roughly with the square of the width, so halving `blocks_per_stage` from 2 to 1 roughly halves them and restoring the count wants widths scaled by about the square root of two. Candidate starting points are `(48, 96, 192)` or `(40, 80, 160)` for SHALLOW-MATCHED and `(24, 48, 96)` for NARROW-MATCHED. **Compute the actual counts with `count_params` in `cnn_architectures.py` and pick the candidate that lands within 3% of its target.** Report the four counts before running anything.

### 1.4 What is measured

For each arm, the **channel-dropout gain**: per-subject paired difference between the channel-dropout run and the no-augmentation run of the same geometry, over the 40 held-out subjects, with a BCa 95% interval, a paired Wilcoxon p-value and a paired Cohen's d. This is exactly the quantity W-3 and W-4 report.

Then the two **interaction contrasts**, each formed per subject as the difference of two differences:

- `BASE gain − SHALLOW-MATCHED gain`. A large positive value means cutting depth at matched capacity costs the augmentation, so **depth** is what matters.
- `BASE gain − NARROW-MATCHED gain`. A large positive value means cutting capacity at matched depth costs the augmentation, so **capacity** is what matters.

Holm correction runs across the family of **two** interaction contrasts. The per-arm gains are reported beside them and are not part of that family.

### 1.5 Decision point one, for Enam and not for you

If the two candidate width tuples both land outside 3% of target, or if matching the parameter count forces a width that is not a multiple of eight and the count moves by more than 5%, **stop and report the options.** Do not pick a geometry that makes the comparison unclean in order to keep the plan moving.

---

## 2. Gates, before any 40-fold run

1. **Parameter counts.** BASE is 535,396. SHALLOW-MATCHED must be within 3% of it. NARROW-MATCHED must be within 3% of the `blocks_per_stage=1, widths=(32,64,128)` count. Report all four numbers.
2. **Inertness.** See section 3. This gate is not optional.
3. **Smoke test.** Run each new geometry on a single held-out subject with `--heldout 1 --epochs 3` and confirm it trains, the output CSV is written, and the fold completes. Two smoke tests, a few minutes each.
4. **Reproduction.** Re-running BASE is not required, but confirm `results_w3_nores_noaug/cnn_arch_subjectwise.csv` and `results_w3_nores_chandrop/cnn_arch_subjectwise.csv` each still hold 40 rows with no duplicate subjects, and that their means are 0.7718 and 0.8172.

---

## 3. The code change, and how to make it inert

`EMGResNet1D.__init__` already accepts `widths` and `blocks_per_stage`. `build_model` hardcodes them, so only the dispatch needs to change.

**Add flags, do not edit the existing arch strings.** Extend `build_model` with optional `widths` and `blocks_per_stage` arguments that default to the current values, and add `--widths` and `--blocks-per-stage` to `run_cnn_arch_loso.py`, defaulting to the current geometry. The strings `simple`, `resnet`, `resnet_se` and `resnet_nores` must construct byte-identical models when the new flags are absent.

**Then prove it.** Before any new run:

- Build `resnet`, `resnet_se` and `resnet_nores` with the new code and with `git stash` on the old code, and assert the parameter counts and the initial parameter bytes are identical under a fixed seed.
- Assert the RNG state after model construction is identical between old and new code for each of the three.

This is the same assertion W-3 and W-4 passed. **If it fails, stop.** A silent change to initialization would decouple every new run from every published number.

---

## 4. Commands

Preprocessing and features already exist. Use the 250 ms windows npz that W-3 and W-4 used.

```
"<PY>" -u run_cnn_arch_loso.py --npz windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
  --meta <the w250 features meta csv> --arch resnet_nores \
  --widths <w1,w2,w3> --blocks-per-stage <1 or 2> \
  --norm-mode per_subject --augmentation none \
  --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 \
  --out results_w5_<arm>_noaug
```

and the same with `--augmentation chandrop --aug-chandrop-p 0.2` writing to `results_w5_<arm>_chandrop`. Every unspecified flag must be left at its default, which is what W-3 and W-4 did.

Use `--resume`. A dropped run should not restart from fold zero.

---

## 5. Pre-registered decision grid. Fix this before running.

Let `I_shallow` and `I_narrow` be the two interaction contrasts of section 1.4, Holm-corrected across the family of two. An arm is said to **lose the effect** when its interaction is Holm-significant at 0.05 **and** its point estimate is at least **2.0 pp**. The 2.0 pp floor is set here, in advance, because the run-to-run standard deviation for this backbone under per-subject normalization is 0.59 pp measured on two repeats, and twice that is 1.2 pp; 2.0 pp clears it with margin without being so large that a real effect is missed.

| Outcome | Condition | Reading and action |
|---|---|---|
| **D. Depth** | SHALLOW-MATCHED loses the effect, NARROW-MATCHED does not | Depth is what channel dropout needs. Section 4.8.1's "depth and capacity" becomes "depth", and the transferable recommendation becomes "use channel dropout on a deep network". Write it up. |
| **C. Capacity** | NARROW-MATCHED loses the effect, SHALLOW-MATCHED does not | Capacity is what it needs. The recommendation becomes "use it on a large network". Write it up. |
| **B. Both** | Both lose the effect | The two are genuinely joint at this scale. Report the two interactions and keep the conjunction in Section 4.8.1, now evidenced rather than assumed. |
| **N. Neither** | Neither loses the effect | The +4.24 pp step is not attributable to depth or capacity within this range. The remaining candidates are the kernel width and the stem, which this design does not vary. **This is an informative negative and must be written up as one**, not buried. |
| **R. Reversal** | Any reduced arm shows a channel-dropout gain **larger** than BASE by a Holm-significant margin of at least 2.0 pp | Not anticipated by the mechanism account. Report the numbers, do not force a letter, and escalate to Enam before writing anything. |
| **X. The measure broke** | Any arm's no-augmentation baseline sits more than **3.0 pp below** BASE's 0.7718 | That arm did not train comparably, so its channel-dropout gain is not interpretable and its interaction must not be read. Report which arm, exclude it, and say plainly that the design lost that cell. |

The 3.0 pp trainability threshold is the one W-3 used for the same purpose and is carried over deliberately.

**On the primary claim:** nothing in this experiment can overturn Section 4.8.2's finding that channel dropout works, or the 5.6-fold occlusion result. If you find yourself writing a sentence that weakens either, stop and escalate.

---

## 6. Decision point two, for Enam and not for you

Whatever the outcome letter, **do not write it into the thesis without asking.** Section 4.8.1 currently ends on a deliberately bounded claim, that channel dropout's effectiveness depends on the depth and capacity of the backbone with those two entangled. Replacing a bounded claim with a sharper one is a change of argument, not an edit, and Enam decides whether Section 4.8.1 absorbs it, whether it goes into Section 5.13 as a closed question, or whether it stays out of a thesis that is already 136 pages.

Report the numbers and stop.

---

## 7. Outputs

Write into `06_Code/`:

- `results_w5_shallow_noaug/`, `results_w5_shallow_chandrop/`, `results_w5_narrow_noaug/`, `results_w5_narrow_chandrop/`, each with `cnn_arch_subjectwise.csv` and `cnn_arch_summary.csv`.
- `w5_depth_capacity_stats.py`, following the shape of `w3_residual_stats.py`: BCa intervals, paired Cohen's d, Holm across the family of two, and an outcome letter chosen by the grid in section 5 with **an explicit cell for N, R and X**.
- `w5_depth_capacity_pairs.csv`, per-subject values for all four arms.
- `results_w5/w5_verdict.md` and `w5_outcome.json`, matching the pattern of `window_ablation_verdict.md`.
- A status header at the top of **this file** recording the outcome, leaving the pre-registration below it unedited. That is the house pattern for every plan in this folder.

---

## 8. When not to run this

The honest case for skipping it, recorded so the decision is available rather than rediscovered:

- It closes a confound in one subsection of one section. It does not touch either of the thesis's two headline findings.
- Section 5.13 already names it as open, in the specific terms this plan would resolve, so the thesis is not currently wrong about anything.
- The outcome most likely on the evidence is **B**, both, which changes the wording of Section 4.8.1 very little.
- Tier 4's other item, gain jitter on the squeeze-and-excitation backbone, is recommended against for the same reason and should stay untried.

Run it if there is GPU time and calendar to spare before the viva. Do not run half of it.

---

## 9. What to report back

1. The four parameter counts and which width tuples were chosen.
2. Whether the inertness assertions passed, stated explicitly.
3. Both smoke tests.
4. Per-arm channel-dropout gains with intervals, p-values and effect sizes, and the two Holm-corrected interaction contrasts.
5. The outcome letter, or an explicit statement that the result fell outside the grid, which is a reportable finding in itself.
6. Confirmation that no downstream artifact was touched and that the deep model of record is unchanged.
