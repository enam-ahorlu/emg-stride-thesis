# Experiment plan: what breaks channel dropout at high dose (P-7)

> **RUN 3 September 2026. OUTCOME M (the mean shift carries it), narrowly.** One 40-fold run,
> `results_p7_mpchandrop_resnet_se_sd0.50` = 0.8379 (n=40). No code changes.
> - **All four gates PASS.** Gate 1 multiplier: derived p'=0.200000, realized multiplier mean
>   1.00132, SD 0.49901. Gate 2 inertness: six existing modes byte-identical, resnet/resnet_se init
>   unchanged. Gate 3 completeness: 40 rows / 40 unique / no dups / subject ids identical to
>   `results_cd_rate_p0.5` and `results_p6_gainjitter_resnet_se_sd0.50`. Gate 4 additivity:
>   D1 + D2 = +1.73 + +1.08 = +2.81 pp, observed gap +2.81 pp, residual +0.00 pp (tol +/-0.30).
> - **Trainability:** mean-preserving arm gain vs no aug +5.57 pp against channel dropout p0.5's
>   +3.83 pp (difference +1.73 pp, well within the 3.0 pp bound; the mean-preserving form trains
>   better, not worse).
> - **Decomposition at SD 0.50** (n=40, share of the +2.81 pp gap): **D1 expected-activation shift
>   +1.73 pp, 61.7%, Holm p = 8.2e-05, d = 0.62**; **D2 form +1.08 pp, 38.3%, Holm p = 0.0026,
>   d = 0.50**. D1 clears the pre-registered 60% bar by 0.05 pp, so the letter is M by the rule, but
>   the split is close to B and D2 is a real, Holm-significant secondary contributor. Report both
>   shares.
> - **Restated at SD 0.40** (era-consistent, vs `results_cd_rate_p0.2` 0.8376): D1 -0.26 pp (n.s.,
>   Holm-uncorrected p = 0.74), D2 +1.28 pp (p = 0.013), observed gap +1.01 pp. So the
>   expected-activation shift does nothing at SD 0.40 (mean 0.8) and carries the majority at SD 0.50
>   (mean 0.5): its contribution grows with dose. The form contributes at both doses.
> - **Reading:** channel dropout is predominantly a variance injector; at high rate it acquires an
>   activation-collapse cost that the mean-preserving form removes. The discrete-zeroing form is a
>   smaller, dose-stable, significant second effect. This converts P-6's described divergence into
>   an explained one.
> - **FDR family:** two new paired Wilcoxon tests, D1 raw p = 4.099e-05, D2 raw p = 0.002639.
>   **Section 4.17 not edited.** No thesis file edited. Deep model of record unchanged
>   (channel-dropout resnet_se, 0.840). Section 7 decision (framing, held pending this result, in
>   `EXPERIMENT_PLAN_CD_PARITY.md` section 6.1) is Enam's. Numbers:
>   `results_parity/p7_verdict.md`. Pre-registration below is unedited.

**Status:** ready to run. Written 3 September 2026.
**Executor:** Claude Code, on Enam's machine.
**Cost: one 40-fold run, about 55 minutes.** No new code. The mode this needs already exists and already passed its
inertness and multiplier gates during P-6.
**Owner decision point:** one, in section 7. Do not resolve it yourself.
**This is the last experiment in the channel-dropout programme.** After it reports, the deferred queue in the
handoff runs: the FDR recompute, the exhibits, and the voice pass. Do not start any of those here.

---

## 0. Read this first

1. **The Python interpreter is `06_Code/.venv/Scripts/python.exe`.** The `jobs_*.txt` files are stale.
2. **`--aug-gain-sd` is the requested multiplicative SD for `mpchandrop`, not a gain-jitter-only flag.** The mode
   derives `p' = gain_sd^2 / (1 + gain_sd^2)`. At 0.50 that is **p' = 0.2**. Never hard-code p'.
3. **Pair against `results_cd_rate_p0.5`, not against any other channel-dropout arm.** Two p = 0.2 runs of the same
   configuration exist and differ by 0.19 pp (`results_cnn_aug_resnet_se_chandrop` 0.8395 against
   `results_cd_rate_p0.2` 0.8376), which G2 recorded as its reproduction gate. The rate sweep and the P-6 extension
   are era-internal with each other, so the comparator for P-7 is the rate-sweep arm.
4. **This plan does not use the 2.0 pp materiality floor**, and section 5 explains why. Applying a floor built for
   a different question is the mistake G2 made and W-1 repeated.

---

## 1. The question

P-6 established that at a multiplicative SD of 0.40 the class of channel-level perturbation is characterized by
its injected variance alone: the expected-activation shift does nothing (+0.45 pp, n.s.) and the form is
detectable but immaterial (-1.28 pp, Holm 0.026).

P-6's extension then found that the forms **diverge at SD 0.50**, by +2.81 pp in favour of gain jitter
(Holm p = 6.8e-07, d = 0.95). Channel dropout falls there, from 0.8376 at SD 0.40 to 0.8205 at SD 0.50, while gain
jitter at matched variance does not fall at all, 0.8477 to 0.8486.

**That divergence re-confounds the two things P-6's core separated.** Plain channel dropout at p = 0.5 has a
multiplier mean of 0.5; gain jitter has a mean of 1.0. C1 tested the mean shift only at SD 0.40, where the mean is
0.8. So at SD 0.50 the 2.81 pp gap could be the mean shift having grown large, or the discrete zeroing beginning
to bite, and nothing in the data distinguishes them.

**One arm resolves it.** Mean-preserving channel dropout at SD 0.50 sits at mean 1.0 with Bernoulli form, which is
exactly the missing cell.

### 1.1 Why this is worth 55 minutes

G2 recorded that channel dropout shows "a plateau with a shallow fall at the top of the range" and stated in
writing that the over-alignment parallel to Section 4.13.2 was **not supported**. P-6's extension shows the fall
is not a property of the dose, because the same variance delivered differently does not produce it. P-7 says which
property of the delivery produces it. That converts a described divergence into an explained one, and it is the
difference between a Chapter 4 observation and a statement about what this class of lever does.

---

## 2. The arm, and the two it completes

All on `resnet_se`, per-subject normalization, 250 ms, seed 42, every other flag at its default.

| Arm | multiplier mean | multiplier SD | channels zeroed | status |
|---|---|---|---|---|
| channel dropout p = 0.5 | 0.5 | 0.50 | 50%, not rescaled | **exists**, `results_cd_rate_p0.5`, 0.8205 |
| **mean-preserving chandrop, p' = 0.2** | **1.0** | **0.50** | **20%, rescaled by 1.25** | **run this** |
| gain jitter sd = 0.50 | 1.0 | 0.50 | none | **exists**, `results_p6_gainjitter_resnet_se_sd0.50`, 0.8486 |

```
"<PY>" -u run_cnn_arch_loso.py --npz windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
  --meta <the w250 features meta csv> --arch resnet_se --norm-mode per_subject \
  --augmentation mpchandrop --aug-gain-sd 0.50 \
  --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --resume \
  --out results_p7_mpchandrop_resnet_se_sd0.50
```

### 2.1 The two contrasts, which mirror P-6's core at a second dose

| Contrast | Matched on | Isolates |
|---|---|---|
| **D1:** mean-preserving chandrop minus channel dropout p = 0.5 | SD, form | the **expected-activation shift** at high dose |
| **D2:** gain jitter sd 0.50 minus mean-preserving chandrop | mean, SD | the **form** at high dose |

Paired per subject over the 40, BCa 95% intervals, paired Wilcoxon, paired Cohen's d, Holm across the family of
two. Report each arm against no augmentation (0.7822) uncorrected, beside the family.

**This is deliberately the same decomposition P-6 ran at SD 0.40.** Reporting both doses together gives how each
property's contribution changes with dose, which is more than either dose gives alone.

---

## 3. Gates. All four before reading anything.

1. **Multiplier gate.** Run `p6_multiplier_gate.py` against the new setting and confirm the realized multiplier has
   mean 1.00 and SD 0.50, and that the derived p' is 0.2. Report the realized figures, not the intended ones.
2. **Inertness.** Re-assert with `p5p6_inertness.py` that all existing modes are byte-identical. It passed during
   P-6 and no code changes here, so this is a regression check and should be cheap.
3. **Completeness.** 40 rows, 40 unique subjects, no duplicates, subject id set identical to the two comparator
   arms.
4. **Additivity.** D1 + D2 must reproduce the observed gain-jitter-minus-channel-dropout gap at SD 0.50 of
   **+2.81 pp**, to within about 0.3 pp. The same check at SD 0.40 gives -0.45 + 1.28 = +0.83 against an observed
   +1.01, so a residual of this order is expected from pairing noise. **If additivity fails badly, the
   decomposition is not valid and no outcome letter may be read.** That is outcome N in section 5.

---

## 4. Trainability

Mean-preserving channel dropout at SD 0.50 zeroes only 20% of channels and rescales, so it should train at least
as well as plain channel dropout at p = 0.5. If its no-augmentation-relative gain falls more than 3.0 pp below
`results_cd_rate_p0.5`'s, treat it as outcome X and report the lost cell rather than reading D1 and D2.

---

## 5. Pre-registered grid. Fixed before the run.

**On the criterion, and why the 2.0 pp floor is not used.** W-5, P-5 and P-6 all used a 2.0 pp materiality floor,
set against a run-to-run SD of about 0.5 pp. That floor is wrong here, because the quantity being decomposed is
itself only 2.81 pp, so a 2.0 pp threshold could only ever be cleared by one of the two components and an even
split would fall through into a null. **The criterion here is share of the observed gap, not absolute size.** A
component "carries the divergence" when it is Holm-significant **and** accounts for at least **60%** of the
+2.81 pp gap, which is at least 1.69 pp. Both components can be material without either carrying it, and that case
has its own cell.

| Outcome | Condition | Reading |
|---|---|---|
| **M. The mean shift carries it** | D1 Holm-significant and at least 60% of the gap | What breaks channel dropout at high dose is the collapse in expected activation, not the zeroing. At p = 0.5 half the signal is removed on average, and that, rather than the invariance, is the cost. **Channel dropout is then a variance injector with an unwanted side effect that grows with rate**, and the transferable recommendation is to use the mean-preserving form. |
| **F. The form carries it** | D2 Holm-significant and at least 60% of the gap | Discrete zeroing differs from continuous jitter once enough channels are being removed, independently of the moments. That is a genuine specificity result for channel dropout and the first the thesis would have. |
| **B. Both contribute** | Both Holm-significant, neither at 60% | The divergence is jointly produced. Report both shares and do not force a single property. |
| **N. Neither, or additivity failed** | Neither Holm-significant, or gate 4 fails | The decomposition does not hold at this dose. **This is a real possible result and it must be reported as one**, not smoothed over: it would mean the SD 0.50 divergence is not a sum of these two properties and something outside the design is producing it. |
| **R. Reversal** | The new arm falls below both comparators, or above gain jitter by more than 1.0 pp | Not anticipated. Check gate 1 before reading anything, then report the numbers without forcing a letter. |
| **X. Did not train** | Section 4 | Exclude, report the lost cell. |

**Every cell is publishable.** N and X included, because a decomposition that does not hold is information about
the design, and this project has twice been caught by grids that had no cell for that.

---

## 6. What this cannot say

State these in the verdict so nobody over-reads it later.

- Two doses is not a dose-response curve for either property. P-7 gives the decomposition at SD 0.40 and SD 0.50
  and nothing between or beyond.
- Nothing here bears on Section 4.8.2's primary finding, the 5.6-fold reduction in single-electrode occlusion
  cost, or on whether channel dropout works. It bears only on what happens at the top of the rate range.
- The deep model of record stays the channel-dropout `resnet_se` at 0.840 whatever this finds. It was fixed a
  priori, p = 0.2 is nowhere near the divergence, and switching cascades through eight downstream result sets.

---

## 7. Decision point, for Enam

Do not write any of this into the thesis. **The framing question in section 6.1 of
`EXPERIMENT_PLAN_CD_PARITY.md` was explicitly held pending this result**, because whether the perturbation ladder
is strong enough to name as a finding in its own right depends on whether the divergence is explained or merely
described. Report and stop.

---

## 8. Outputs

`results_p7_mpchandrop_resnet_se_sd0.50/` with the usual two CSVs. In `results_parity/`: `p7_divergence_tests.csv`,
`p7_verdict.md` and `p7_outcome.json`, following the shape of the P-6 files, with the outcome letter chosen by the
grid in section 5 and **the section 3 gate results printed in full**. Add a status header to the top of this file
recording the outcome, leaving the pre-registration below it unedited.

P-7 contributes two paired Wilcoxon tests to the whole-thesis family. **Report their raw p-values and do not touch
Section 4.17**; the family is recomputed in one pass afterwards, and it currently stands at 145 with about sixteen
additions already queued from W-5 and the parity stages.

---

## 9. What to report back

1. All four gates, with realized numbers.
2. D1 and D2 with intervals, effect sizes, Holm-corrected p-values, and each one's share of the +2.81 pp gap.
3. The same decomposition at SD 0.40 restated beside it, so the dose comparison is visible in one table.
4. The outcome letter, or an explicit statement that the result fell outside the grid.
5. The two raw p-values for the FDR family.
6. Confirmation that no thesis file was edited, Section 4.17 was not touched, and the deep model of record is
   unchanged.
