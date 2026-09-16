# EXPERIMENT_PLAN_DEEPCORAL.md: is the CNN-side "simple beats learned adaptation" result actually established, and does tuning overturn it?

**Why.** §4.7 and Table 4.10 carry one of the thesis's most quotable sentences: on the ResNet-SE+CD
backbone, "a one-line, label-free standardization outperforms both a learned covariance-alignment loss
and a parameter-free batch-statistics swap". The numbers behind it are per-subject normalization 0.840,
Deep CORAL 0.826, AdaBN 0.818, all from a 0.772 global-norm base.

Two things are wrong with how that claim currently stands, and they pull in opposite directions.

**First, it is untested.** The classical CORAL comparison in Table 4.9 carries Cohen's d of 1.13 and
0.95, p below 0.0001, BCa intervals and BH adjustment. The CNN-side comparison in §4.7, Table 4.10 and
Figure A.6 reports three level differences and nothing else: no Wilcoxon, no effect size, no interval,
no p-value. §3.16 puts GPU run-to-run nondeterminism at about 0.5 pp on the 40-fold mean, so a 1.4 pp
untested gap is under three of those. The thesis's strongest CNN-side rhetorical claim rests on a
difference that has not been shown to be reliable.

**Second, the competitor was not tuned.** §3.12.2 says the alignment weight was fixed at one and says
the comparison establishes superiority "at this fixed setting rather than at its best one". That
disclosure is honest and sits ahead of the result. The vulnerability is the asymmetry: classical CORAL's
regularizer was swept across four orders of magnitude (§4.7, 68.4% at 0.01 rising to 73.7% at 10) and
that sweep produced one of the better arguments in the thesis. Deep CORAL got no equivalent.

**D1 settles the first and is free. D2 settles the second and needs GPU. D2 is gated on D1.**

**Verified on disk.** Per-subject F1 vectors exist for all three arms, 40 rows each:

| arm | file | columns | published mean |
|---|---|---|---|
| per-subject norm | `results_cnn_aug_resnet_se_chandrop/cnn_arch_subjectwise.csv` | subject, arch, f1_macro, bal_acc | **0.8395** (SD 0.0671) |
| Deep CORAL, lambda = 1 | `results_deep_coral_chandrop/deep_coral_subjectwise.csv` | subject, arch, coral_lambda, f1_macro, bal_acc | **0.8257** (SD 0.0646) |
| AdaBN | `results_adabn_chandrop/adabn_subjectwise.csv` | subject, arch, f1_pre_adabn, f1_macro, bal_acc, delta_pp | **0.8182** (SD 0.0596), pre-adaptation 0.7874 |

`run_deep_coral_cnn_loso.py` takes `--coral-lambda` (default 1.0), `--arch resnet_se`,
`--augmentation chandrop`, `--epochs 40`, `--resume`, `--out`.

Shared paths per `EXPERIMENTS_README.md`:
```
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
```

Run from `06_Code/` in the project `.venv`. Do NOT fabricate numbers. Do NOT edit any thesis chapter.

---

# D1: the paired statistics that should already exist [RUN, CPU, no retraining]

## Phase 0. Reproduce the three means and check one arithmetic claim

1. Load the three subjectwise CSVs, align on `subject`, and confirm all three cover the same 40
   subjects with no duplicates.
2. **GATE:** the three means must come out at 0.8395, 0.8257 and 0.8182. If any misses, stop.
3. **Check one number the thesis states.** §4.7 says per-subject normalization "beats Deep CORAL by
   1.4 pp and beats AdaBN by 2.2 pp". The first is right: 0.8395 minus 0.8257 is 1.38 pp. The second
   looks wrong: 0.8395 minus 0.8182 is 2.13 pp, which rounds to 2.1 rather than 2.2. For paired data
   the mean of the differences equals the difference of the means, so there is no averaging route to
   2.2. Compute it from the per-subject vectors and report the value to three decimals. If it really
   is 2.1, say so: it is a one-character correction in §4.7 and I would rather have it found here than
   at a viva.

## Phase 1. The paired tests

Per-subject paired contrasts across the 40 subjects, on `f1_macro`:

- per-subject normalization against Deep CORAL at lambda = 1,
- per-subject normalization against AdaBN,
- Deep CORAL against AdaBN.

For each: the mean difference in pp, paired Wilcoxon signed-rank p, Cohen's d for paired data, a BCa
95% interval on the mean difference, and the number of subjects improving. Holm-correct within this
family of three.

Also report, from the AdaBN file, the **within-run** paired lift, `f1_macro` against `f1_pre_adabn`.
§4.7 already discloses that the +4.6 pp AdaBN figure is measured across runs while the paired lift
inside that execution was +3.1 pp. Confirm that number too, since it is the cleaner of the two and the
thesis may want to lead with it.

## Phase 2. Put the differences next to the noise

The point of this phase is to stop a p-value doing work it cannot do. §3.16 measures run-to-run
nondeterminism at about 0.5 pp on the 40-fold mean, with a median per-fold movement of 3.6 pp and a
worst case of 26.3 pp. Report each of the three mean differences as a multiple of that 0.5 pp figure,
and state plainly whether the paired interval excludes a difference of that size. A significant
Wilcoxon on a difference smaller than the training process's own variability is not evidence the
thesis should lean on.

## Gate D1, and it decides whether D2 runs at all

**Outcome 1, the 1.4 pp is significant with a moderate or large effect size and an interval that
clears the nondeterminism band.** Then the thesis's claim is sound and was merely unreported. Add the
statistics to §4.7 and Table 4.10 so the CNN side matches the rigour of the classical side. **Run D2**,
because a real 1.4 pp gap against an untuned competitor is exactly the gap an examiner will ask about.

**Outcome 2, the 1.4 pp is not significant, or is significant but sits inside the nondeterminism
band.** Then the honest claim is a near-tie, not a win, and §4.7, §5.8, §6.2 and the abstract all need
softening to something like "within run-to-run variation of, at a fraction of the cost". **D2 becomes
optional rather than necessary**, because tuning a competitor you have not shown yourself to be beating
answers a question nobody needs answered. Report and stop for a decision.

**Outcome 3, AdaBN and Deep CORAL are indistinguishable from each other.** Worth stating either way,
since §4.7 currently reports Deep CORAL as "a further 0.8 pp above AdaBN" as though that ordering
were meaningful.

**Check whether these tests are already in the correction family.** Read Appendix A.6's enumerated
composition of the 189-test Benjamini-Hochberg family and report whether the CNN-side adaptation
contrasts appear in it. If they do not, these are new members. If they do, something is inconsistent
between the family and the chapter text, and that needs saying.

---

# D2: the lambda sweep [RUN, GPU, only behind Gate D1]

## The prediction, fixed before the runs

This is not a fishing expedition. The classical sweep in §4.7 found that CORAL "performs better the
less second-order alignment it applies", with macro-F1 rising monotonically as the covariance
regularizer grew and the aligning transform tended toward the identity. Deep CORAL's `coral_lambda`
runs the other way: **larger lambda means more alignment**, and at lambda = 0 the method degenerates to
the global-normalization baseline it starts from.

So the over-alignment account predicts, specifically, that Deep CORAL's macro-F1 should **fall as
lambda rises** and **rise toward the global-norm baseline of about 0.772 as lambda falls toward zero**.
That gives the sweep a built-in sanity check and a bounded outcome: if Deep CORAL's best setting is a
very small lambda where it is barely aligning at all, it cannot overtake per-subject normalization by
construction, and the over-alignment story gains a third independent instance.

Record this prediction in the output before running, so the result is read against it rather than after
the fact.

## Phase 3. The sweep

Four new runs. Lambda = 1.0 already exists in `results_deep_coral_chandrop` and is the comparator.

```
foreach L in 0.01 0.1 10 100:
python run_deep_coral_cnn_loso.py --npz $NPZ --meta $META \
    --arch resnet_se --augmentation chandrop --coral-lambda $L --epochs 40 \
    --out results_deep_coral_lam$L --resume
```

**Reproduction gate first.** Before trusting any new arm, re-run lambda = 1.0 into a fresh directory
and check it lands within 1.5 pp of the published 0.8257. That bound is deliberately loose because it
has to absorb the 0.5 pp run-to-run nondeterminism; if the re-run misses by more than that, the harness
has drifted and the sweep is not comparable to the published arm.

Use `--resume`. These are 40 LOSO folds each on GPU, so treat each lambda as an independently
resumable job rather than one long chain.

## Phase 4. Analysis

1. Macro-F1 against lambda across all five settings, with the global-norm baseline and the per-subject
   figure of 0.8395 as horizontal references.
2. Paired Wilcoxon, Cohen's d and BCa intervals for the best lambda against per-subject normalization,
   and for each lambda against lambda = 1. Holm within the sweep family.
3. State explicitly whether the prediction above held: did macro-F1 fall as lambda rose, and did the
   smallest lambda approach the global-norm baseline.
4. Figure `report_figs/new_experiments/deep_coral_lambda.png`, matching the style of the classical
   CORAL sweep figure so the two can be read side by side. That pairing is the point: it would show the
   same over-alignment shape inside two different methods' own hyperparameters.

## Escalation rule, and it is firm

**If any lambda beats 0.8395 by more than 1.5 pp with a paired test that survives Holm, stop and
report.** Do not rewrite anything, do not soften anything, do not run further arms. That result would
change what the thesis claims about simple versus learned adaptation, which is Enam's decision and not
the runner's. Report the number, the test, and which lambda.

If the best lambda merely narrows the gap without overtaking, that is the expected case and needs no
escalation: it sharpens §4.7 from "beats Deep CORAL at a fixed setting" to "beats Deep CORAL at its
best setting among those tested", which is a strictly stronger claim than the thesis currently makes.

---

## After the runs: hand back for the write-up cascade

Leave CSVs and PNGs. Do not edit chapters. Report, in this order:

1. Whether D1's Phase 0 gate reproduced the three means.
2. The value of the AdaBN difference to three decimals, and whether §4.7's "2.2 pp" should be 2.1.
3. The three paired tests with effect sizes and intervals, and each difference as a multiple of the
   0.5 pp nondeterminism figure.
4. Which D1 outcome occurred, in one sentence, and therefore whether D2 was warranted.
5. Whether the CNN-side contrasts already appear in Appendix A.6's family.
6. If D2 ran: the reproduction check on lambda = 1, the five-point sweep, whether the pre-registered
   prediction held, and whether the escalation rule fired.

---
---

# D2 AMENDMENT, 15 September 2026. This section supersedes the original D2 above.

**Read this section, not the original D2.** The original was written on 10 September, before D1 ran.
D1 changed the premise it rested on, and two of its rules are now wrong rather than merely stale. The
original text is kept above for the audit trail and must not be followed.

## Why D2 is being run at all, when D1 said it was optional

D1 resolved to **Outcome 2**. Per-subject normalization beats Deep CORAL at lambda = 1 by 1.378 pp
with a BCa interval of [0.15, 2.40], which does not clear the 0.5 pp run-to-run nondeterminism band of
§3.16. The thesis therefore reports a near-tie rather than a win, and the original plan's Gate D1 made
D2 optional on the reasoning that tuning a competitor you have not shown yourself to be beating
answers a question nobody needs answered.

That reasoning was sound about the science and wrong about the viva. It is the reverse that now holds:
because the margin is a near-tie, an untuned competitor is a **worse** defence than it would have been
against a clear 1.4 pp win. The asymmetry is the exposed part. §4.7 swept the classical CORAL
regularizer across four orders of magnitude and built an argument out of the shape of that sweep;
the deep side got one setting. §3.12.2 discloses the asymmetry honestly and calls it one of compute
rather than intent, which is true and is also the answer an examiner is least likely to accept when
the resulting margin is inside the noise band. Every other item on the remediation list is closed.
This is the last open question that could move a conclusion, and it is answerable in a day of GPU.

Enam's decision, 15 September: run it.

## The one thing that must not be got wrong

**The per-subject comparator is `results_cnn_aug_resnet_se_chandrop/cnn_arch_subjectwise.csv`, mean
0.839490.** It is NOT `results_cnn_aug_resnet_se_chandrop_proba`, mean 0.841566. The two differ by
0.21 pp because they are different repeat runs of the same arm. Only 0.839490 rounds to the 0.840 that
Table 4.21 and Figure A.6 disclose, §4.7 reports D1's figures, and the whole-thesis correction family
was re-homed to a D-1 family in v6 precisely to fix two rows that had been sourced from the wrong one.
Sourcing this sweep from `_proba` would silently reintroduce the defect v6 was opened to remove.
Verify the mean to six decimals before computing a single contrast, and abort if it is not 0.839490.

## Corrected understanding of the prediction

The original plan's prediction sentence is confusing and, read literally, self-contradictory. State it
correctly before running, because the sweep's interpretation depends on it.

Deep CORAL's `coral_lambda` scales the alignment term, so larger lambda means more alignment. At
lambda = 0 the method degenerates to the global-normalization baseline it starts from, which scores
about 0.772. At lambda = 1 it scores 0.8257. So performance **rises** from lambda 0 to lambda 1. The
over-alignment account of §4.7.2 predicts it must **fall** again once alignment is pushed far enough,
because alignment eventually removes class-discriminative covariance along with subject variance.

The prediction is therefore **an interior maximum**, not a monotone curve, and the open question is
where it sits. If the peak is at or below lambda = 1, the thesis's comparison was fair by luck and the
over-alignment story gains a third independent instance. If the peak sits above lambda = 1, Deep CORAL
has room the thesis never gave it, and the size of that room is what this experiment measures.

Write this prediction into the output directory as `PREDICTION.md` before the first run starts, with a
timestamp, so the result is read against it rather than after it.

## Phase D2.0. Reproduction gate, and it runs first

Re-run lambda = 1.0 into a fresh directory before any new setting. This is not ceremony. Every
conclusion here compares new arms against a per-subject arm trained weeks ago; if the harness, the
torch build or the CUDA stack has moved since, a shift in the curve would be drift misread as an
effect.

```
python run_deep_coral_cnn_loso.py --npz $NPZ --meta $META \
    --arch resnet_se --augmentation chandrop --coral-lambda 1.0 --epochs 40 \
    --out results_deep_coral_lam1p0_repro --resume
```

**GATE.** The re-run mean must land within 1.5 pp of 0.825712. The bound is deliberately loose because
it absorbs the 0.5 pp nondeterminism on the 40-fold mean. If it misses by more, **stop and report**.
The sweep is not comparable to the published arm and nothing further should be spent.

If it passes, record the offset. Every later arm inherits it, and the write-up needs to know whether
the sweep sits a few tenths above or below the published lambda = 1 figure.

## Phase D2.1. Time one fold before committing to the grid

There is no timing data for this script anywhere in `06_Code`, so the cost of the sweep is currently
unknown rather than estimated. Do not start four arms blind.

Run a single fold first, `--heldout 1`, into a scratch directory, and time it. Multiply by 40 for one
arm and by 4 for Stage A. **Report the projection before proceeding.** If Stage A projects beyond
about eight hours of GPU, stop and report rather than starting it; the grid can be cut to lambda 0.1
and 10 and still bracket the peak, and that is Enam's call to make, not the runner's.

Note for the projection: `--epochs 40` is a ceiling, not a count. Early stopping is on the source
validation loss with `--patience 7`, so folds finish early and the first fold is an upper bound rather
than an average.

## Phase D2.2. Stage A, three new arms

```
foreach L in 0.1 10 100:
python run_deep_coral_cnn_loso.py --npz $NPZ --meta $META \
    --arch resnet_se --augmentation chandrop --coral-lambda $L --epochs 40 \
    --out results_deep_coral_lam<L> --resume
```

Every lambda gets its **own** `--out` directory. `--resume` reads the subjects already present in that
directory's `deep_coral_subjectwise.csv` and skips them, so pointing two lambdas at one directory would
make the second silently skip all forty folds and inherit the first one's numbers. Name the directories
`results_deep_coral_lam0p1`, `results_deep_coral_lam10`, `results_deep_coral_lam100`.

Treat each lambda as an independently resumable job. Do not chain them behind one shell. Read
`deep_coral_subjectwise.csv` for analysis, never `deep_coral_summary.csv`, which rounds to four
decimal places.

## Phase D2.3. Stage B, conditional and pre-registered

Stage A brackets the peak only if performance has clearly turned over by lambda = 10.

**Trigger: if lambda = 10 lands within 0.5 pp of lambda = 1, or above it, the peak is not bracketed.**
Then, and only then, run lambda = 3 and lambda = 30 to locate it. If lambda = 10 sits clearly below
lambda = 1, the curve has turned over, the peak is at or below 1, and Stage B is not run.

This trigger is fixed now so that the decision to spend more GPU is made by the data rather than by
how the result is going.

## Phase D2.4. Analysis

1. Macro-F1 against lambda across every setting run, with two horizontal references: the global-norm
   baseline at about 0.772 and per-subject normalization at 0.839490.
2. Paired contrasts across the 40 subjects on `f1_macro`, aligned on `subject`:
   - each new lambda against per-subject normalization,
   - each new lambda against lambda = 1,
   - the best lambda against AdaBN, from `results_adabn_chandrop/adabn_subjectwise.csv`, mean 0.8182.
   For each: mean difference in pp, paired Wilcoxon signed-rank p, paired Cohen's d, BCa 95% interval,
   and the number of subjects improving. Holm-correct within the sweep.
3. Express every mean difference as a multiple of the 0.5 pp nondeterminism figure, and say plainly
   whether each BCa interval excludes a difference of that size. A Wilcoxon p on a difference smaller
   than the training process's own variability is not evidence and must not be reported as though it
   were.
4. State whether the interior-maximum prediction held, and where the peak sits.
5. Figure `report_figs/new_experiments/deep_coral_lambda.png`, matching the style of the classical
   CORAL sweep figure so the two read side by side. That pairing is the point of the experiment beyond
   its defensive value: the same over-alignment shape appearing inside two different methods' own
   hyperparameters.

## Phase D2.5. The correction family

The whole-thesis Benjamini-Hochberg family stands at **218 tests, 153 survivors** in
`recompute_unified_fdr_v7.py`. This sweep adds members and the family must be recomputed as **v8**.

Admission follows the rule fixed in P2.5 and applied in v7: **genuine tests only.** A contrast between
two bit-identical arms is not a test, and an exact duplicate of a contrast already counted is not a
new one. Neither exclusion should arise here, since every arm is a distinct training run, but apply
the test rather than assuming.

Admit every lambda-against-per-subject contrast and every lambda-against-lambda-1 contrast, one member
each. Do **not** admit only the best lambda's contrast: choosing the arm after seeing the results and
then correcting as though one test had been planned is exactly the selection the family exists to
price. Record the composition and any exclusion in Appendix A.6's exclusions paragraph and in the v8
docstring, in the same form v7 used.

Report whether anything crosses the boundary in either direction. Two contrasts already sit close to
it, the Random Forest's ENABL3S replication and AdaBN's pre-to-post contrast on the same cohort, both
at raw p 0.037 and at an adjusted 0.052 in a family of 218. Growing the family moves them further out,
not back in, but say so explicitly rather than leaving it inferred.

## The decision rules, fixed before the runs

The original plan's escalation threshold was "any lambda beats 0.839490 by more than 1.5 pp". **That
threshold is now wrong and is replaced.** It was set when the per-subject margin was believed to be a
real 1.4 pp win. After D1's Outcome 2 the margin is a near-tie, so an overtake of 1.0 pp would already
invert the direction of the thesis's CNN-side claim while sitting comfortably under the old trigger.
The threshold is therefore the 0.5 pp nondeterminism band that the rest of the thesis uses.

**Outcome A. The best lambda stays below per-subject normalization by more than 0.5 pp, with a paired
test that survives Holm.** The expected case. The claim strengthens from "outperforms Deep CORAL at a
fixed setting" to "at its best setting among those tested", which is strictly stronger than anything
the thesis currently says, and the over-alignment account gains its third instance. No claim inverts.
Write it up as a straightforward hardening.

**Outcome B. The best lambda closes the gap to inside the 0.5 pp band.** Then per-subject
normalization and a tuned Deep CORAL are indistinguishable on this backbone, and the honest claim is
"matches a tuned learned alignment at a fraction of its cost", which is still a good result and a more
defensible one than the present wording. The abstract's "edges its deep variant by about a point" must
then come out, along with the corresponding sentences in §4.7, §5.8 and §6.2. **This is a change to
what the thesis claims. Report and stop. Do not edit any chapter.**

**Outcome C. The best lambda overtakes per-subject normalization by more than 0.5 pp with a paired
test that survives Holm.** Then the CNN-side comparison inverts and the thesis says the opposite of
what it now says on that point. **Stop immediately. Report the lambda, the margin, the interval and the
test. Do not rewrite, do not soften, do not run further arms.** This is Enam's decision and not the
runner's, and it is the outcome the experiment was run to expose rather than to avoid.

Note for all three outcomes: the normalization finding itself does not depend on this. It rests on
consistency across three model families, on the classical arm where CORAL was swept, on the alignment
ladder, on ENABL3S and on the active-only control. Outcome C would narrow one sentence about one
backbone. Nothing here can touch the spine of the thesis, and the runner should not write as though it
could.

## Operating constraints

- Run from `06_Code/` in the project `.venv`. Absolute path for the inner python.
- Do **not** edit any thesis chapter, `MSc Thesis.docx`, the remediation tracker or the handoff.
  Leave CSVs, the PNG and a written report. The write-up cascade is a separate pass.
- Do **not** fabricate numbers. If an arm does not finish, report it unfinished.
- Do **not** delete or overwrite `results_deep_coral_chandrop`. It is the published lambda = 1 arm and
  the reproduction run goes to a new directory.
- `--xkey` stays at its default `X_env`. The deep arm reads the centred envelope, which P3.1 confirmed
  is the stage that still looks forward on the convolutional side. That is the published configuration
  and this experiment is a perturbation study on lambda alone, not an occasion to change it.
- Seed stays at 42. Do not add repeats per lambda; the nondeterminism band is already characterised in
  §3.16 and is what the decision rules are written against.

## Report back, in this order

1. The reproduction gate: the lambda = 1 re-run mean, its offset from 0.825712, pass or fail.
2. The single-fold timing and the projection you made from it.
3. The comparator check: the per-subject mean to six decimals, and that it came from
   `results_cnn_aug_resnet_se_chandrop`.
4. The sweep table: lambda, mean, SD, n, for every arm run.
5. Whether Stage B triggered, and why.
6. The paired contrasts with effect sizes, intervals, subjects improving, and each difference as a
   multiple of 0.5 pp.
7. Whether the interior-maximum prediction held and where the peak sits.
8. Which outcome fired, A, B or C, in one sentence.
9. The v8 family: new member count, total, survivors, and anything that changed side.

---
---

# D2b AMENDMENT, 16 September 2026. The comparator control, and why D2 cannot be read without it.

## What D2 returned, and the part of it that is sound

Six arms, 240 folds. Lambda is inert: across 0.1 to 100 every arm sits within 0.3 pp of every other,
every lambda-against-lambda-1 contrast is at most 0.18 pp at p no better than 0.26, and the
pre-registered interior maximum did not appear. **That half of the result needs no control and is
already the answer to the objection D2 was run for.** Deep CORAL was not handicapped by the published
lambda = 1, because nothing in three orders of magnitude beats it. Those five contrasts are all
measured inside a single harness, so they compare like with like.

## The part that cannot be read yet

The report resolves **Outcome B** on the best lambda sitting 0.479 pp below per-subject normalization.
That single contrast compares arms trained on 15 and 16 September against a per-subject arm trained
weeks earlier. The reproduction gate measured what that costs and the answer was not small:

| term | pp | share of the closure |
|---|---|---|
| reproduction offset: lambda = 1 re-run against lambda = 1 published | +0.721 | **80%** |
| lambda effect: best lambda = 30 against lambda = 1 re-run | +0.178 | 20% |

The closure from D1's 1.378 pp to the reported 0.479 pp is 0.899 pp, and four fifths of it is the
same configuration re-run rather than anything lambda did. The remaining fifth carries p = 0.90.

The offset is broad rather than local. Paired across the 40 subjects, the re-run beats the published
arm on 26 of 40, the median move is +0.632 pp, dropping the three largest movers leaves +0.635 pp,
and the paired Wilcoxon sits at p = 0.082. A few unstable folds would show as a large mean with a
median near zero. This does not. At about 1.5 times the 0.47 pp figure of §3.16 it is a large draw
if it is a draw at all, and it is equally consistent with the environment having moved since the
published deep arms were trained.

**The gate did its job and the conclusion then ignored it.** The 1.5 pp bound was set to catch gross
drift, not to license carrying a 0.72 pp offset into a contrast whose residual is 0.479 pp.

## D2b: re-run the per-subject arm under today's harness

One arm, 40 folds, roughly 2.2 hours at the realised 200 s per fold. Same script and configuration
that produced `results_cnn_aug_resnet_se_chandrop`: ResNet-SE backbone, channel-dropout augmentation,
per-subject normalization, seed 42, `--epochs 40`. Into a **fresh** directory
`results_persubj_chandrop_repro`. Do not touch `results_cnn_aug_resnet_se_chandrop`.

The producing script is `run_cnn_arch_loso.py`, not `run_deep_coral_cnn_loso.py`. It is the script
that writes `cnn_arch_subjectwise.csv`, and the directory naming follows the convention
`results_cnn_aug_resnet_se_<augmentation>` that `g8_resnet_se_augmentation_stats.py` reads. That
directory has no `run_config.json` and `EXPERIMENTS_README.md` does not record the invocation, so it
is reconstructed from the script's own defaults:

```
python run_cnn_arch_loso.py --npz $NPZ --meta $META \
    --arch resnet_se --norm-mode per_subject --augmentation chandrop --epochs 40 \
    --out results_persubj_chandrop_repro --resume
```

Everything else stays at the script's defaults, which are the published settings: `--xkey X_env`,
`--batch 512`, `--lr 1e-3`, `--patience 7`, `--val-frac 0.15`, `--seed 42`, `--aug-chandrop-p 0.2`.
Note that `--batch` defaults to 512 here and to 256 in the Deep CORAL script; that difference is
between the two arms of record as published and must be preserved, not harmonized.

Print the command before running it. If anything in `EXPERIMENT_PLAN_CHANDROP.md` or elsewhere
contradicts the reconstruction, stop and report rather than choosing.

## Reading the result. These rules are fixed now.

Let `R` be the re-run per-subject mean and `0.839490` the published one.

**Outcome A1. R rises by roughly the same offset, about +0.5 pp or more.** Then the environment has
moved and both arms moved with it. Recompute the gap as R minus the best lambda, both measured this
week. If that gap exceeds 0.5 pp, the original claim stands and **D2 resolves to Outcome A**: the
thesis hardens from "outperforms Deep CORAL at a fixed setting" to "at its best setting among those
tested, and lambda is inert across three orders of magnitude", which is strictly stronger than the
present wording and requires no softening anywhere.

**Outcome B1. R holds near 0.839490, within about 0.2 pp.** Then the offset was specific to the Deep
CORAL runs or was a genuine high draw, the like-for-like gap really is about 0.5 pp, and **D2's
Outcome B stands on a fair comparison.** The abstract's "edges its deep variant by about a point"
comes out and §4.7, §5.8 and §6.2 soften to indistinguishability at a fraction of the cost.

**Outcome C1. R falls, or rises by more than about 1.5 pp.** Neither is expected. Stop and report;
something has changed in the environment beyond run-to-run variation and the deep arms of record may
need revisiting, which is a much larger question than this experiment.

In all three cases the within-harness finding that lambda is inert is unaffected and is reported as
established.

## The family

Do **not** recompute the family for D2b. The re-run is a control on an existing comparator, not a new
contrast, and admitting "the same arm against itself in a different week" would put a non-test into a
correction whose purpose is to price genuine tests, which is the rule v7 applied to the filter chain's
32 offered rows. If the outcome is A1, the lambda-against-per-subject contrasts already in v8 are
computed against the wrong comparator and must be **recomputed in place as v9 against R**, with the
substitution recorded in the v9 docstring and in Appendix A.6. Do not add members for it.

## Operating constraints

Unchanged from the D2 amendment. Run from `06_Code/` in the project `.venv`, do not edit any chapter
or the tracker or the handoff, do not fabricate numbers, do not delete or overwrite any published
results directory, seed stays at 42, no repeats.

## Report back

1. The command you ran and where you took it from.
2. R to six decimals, n, SD.
3. R minus 0.839490 in pp, paired across subjects: mean, median, subjects improved, Wilcoxon p, and
   the mean after dropping the three largest movers by absolute delta. This is the same diagnostic
   that exposed the Deep CORAL offset and it is what distinguishes drift from a noisy draw.
4. The like-for-like gap: R minus the best lambda arm, paired, with BCa interval, Cohen's dz and
   subjects improved, and as a multiple of 0.5 pp.
5. Which outcome fired, A1, B1 or C1, in one sentence.
6. If A1: the v9 recomputation and whether anything changed side.
