# Plan: audit remediation B1 to B4 (plus B5 hygiene)

> **RUN 3 September 2026 (B1, B2, B3, B4, B5). B8 / section 4A queued next as RUN_QUEUE Block 3.**
> - **B1 manifests.** `build_run_manifest.py` -> `RUN_MANIFEST.csv` (143 dirs). Per-field
>   provenance: 0 dirs fully `verified`, **142 carry at least one `UNKNOWN` core field** because no
>   run ever wrote a config file. Cross-check vs the handoff Section 4 table:
>   **the occlusion pair (`results_g3_noaug_instr`, `results_cd_resnet_nose_chandrop`) is the ONLY
>   backbone mismatch** - both verify as `--arch resnet`, three passages imply `resnet_se`. P-9
>   (backbone S) shows the reduction transfers, so the number stands and only the wording needs
>   the backbone named (write-up queue). Six other canonical dirs have an unverifiable backbone
>   (no `arch` recorded) but no contradiction. **Section 1.5 done:** `run_config_dump.py` added;
>   `run_cnn_arch_loso.py`, `train_cnn_loso.py` and `train_classical_loso.py` now write
>   `run_config.json` (args, resolved paths, git commit, lib versions), guarded and RNG-inert
>   (asserted; unit test + CNN smoke pass; `p5p6_inertness.py` still PASS). Report:
>   `B1_MANIFEST_REPORT.md`.
> - **B2 literature band. The 70 to 85 percent band does not populate.** No study in
>   `05_Sources/Literature/` meets all four inclusion criteria (true LOSO, lower limb, ~4 classes,
>   ~40 subjects). The two papers cited for the upper bound: Ding et al. 2024 is *calibrated
>   few-shot* 4-way accuracy (best 85.1% at 10 shots, max not mean), Wang et al. 2024 is
>   *upper-limb* hand gestures (81.7% accuracy). Genuine true-LOSO EMG accuracy is typically 30 to
>   55 percent, reaching low 80s only with HD-EMG or calibration. The lower anchor (~70 to 78
>   percent) is sound. **Metric:** the comparators report accuracy, the thesis reports macro-F1,
>   which on this data (STDUP 55.7 percent of windows) is the harder metric, so the comparison is
>   conservative and that is currently unstated. **Section 2.5 is a decision point for Enam** -
>   reported, not resolved. Table: `04_Reviews_and_QA/LITERATURE_BAND_TABLE.md`.
> - **B3 five-point correlations. Section 3.2 WORKS.** `b3_per_subject_silhouette.py`: per-subject
>   class silhouette across the 5 rungs, 40x5. Rung-3 peak resolved paired: rung 3 vs rung 4
>   (full whiten) +0.038, d = 1.91, p = 1.8e-12, **40/0**; rung 3 vs rung 0 +0.007, d = 0.78,
>   p = 9.6e-6. Within-subject Spearman(silhouette, F1) over the 5 rungs: 40/40 defined, mean rho
>   +0.44, median +0.46, 35/4 positive, Wilcoxon vs 0 p = 1.3e-6. The ordering holds *within
>   subjects*, tested. **Section 3.4 is a decision point for Enam** (add the test / replace the
>   correlations / report both). New paired Wilcoxon: rung3-vs-rung4 silhouette p = 1.8e-12,
>   rung3-vs-rung0 silhouette p = 9.6e-6. Table 4.16 note recorded (MMD "n/a" vs 0.0). Report:
>   `results_locus/b3_verdict.md`.
> - **B4.1 alignment ladder.** `b4_regen_ladder.py`: `alignment_ladder_full.csv`,
>   `..._summary.csv`, `..._stats.csv` regenerated complete for all 5 rungs from the per-subject
>   files, which reproduce Table 4.16 (0.7094 / 0.7482 / 0.7186 / 0.7767 / 0.6752) exactly.
>   Originals in `_ARCHIVE/prebackups/b4_ladder_20260903/`. New paired Wilcoxon contrasts
>   rung1/2/4_vs_rung0 (p = 2.0e-5 / 0.022 / 1.2e-3) and rung4_vs_rung3 (p = 3.6e-12, d = -2.34).
>   Geometry columns untouched.
> - **B4.2 the 400 ms ladder. Section 4.5 is a decision point for Enam.**
>   `results_win400_ladder/alignment_ladder_full.csv` carries the 250 ms geometry columns verbatim
>   (byte-identical mmd/w1/probe/silhouette) while its `f1_macro_mean` is genuinely 400 ms. The
>   summary and stats files there are already complete. **Recommendation: delete the four stale
>   geometry columns and keep the F1**, because nothing in the thesis quotes the 400 ms ladder
>   geometry, recomputing it costs CPU on numbers no claim needs, and the delete is one edit that
>   removes the misleading copy. Not executed; awaiting Enam.
> - **B4.3 / B5 em dashes.** 59 em dashes removed from `README.md` (29), `REPRODUCE.md` (24),
>   `_STRUCTURE.md` (6), meaning preserved (spaced hyphen). `VIVA_DEFENSE_NOTES.md` (~50, not
>   public, lower priority) deferred per the plan.
> - **B4.4.** `ensemble_v2_combine.py`: `subject_probs` `min()` -> assertion that member
>   probability files are row-aligned per subject (verified it does not fire); `ensemble_v2_
>   subjectwise.csv` now written with an explicit `subject` column, and the three published
>   `results_ensemble_v2*/` copies were patched with the column (positional, 1..40).
> - **Hygiene.** Bare `except Exception` in `build_full_dataset` (`preprocess_emg.py`) now
>   re-raises real failures with context and records tolerated missing-file skips.
> - No thesis file edited. Section 4.17 not touched. Deep model of record unchanged. Plan below
>   is unedited.

**Status:** ready to run. Written 3 September 2026, from `04_Reviews_and_QA/METHODOLOGY_AUDIT_3SEP.md`.
**Execution:** local, on Enam's machine. **Queue this after the locus programme finishes.**
**Cost: B1 to B4 need no GPU. B8 (section 4A) needs well under an hour of GPU for the CNN arm and no LOSO.** B1 and B4 are scripting, B2 is literature work, B3 is a CPU analysis.
Budget 3 to 5 hours, most of it B2.
**Owner decision points: three**, at 2.5, 3.4 and 4.5. Do not resolve any of them yourself.

---

## 0. Read this first

1. **The Python interpreter is `06_Code/.venv/Scripts/python.exe`.** The `jobs_*.txt` files are stale.
2. **Do not edit any thesis file in B1, B2 or B4.** B3 produces a recommendation, not an edit. All thesis edits
   from this plan go through Enam and into the deferred write-up queue.
3. **Do not touch Section 4.17.** B3 may add paired tests; report their raw p-values and let the family be
   recomputed in one pass afterwards.
4. **A guessed provenance is worse than a missing one.** B1's single most important rule is in 1.3. Read it before
   writing any code.

---

## 1. B1: run manifests. The root cause of the backbone defect.

### 1.1 The problem

Across roughly 123 `results_*` directories there is **not one configuration file**, and neither
`run_cnn_arch_loso.py` nor `train_classical_loso.py` writes its arguments. `cnn_arch_summary.csv` records `arch`
and nothing else: no normalization mode, no augmentation or rate, no window, no seed, no dataset. Provenance
lives in directory naming, the plan markdown and `REPRODUCE.md`.

That is how the 84.2 pp to 15.0 pp occlusion result, measured on the SE-free `resnet`, came to be attributed to
the `resnet_se` model of record in three passages. **The repository is public with a minted DOI**, so a third
party reproducing has to infer configuration from folder names.

### 1.2 Build `build_run_manifest.py`

Walk every `results_*` directory and emit `06_Code/RUN_MANIFEST.csv` with one row per directory:

`dir, dataset, backbone, norm_mode, augmentation, aug_param, window_ms, seed, n_subjects, n_rows, subject_id_set_hash, source_of_config, confidence, mtime_earliest, mtime_latest`

### 1.3 The rule that matters: three confidence states, never a guess

Every config field carries one of exactly three provenance states, and **the state is recorded per field**:

- **`verified`** — read from the data itself. `arch` from `cnn_arch_summary.csv`; window and feature set from the
  npz filename embedded in classical subjectwise filenames; `n_subjects` and the subject-ID set from the
  subjectwise file; dataset from the subject-ID range (1 to 40 SIAT, 156 and 185 to 194 ENABL3S).
- **`documented`** — stated in a plan file or `REPRODUCE.md`. **Record the file and line that says so**, in
  `source_of_config`. A claim with no citable line is not `documented`.
- **`unknown`** — neither. **Write `UNKNOWN`. Do not infer from the directory name.** Directory names are a
  convention, not a record, and inferring from them is exactly the reasoning that produced the defect.

**A manifest that guesses is worse than no manifest, because it looks authoritative.** If most fields for a
directory come back `unknown`, that is the finding: that directory cannot safely back a claim without someone
re-deriving its provenance by hand.

### 1.4 The cross-check, which is the point of the exercise

Take every quantity in the handoff's Section 4 canonical numbers table, which names a source directory for each.
For each, compare the manifest's config against what the thesis says about that number. **Report every mismatch
and every `unknown` on a directory that backs a canonical number.**

Expect at least one hit: the occlusion pair is already known to be `resnet` while three passages imply
`resnet_se`. **If that is the only hit, say so explicitly**, because a clean sweep here is the assurance Enam
asked for. If there are others, they are the same class of defect and each needs its own note.

### 1.5 Stop future recurrence

Add an args dump to `run_cnn_arch_loso.py` and `train_classical_loso.py`: write `run_config.json` into the output
directory with `vars(args)`, the resolved input paths, the git commit, and the torch and cuDNN versions. **Follow
the `gainjitter` precedent**: guarded, additive, and proven inert with byte-identical RNG-state and
initial-parameter assertions on all existing modes before anything else runs. Writing a file after the run cannot
perturb training, but assert it anyway, because that is the house rule.

---

## 2. B2: the literature band. The most examinable soft spot in the performance framing.

### 2.1 The problem

The 70 to 85% band appears in **§2.3.2, §5.1.1 and §6.2**, and the positioning claim is that the 85.8% headline
"sits at the upper edge of that competitive band".

§2.3.2 anchors the lower part properly, at "roughly 70-78% macro F1" for unadapted classical pipelines, metric
named. **The upper bound of 85 has no named source and no stated metric.** The band is asserted in a clause
citing Ding et al. (2024) and Wang et al. (2024), with no enumeration of studies, cohort sizes, class counts,
protocols or metrics. So the frame for the engineering result turns on 0.8 of a percentage point measured against
a boundary with no provenance, and the thesis's own figure is macro-F1 while the surrounding discussion moves
between accuracy and F1.

### 2.2 Build the enumeration

From `05_Sources/Literature/` first, then the open web for anything missing. For every study that could plausibly
sit in the band, record: **author, year, dataset, n subjects, n classes, limb, protocol (true LOSO or not),
metric (accuracy or macro-F1 or other), and the reported value.**

**Inclusion is the whole point, so state the rule before collecting:** true leave-one-subject-out (not k-fold, not
leave-one-session-out), lower limb, and a cohort and class count in the same region as 40 subjects and 4 classes.
**Record the studies you exclude and why**, because the exclusions are what make the band defensible.

Retrieve each in full. **Anything not retrievable in full is cited only for what its title and abstract state**,
per the Jing (2022) precedent in the handoff's correction log.

### 2.3 Then answer the question honestly

With the table built, does the band hold?

- If the enumerated studies span roughly 70 to 85 on a consistent metric, the claim is vindicated and the table
  goes into §2.3.2 as its evidence.
- **If they do not, the band changes.** That is a real possible outcome. It might narrow, widen, split by metric,
  or turn out to be two bands, one for accuracy and one for macro-F1. Any of those is reportable and none is a
  disaster; what is not acceptable is keeping a number the enumeration does not support.

### 2.4 The metric question, separately

Determine whether the band as used is accuracy or macro-F1. **If the enumerated studies mostly report accuracy
while the thesis compares macro-F1, say so plainly.** On this dataset, with STDUP at 55.7% of windows, the two
differ materially, and macro-F1 is the harder metric, so the comparison is conservative rather than flattering.
That is a point in the thesis's favour **and it currently goes unstated**, which is the worse of the two errors.

### 2.5 Decision point, for Enam

Report the table and the verdict. **Do not rewrite §2.3.2, §5.1.1 or §6.2.** Whether the fix is to insert the
table, to narrow the claim to what is anchored, or to restate the comparison on a consistent metric is Enam's
call, and it touches three sections including the Conclusion.

---

## 3. B3: the five-point correlations in Section 4.13.2

### 3.1 The problem

§4.13.2 states that "macro-F1 tracks class separability at a Spearman correlation of 0.90, while its correlation
with the subject-identity probe is 0.10 and with the proportion of between-subject discrepancy removed is -0.10".

All three reproduce exactly. But **n = 5 rungs**. The 95% interval on rho = 0.90 at n = 5 runs about
**[0.09, 0.99]**, and the three values are not distinguishable from one another. They also sit outside the FDR
family. The finding survives because the argument rests on the ordering, and §4.13.2 says so itself, but the
correlations are presented as a discriminating contrast that at this n they cannot be.

### 3.2 The substantive fix, and it is free

**Compute class silhouette per subject at each rung, not once on the pooled feature space.** That turns a 5-point
descriptive correlation into a 40-by-5 paired analysis:

- For each subject and each of the five rungs, compute the silhouette of the four movement classes within that
  subject's own windows in the 72-dimensional feature space, after that rung's operator.
- Test the rung-3 peak directly: **paired across the 40 subjects, is silhouette at mean-and-scale higher than at
  full whitening, and higher than at global?** Those are two paired Wilcoxon tests with real n.
- Then, per subject, correlate the five silhouettes against that subject's five macro-F1 values, and report the
  distribution of those 40 within-subject correlations against zero.

**That is the difference between "the ordering looks right across five points" and "the ordering holds within
subjects, tested".** It would materially strengthen the supporting argument for Finding A, which is the thesis's
headline, and it needs no GPU and no new runs. The per-subject F1 values are already in
`results_alignment_ladder_loso/ladder_loso_{0..4}_SVM_subjectwise.csv` and the features are on disk.

### 3.3 The fallback if 3.2 does not work

If per-subject silhouette is too noisy at this window count, or the within-subject correlations are
uninterpretable, **report that and stop**. The presentational fix then applies: state n = 5, mark the three
correlations descriptive rather than inferential, and let the ordering carry the argument as the section already
says it does. Also fix the small inconsistency that Table 4.16 shows MMD removed as "n/a" at the baseline while
the correlation treats it as 0.0.

### 3.4 Decision point, for Enam

If 3.2 works, whether §4.13.2 gains the per-subject test, replaces the correlations with it, or reports both is
Enam's call. Report the numbers and stop. Note the new paired tests and their raw p-values for the family; **do
not touch §4.17**.

---

## 4. B4: incomplete and wrong result files in a public repository

### 4.1 The alignment ladder

`results_alignment_ladder_loso/alignment_ladder_full.csv` carries `f1_macro_mean` for **rungs 0 and 3 only**;
rungs 1, 2 and 4 are blank. `alignment_ladder_loso_summary.csv` holds only those two rungs.
`alignment_ladder_loso_stats.csv` holds only the rung3-vs-rung0 contrast.

**The thesis is correct** and Table 4.16 reproduces exactly from the five per-subject files, whose means are
0.7094, 0.7482, 0.7186, 0.7767, 0.6752. But a reader reproducing Finding A from the public repository finds
blanks where the thesis prints numbers.

Regenerate all three files from the per-subject data, complete for all five rungs, and **verify the regenerated
values match Table 4.16 before overwriting anything**. Back up the originals first.

### 4.2 The 400 ms ladder, already known and still unfixed

`results_win400_ladder/alignment_ladder_full.csv` carries the **250 ms geometry columns verbatim** in a 400 ms
file: `mmd_removed_pct`, `w1_removed_pct`, `subject_probe_bal_acc` and `silhouette_by_class` are byte-identical to
the 250 ms file while `f1_macro_mean` is genuinely 400 ms. Nothing in the thesis quotes it. Either recompute the
geometry at 400 ms or delete the four columns and leave the F1. **Do not leave it as is**; it has been in a
public repository since the release.

### 4.3 B5, adjacent hygiene, cheap

- **59 em dashes in public-facing repository documents**: `README.md` 29, `REPRODUCE.md` 24, `_STRUCTURE.md` 6.
  Enam's first standing rule is no em dashes, and these carry his name in public. Replace without changing
  meaning.
- `VIVA_DEFENSE_NOTES.md` carries about fifty more in its pre-August sections. Lower priority, not public.

### 4.4 Two code robustness notes, verified not to affect any result

- `ensemble_v2_combine.py`'s `subject_probs` takes `n = min(len(...))` across models and truncates **without
  asserting the models' `y_true` agree**. Verified on 3 September that this never fires: 40 subjects, 4 models,
  both probability directories, zero length or label mismatches. Turn the `min()` into an assertion so it cannot
  fire silently in future.
- `ensemble_v2_subjectwise.csv` is written without a subject column, so its 40 rows are positional. Add the
  column. Consistent as produced, fragile as published.

### 4.5 Decision point, for Enam

Nothing in B4 changes a reported number, so it can proceed without approval **except** the 400 ms ladder, where
recompute and delete are different choices with different costs. Report which you recommend and why, and wait.

---

## 4A. B8: the subject-dependent protocol. Enam chose the FULLEST option, 3 September.

Full detail in `04_Reviews_and_QA/METHODOLOGY_AUDIT_3SEP.md` §B8. Three defects: §3.5.1 describes a per-subject
CV while `train_classical_patched.py` runs a **pooled** `StratifiedKFold(shuffle=True)` over all 26,347 windows;
Table 4.1 therefore places pooled classical numbers beside a genuinely per-subject CNN number and §4.1 reads a
model ranking off that; and both protocols split **50%-overlapping** windows at random with no time grouping.

**This costs no LOSO.** Subject-dependent evaluation is within-subject by definition, so nothing here retrains a
40-fold cross-subject model. Classical is 40 subjects x 5 folds x 3 models on roughly 526 windows per fit, which
is seconds of CPU. The CNN is 40 subjects x 5 folds on the same small folds, far less total compute than one LOSO
run, so budget well under an hour on GPU.

### 4A.1 The split design, and the trap that rules out the obvious approach

**Do not split on `t_start` contiguously. It does not work, and here is the measurement.** SIAT-LLMD records each
movement as a **separate trial with its own clock**, so every movement restarts at t = 0 and sorting a subject's
windows by `t_start` interleaves the four. Splitting subject 1's t_start-ordered windows into five contiguous
blocks gives block 3 at 65% STDUP and **block 4 at 100% STDUP**, because the shorter recordings run out first.
Stratification is then impossible.

**The correct design is to block within each movement's own clock, then combine across movements.** For each
subject and each movement, sort that movement's windows by `t_start`, cut them into `n_splits` contiguous chunks,
and assign chunk *i* of every movement to fold *i*. Each fold then holds a contiguous time chunk of all four
movements and is class-balanced by construction.

**Add a guard band.** Drop windows within one window length of each chunk boundary, so no overlapping pair can
straddle a fold edge. At roughly 690 windows per subject this costs about 1% of the data; report the exact
fraction dropped.

Feasibility is confirmed: every one of the 160 subject-by-movement cells has at least 56 windows, so five
contiguous chunks per cell is comfortable.

### 4A.2 What to re-run

| Arm | Protocol | Cost |
|---|---|---|
| Classical SD, 250 ms, Freq-72, SVM + RF + LDA | per-subject, movement-blocked, guard band | CPU, minutes |
| Classical SD, 150 ms and 250 ms, base and extended sets, SVM + RF | same | CPU, minutes. **Needed because Figure 4.1 and §4.1's window comparison use these** |
| CNN SD, 250 ms | already per-subject; re-run **movement-blocked with guard band** | GPU, well under an hour |

Report the old and new figures side by side for every arm.

### 4A.3 Pre-registered expectation, and the cell that matters

**The blocked numbers will fall**, because removing the overlap leak removes an advantage. The direction is
predictable; the magnitude is not.

| Outcome | Condition | Reading |
|---|---|---|
| **Contained** | Gaps stay inside the 10 to 25 pp literature range cited in §6.1 | Update Table 4.1, Table 4.7 and the figures. §6.1's Objective 3 claim stands. |
| **Gaps narrow out of range** | Corrected gaps fall below about 10 pp | **A real result and it must be reported as one.** It would mean the published gap was partly a protocol artifact, and §6.1's "squarely within the 10 to 25 pp range" claim has to change. Do not soften this into a footnote. |
| **Ranking flips** | The CNN no longer leads under SD once all three are on the same protocol | Report it. §4.1's "the CNN leading narrowly" is then wrong and Chapter 4's opening changes. |
| **X** | Any subject has a class with too few windows for five blocked chunks after the guard band | Report which, reduce splits for that subject or exclude it with the exclusion stated. |

### 4A.4 The cascade Enam has accepted

Table 4.1, Table 4.7, Figure 4.1, Figure 5.1, Figures A.7, A.8 and A.14, plus text in §3.5.1, §4.1, §4.4, §5.x
and §6.1. **Enam has accepted the write-up cost.** Report the numbers; do not edit the thesis.

**Not affected, and say so in the verdict so nobody over-reads the change:** LOSO is untouched, since its outer
split is by subject and no overlapping window can cross it. Finding A, the alignment ladder, the 85.8% headline,
the 81.7% causal figure and the external replication all stand. The gap *reduction* argument is also untouched,
because the subject-dependent number is held constant across the baseline and optimized rows, so the change in
the gap equals the LOSO improvement exactly: 16.6 - 9.7 = 6.9 = 77.7 - 70.8.

---

## 5. Outputs

`06_Code/RUN_MANIFEST.csv` and `build_run_manifest.py`; `04_Reviews_and_QA/LITERATURE_BAND_TABLE.md` with the
enumeration, the inclusion rule and the exclusions; `results_locus/b3_per_subject_silhouette.csv` and a verdict
markdown; the regenerated ladder files with the originals backed up to `_ARCHIVE/`. Add a status header to the top
of this file recording each item's outcome, leaving the plan below it unedited.

## 6. What to report back

1. **B1:** the manifest, how many directories came back fully `verified`, how many carry any `unknown`, and
   **every mismatch against the canonical numbers table**. State explicitly whether the occlusion backbone is the
   only one.
2. **B2:** the enumeration table, the inclusion rule, the exclusions, whether the band survives, and the metric
   verdict.
3. **B3:** whether the per-subject silhouette analysis works, the two paired tests if so, and your recommendation.
4. **B4:** confirmation the regenerated ladder matches Table 4.16, and your recommendation on the 400 ms file.
5. Confirmation that no thesis file was edited and §4.17 was not touched.
