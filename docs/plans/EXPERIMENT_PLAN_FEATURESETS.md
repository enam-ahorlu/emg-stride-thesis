# EXPERIMENT_PLAN_FEATURESETS.md: does the feature-set ordering survive the cross-subject protocol?

**Why.** §4.1.1 settled the feature set for the whole thesis on a subject-dependent comparison, where
Freq-72 led for the SVM at 0.874 against Combined-81's 0.873, Base-36's 0.860 and Extended-54's
0.842. §4.1.3 then showed that comparison ran on the pooled window-level split, which inflates
subject-dependent macro-F1 by 3.7 to 6.1 pp. §4.1.1 states the consequence itself: the choice "was
settled under the pooled window-level split", so "whether the ordering between sets survives a
movement-blocked comparison is untested. And it was not made under the cross-subject protocol, which
is the protocol every result in this thesis is reported on."

Every headline number therefore rests on a feature set chosen by a procedure the thesis discredits.
It is disclosed rather than hidden, and the mitigation is real, since the two leading sets were
statistically indistinguishable and §4.9 shows LOSO performance is flat across RFE-36, RFE-27 and
Full-72. But it is a live gap and closing it is cheap.

**Half the answer is already on disk, and it is a good answer.** Two of the four sets have
cross-subject numbers under per-subject normalization:

| set | source dir | SVM LOSO macro-F1 | RF LOSO macro-F1 |
|---|---|---|---|
| Base-36 | `results_loso_norm_persubj` | **0.7769** | **0.7711** |
| Freq-72 | `results_loso_freq_persubj` | **0.7767** | 0.7732 |

Under the cross-subject protocol the 1.4 pp subject-dependent lead of Freq-72 over Base-36 **is gone**.
The SVM figures differ by 0.02 pp, in Base-36's favour; the RF figures differ by 0.21 pp, in Freq-72's
favour. That is the shape of a null, not of an ordering. It is also consistent with the thesis's own
argument in §5.4 that preprocessing dominates feature engineering once the base architecture is sound,
so if it holds across all four sets it supports the thesis rather than threatening it.

**What is missing** is Extended-54 and Combined-81 under the same protocol. Those two runs complete the
four-way and are the whole cost of this experiment.

Run from `06_Code/` in the project `.venv`. CPU. `--n-jobs 1` inside the CV loop, `--rf-n-jobs 4`, per
`EXPERIMENTS_README.md`. Do NOT fabricate numbers. Do NOT edit any thesis chapter.

**Outputs:** `results_featureset_loso/featureset_loso_summary.csv`,
`results_featureset_loso/featureset_loso_subjectwise.csv`,
`results_featureset_loso/featureset_loso_wilcoxon.csv`,
`report_figs/new_experiments/featureset_loso.png`

---

## Phase 0. Confirm the two existing runs are what this plan assumes [RUN, CPU]

1. Read the four summary CSVs in `results_loso_norm_persubj` and `results_loso_freq_persubj` and
   confirm the table above.
2. **GATE:** `results_loso_freq_persubj` must reproduce §4.2.1's published Freq-72 per-subject
   figures, 0.777 for the SVM and 0.773 for the RF. If it does not, stop: this directory is the anchor
   the whole comparison hangs on.
3. Confirm from each directory's `run_config.json` that both runs used `--norm-mode per_subject`,
   `--cv-scheme loso`, `--inner-splits 5` and `--seed 42`, and record any setting that differs between
   them. If the two existing runs were not run under matched settings, say so before adding two more,
   because then the four-way is not a clean comparison and the two new runs must match whichever
   configuration is canonical.
4. Confirm the feature dimensionality of each npz: Base-36 at 36, Extended-54 at 54, Freq-72 at 72,
   Combined-81 at 81. If any set is not the width §3.3 claims, stop and report.

## Phase 1. The two missing sets under the same protocol [RUN, CPU]

Full nested search for both. Do **not** pass `--reuse-params-dir`: the tuned hyperparameters in
`results_loso_freq_persubj` were selected for a 72-dimensional space and carrying them to a
54- or 81-dimensional one would hand those two sets a handicap the other two did not have, which is
exactly the kind of asymmetry this experiment exists to remove.

```
# Extended-54
python train_classical_loso.py \
    --features features_out/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz \
    --meta     features_out/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
    --out results_featureset_loso/ext54 --models SVM,RF --norm-mode per_subject \
    --cv-scheme loso --inner-splits 5 --n-jobs 1 --rf-n-jobs 4 --seed 42 \
    --save-preds --flush-preds --resume

# Combined-81
python train_classical_loso.py \
    --features features_out/combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_full.npz \
    --meta     features_out/combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
    --out results_featureset_loso/combined81 --models SVM,RF --norm-mode per_subject \
    --cv-scheme loso --inner-splits 5 --n-jobs 1 --rf-n-jobs 4 --seed 42 \
    --save-preds --flush-preds --resume
```

**SANITY:** both must produce 40 subjects with no duplicates and no missing folds before anything is
compared. Check the subjectwise CSV row counts, not just the summary.

These are the long runs in this plan. Use `run_with_memory_guard.py` if the machine is loaded, and
rely on `--resume` rather than restarting from zero after an interruption.

## Phase 2. The four-way comparison [RUN, CPU]

Assemble one table: four feature sets, two models, LOSO macro-F1 mean and inter-subject SD, with the
subject-dependent figures from §4.1.1 in adjacent columns so the two protocols can be read against
each other.

Per-subject paired contrasts across the 40 subjects, within model, Wilcoxon signed-rank with Cohen's d
and BCa 95% intervals, Holm-corrected within this family of six comparisons per model:

- Freq-72 against each of Base-36, Extended-54 and Combined-81,
- Combined-81 against Base-36 and Extended-54,
- Extended-54 against Base-36.

**Report the ordering under both protocols side by side, and state plainly whether the subject-dependent
ordering survives.** That sentence is the deliverable. Everything else is supporting detail.

These paired contrasts add members to the whole-thesis Benjamini-Hochberg family. Record how many. Do
not rebuild the family here; that happens once, in the write-up wave.

## Phase 3. Figure [RUN, CPU]

`report_figs/new_experiments/featureset_loso.png`. Grouped bars, four sets by two models, LOSO
macro-F1, error bars at one standard error across the 40 subjects, with the subject-dependent values
overlaid as open markers so the divergence between protocols is visible in one glance. Match the
existing `report_figs/new_experiments` style.

---

## What each outcome means

**Outcome A, the four sets are indistinguishable under LOSO.** This is what the two existing runs
point at. Then §4.1.1's caveat closes in the best possible way: the choice was made on a discredited
protocol, and it turns out not to have mattered, because no feature set has a cross-subject advantage
over any other. This strengthens §5.4's claim that preprocessing dominates feature engineering, and it
removes the circularity objection entirely rather than merely bounding it.

**Outcome B, Combined-81 leads clearly under LOSO.** Then the thesis reports its results on a
second-best feature set. This does not invalidate anything, since every comparison in the thesis is
internally consistent on Freq-72, but it has to be stated in §4.1.1 and §5.5, and the size of the lead
decides whether anything more is warranted. Do **not** re-run the thesis on Combined-81 on your own
initiative: that is a scope decision, and the escalation rule is to report and stop.

**Outcome C, Freq-72 leads clearly under LOSO.** Then the original choice was right for the wrong
reason, which is worth saying in exactly those terms.

In all three cases the reported results stay on Freq-72. What changes is only what §4.1.1 is allowed
to say about how that set was chosen.

## After the runs: hand back for the write-up cascade

Leave the CSVs and the PNG. Do not edit chapters, and do not touch `results_loso_norm_persubj` or
`results_loso_freq_persubj`. Report:

1. Whether the Phase 0 gate reproduced §4.2.1, and whether the two existing runs used matched settings.
2. The four-set, two-model LOSO table with inter-subject SDs.
3. The paired statistics, Holm-corrected.
4. One sentence: does the subject-dependent ordering survive the cross-subject protocol.
5. Which of Outcome A, B or C occurred.
6. How many new BH family members this adds.

---

## AMENDMENT, 11 September: the first attempt died of out-of-memory. Run the two jobs SEQUENTIALLY.

**What happened.** Both jobs were launched concurrently and both crashed after 3 to 4 subjects with
`numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.08 MiB for an array with shape
(5228, 54)`. A one-megabyte allocation failing means the machine had no RAM left at all, not that any
single job was too large. The orchestrator then sat in its watch loop logging `ext54=0/2
combined81=0/2` every five minutes against two dead jobs.

**This is a defect in the plan above, not in the runner.** `EXPERIMENTS_README.md` records the lesson
already: nested grid search over 40 LOSO folds is memory-hungry, `GridSearchCV`'s own `n_jobs` must
stay at 1, and `run_with_memory_guard.py` and `run_multi_guard.py` exist for exactly this. The plan
said to use the guard only "if the machine is loaded", which left concurrency open. It should have
forbidden it. Two full nested searches at once on a 15.7 GB machine is not a supported configuration.

**Corrected instructions for Phase 1.**

1. **One job at a time. Do not launch ext54 and combined81 together, and do not use an orchestrator
   that watches both.** Combined-81 is the larger of the two, so run Extended-54 first and finish it.
2. Wrap each job in the memory guard:

```
python run_with_memory_guard.py --max-mem-percent 94 --min-free-gb 1.0 -- \
    python train_classical_loso.py \
    --features features_out/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz \
    --meta     features_out/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
    --out results_featureset_loso/ext54 --models SVM,RF --norm-mode per_subject \
    --cv-scheme loso --inner-splits 5 --n-jobs 1 --rf-n-jobs 3 --seed 42 \
    --save-preds --flush-preds --resume
```

   then the same for `combined81` with its own features, meta and `--out`.
3. `--rf-n-jobs` drops from 4 to **3**, per the README's guidance that unbounded RF parallelism risks
   memory spikes on a tight machine.
4. Consider `--models SVM` and `--models RF` as separate sequential invocations rather than `SVM,RF`
   in one, if memory is still tight. The checkpointing is per fold, so splitting by model costs
   nothing but wall clock.
5. **The existing checkpoints are good.** `ext54` holds 3 SVM and 3 RF folds, `combined81` holds 3 SVM
   and 2 RF folds. `--resume` will pick up from there. Do not delete the directories.
6. Before starting, confirm nothing is still holding memory: the earlier attempt left several idle
   python processes and a looping orchestrator. Kill those first.

Everything else in the plan above stands unchanged, including the ban on `--reuse-params-dir` and the
Phase 0 gate.
