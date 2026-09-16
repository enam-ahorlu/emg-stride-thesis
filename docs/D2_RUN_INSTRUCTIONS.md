# Run instructions for P3.3 (Deep CORAL lambda sweep, D2)

The instructions handed to whoever runs this are everything below the line.

---

Run the Deep CORAL lambda sweep (D2) for the MSc thesis.

**Read first:** `06_Code/docs/EXPERIMENT_PLAN_DEEPCORAL.md`. Go to the section headed
**"D2 AMENDMENT, 15 September 2026"** at the end of that file and follow it. It supersedes the
original D2 section earlier in the same file, which was written before D1 ran and has two rules that
are now wrong. Do not follow the original. Read the D1 section above it for context only.

**Working directory:** `06_Code/`, in the project `.venv`. Use the absolute path to the inner python.

**Inputs, all verified present:**
```
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
SCRIPT=run_deep_coral_cnn_loso.py
```

**The one thing that must not be got wrong.** The per-subject comparator is
`results_cnn_aug_resnet_se_chandrop/cnn_arch_subjectwise.csv`, mean **0.839490**. It is NOT
`results_cnn_aug_resnet_se_chandrop_proba`, mean 0.841566. Those are different repeat runs of the same
arm. Only 0.839490 rounds to the 0.840 that Table 4.21 and Figure A.6 disclose, and the whole-thesis
correction family was re-homed in v6 specifically to fix two rows sourced from the wrong one. Verify
the mean to six decimals before computing any contrast and abort if it is not 0.839490.

**Order of operations. Do not reorder and do not skip ahead.**

1. Write `PREDICTION.md`, timestamped, into your output directory before any run starts. State the
   interior-maximum prediction as the amendment gives it: Deep CORAL degenerates to the 0.772
   global-norm base at lambda = 0 and scores 0.8257 at lambda = 1, so performance rises over that
   interval and the over-alignment account of §4.7.2 says it must fall again once alignment is pushed
   far enough. The open question is whether the peak sits above lambda = 1. Also record the three
   outcome rules from the amendment, so the result is read against them rather than after them.
2. **Reproduction gate.** Re-run lambda = 1.0 into `results_deep_coral_lam1p0_repro`. It must land
   within 1.5 pp of 0.825712. If it misses, stop and report; the harness has drifted and the sweep is
   not comparable to the published arm. Do not touch `results_deep_coral_chandrop`.
3. **Time one fold** with `--heldout 1` into a scratch directory, project the cost of one arm and of
   Stage A, and report the projection. If Stage A projects beyond about eight hours of GPU, stop and
   report rather than starting it. Note that `--epochs 40` is a ceiling: early stopping on the source
   validation loss with `--patience 7` means the first fold is an upper bound, not an average.
4. **Stage A:** lambda 0.1, 10, 100, each into its own `--out` directory
   (`results_deep_coral_lam0p1`, `results_deep_coral_lam10`, `results_deep_coral_lam100`), each with
   `--resume`, each an independently resumable job. Never point two lambdas at one directory:
   `--resume` reads the subjects already in that directory's CSV and would silently skip all forty
   folds and inherit the other lambda's numbers.
5. **Stage B, only if triggered.** If lambda = 10 lands within 0.5 pp of lambda = 1 or above it, the
   peak is not bracketed; run lambda 3 and 30. If lambda = 10 sits clearly below lambda = 1, the curve
   has turned over and Stage B does not run. The trigger is fixed in the plan; do not renegotiate it
   against how the result is going.
6. **Analysis** per Phase D2.4: the sweep curve with the 0.772 and 0.839490 references, paired
   Wilcoxon with paired Cohen's d and BCa intervals and subjects improving for every lambda against
   per-subject normalization and against lambda = 1 plus the best lambda against AdaBN
   (`results_adabn_chandrop/adabn_subjectwise.csv`, mean 0.8182), Holm within the sweep, every mean
   difference expressed as a multiple of the 0.5 pp nondeterminism figure from §3.16, and
   `report_figs/new_experiments/deep_coral_lambda.png` in the style of the classical CORAL sweep
   figure. Read `deep_coral_subjectwise.csv` for all analysis, never `deep_coral_summary.csv`, which
   rounds to four decimals.
7. **Family.** Recompute the whole-thesis Benjamini-Hochberg family as **v8** from v7's 218 tests and
   153 survivors. Admit every lambda-against-per-subject contrast and every lambda-against-lambda-1
   contrast as one member each, applying v7's genuine-tests-only rule (no bit-identical arms, no exact
   duplicates). Do not admit only the best lambda: picking the arm after seeing results and correcting
   as though one test had been planned is the selection the family exists to price. Record the
   composition and any exclusion in the v8 docstring in the form v7 used.

**Hard stops. These are not suggestions.**

- If the best lambda closes the gap to inside 0.5 pp of per-subject normalization (**Outcome B**), or
  overtakes it by more than 0.5 pp with a paired test that survives Holm (**Outcome C**): stop, report
  the lambda, the margin, the interval and the test, and run no further arms. Both outcomes change
  what the thesis claims and both are Enam's decision, not yours.
- Do not edit any thesis chapter, `MSc Thesis.docx`, the remediation tracker or the handoff. Leave
  CSVs, the PNG and a written report.
- Do not fabricate numbers. If an arm does not finish, say it did not finish.
- `--xkey` stays at its default `X_env`, seed stays at 42, no repeats per lambda. This is a
  perturbation study on lambda alone.

**Report back in the order given at the end of the amendment:** reproduction gate and its offset, the
single-fold timing and your projection, the comparator check to six decimals, the sweep table, whether
Stage B triggered and why, the paired contrasts with effect sizes and intervals and noise multiples,
whether the interior-maximum prediction held and where the peak sits, which outcome fired in one
sentence, and the v8 family counts with anything that changed side.

One framing note so you write the report at the right pitch. Whatever this returns, it narrows or
widens one sentence about one backbone. The normalization finding rests on consistency across three
model families, on the classical arm where CORAL was swept across four orders of magnitude, on the
alignment ladder, on the ENABL3S replication and on the active-only control. Outcome C would be a
real and interesting result and would not touch the spine of the thesis. Do not write as though it
could, in either direction.
