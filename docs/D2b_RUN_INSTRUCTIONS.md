# Run instructions for P3.4 (D2b, the comparator control)

The instructions handed to whoever runs this are everything below the line.

---

Run D2b, the comparator control for the Deep CORAL lambda sweep.

**Read first:** `06_Code/docs/EXPERIMENT_PLAN_DEEPCORAL.md`, the section headed **"D2b AMENDMENT,
16 September 2026"** at the very end of the file. Follow that. The D2 amendment above it is done;
read it for context only.

**Why this exists.** Your D2 run was careful and the reproduction gate was the right instrument, but
the conclusion then carried the gate's own finding into the answer. Decomposing the closure from
D1's 1.378 pp to your reported 0.479 pp: the lambda = 1 re-run beat the published lambda = 1 arm by
0.721 pp, and the best lambda beat that re-run by 0.178 pp at p = 0.90. So 80% of the gap closure is
the same configuration re-run and 20% is tuning. The offset is broad rather than a few unstable
folds: 26 of 40 subjects rose, the median move is +0.632 pp, and dropping the three largest movers
still leaves +0.635 pp. Outcome B therefore rests on comparing arms trained this week against a
per-subject arm trained weeks ago, with an offset term larger than the residual gap. One control arm
settles it.

**Your within-harness finding is untouched and stands as established:** lambda is inert across three
orders of magnitude, every lambda-against-lambda-1 contrast at most 0.18 pp, no detectable peak or
fall. That was measured inside one harness and needs no control. It is also, on its own, the answer
to the objection D2 was run for.

**Working directory:** `06_Code/`, in the project `.venv`, absolute path to the inner python.

**The run.** One arm, 40 folds, about 2.2 hours at your realised 200 s per fold.

```
python run_cnn_arch_loso.py --npz $NPZ --meta $META \
    --arch resnet_se --norm-mode per_subject --augmentation chandrop --epochs 40 \
    --out results_persubj_chandrop_repro --resume
```

Note the script: `run_cnn_arch_loso.py`, not the Deep CORAL one. Everything else stays at that
script's defaults, which are the published settings: `--xkey X_env`, `--batch 512`, `--lr 1e-3`,
`--patience 7`, `--val-frac 0.15`, `--seed 42`, `--aug-chandrop-p 0.2`. `--batch` defaults to 512
here and 256 in the Deep CORAL script; that difference is between the two arms as published and must
be preserved, not harmonized. Print the command before you run it. If anything in
`EXPERIMENT_PLAN_CHANDROP.md` contradicts this reconstruction, stop and report rather than choosing.

Do not touch `results_cnn_aug_resnet_se_chandrop` or any other published results directory.

**Reading the result, fixed in the amendment.** Let R be the re-run mean against the published
0.839490.

- **A1, R rises by about +0.5 pp or more.** The environment moved and both arms moved with it.
  Recompute the gap as R minus the best lambda, both from this week. If it exceeds 0.5 pp, D2
  resolves to **Outcome A** and the thesis hardens rather than softens.
- **B1, R holds within about 0.2 pp of 0.839490.** The offset was local to the Deep CORAL runs, the
  like-for-like gap really is about 0.5 pp, and D2's Outcome B stands on a fair comparison.
- **C1, R falls, or rises by more than about 1.5 pp.** Neither is expected. Stop and report.

**The family.** Do not recompute it for D2b and do not admit the control as a new test; the same arm
against itself in a different week is not a test, by the rule v7 applied to the filter chain. **If
the outcome is A1**, the lambda-against-per-subject contrasts already in v8 were computed against
the wrong comparator and must be recomputed in place as **v9 against R**, with the substitution
recorded in the v9 docstring and flagged for Appendix A.6. Members are substituted, not added.

**Constraints unchanged:** do not edit any thesis chapter, `MSc Thesis.docx`, the remediation tracker
or the handoff. Do not fabricate numbers. Seed 42, no repeats.

**Report back:**

1. The command you ran, printed before the run.
2. R to six decimals, n, SD.
3. R minus 0.839490 in pp, paired across subjects: mean, median, subjects improved, Wilcoxon p, and
   the mean after dropping the three largest movers by absolute delta. This is the same diagnostic
   that exposed the Deep CORAL offset, and it is what separates environment drift from a noisy draw.
4. The like-for-like gap, R minus the best lambda arm: paired mean, BCa 95%, Cohen's dz, subjects
   improved, and as a multiple of 0.5 pp.
5. Which outcome fired, A1, B1 or C1, in one sentence.
6. If A1: the v9 recomputation and whether anything changed side.
