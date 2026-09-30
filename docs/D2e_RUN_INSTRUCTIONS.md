# D2e: where does Deep CORAL's gain come from? Run instructions for Claude Code

Written 20 September 2026, before any D2e run. Approved by Enam ("confirm first").

## Why

D2c Stage B ran lambda = 0 in `run_deep_coral_align_loso.py` and got mean macro-F1 0.8300 (40/40 folds), 5.8 pp above the
0.772 global-normalization ResNet-SE+CD reference (three runs 0.7723 / 0.7667 / 0.7760). With no CORAL loss, lambda 0 sits level
with lambda 0.1 (0.8243) and lambda 100 (0.8415). The training loop forwards a target batch every step in `model.train()`, so
BatchNorm running statistics absorb the held-out subject's statistics. `run_deep_coral_cnn_loso.py`, which produced every Deep
CORAL number in the thesis, has the same loop. Hypothesis H: Deep CORAL's lift over global normalization comes from target
statistics entering batch normalization, not from the alignment loss. D2e tests H directly. It changes no existing result file.

## Code change (one flag, default unchanged)

Add `--target-pass {train,eval,none}` to `run_deep_coral_align_loso.py`, default `train` (current behaviour, bit-identical).

- `none`: no target DataLoader is built and no target batch is forwarded; the CORAL term is not computed. Only legal with
  `--coral-lambda 0` (raise otherwise). Note in the log that the RNG stream differs from `train` because the target loader's
  shuffle no longer draws.
- `eval`: the target batch is forwarded with `model.eval()` (BatchNorm uses and does not update running statistics; dropout off),
  then `model.train()` is restored before the backward pass. Gradients still flow through the CORAL term. Do not use no_grad.

Check before the runs, on CPU synthetic data: `train` reproduces the current script bit-for-bit; under `eval` and `none`, the
BatchNorm `running_mean` after one epoch equals that of a run forwarding source batches only.

## Arms

Same configuration as D2c: ResNet-SE, `--augmentation chandrop` (p = 0.2), global per-channel z-score fit on the training fold,
`X_env`, seed 42, batch 256, 40-epoch ceiling, patience 7, no repeats, alignment measurement on.

| Arm | Flags | Directory | Role |
|---|---|---|---|
| E1, required | `--coral-lambda 0 --target-pass none` | `results_deep_coral_align_lam0_notgt` | tests H |
| E2, recommended | `--coral-lambda 30 --target-pass eval` | `results_deep_coral_align_lam30_tgteval` | the CORAL loss without target BatchNorm statistics |

Comparator for both: `results_deep_coral_align_lam0` (lambda 0, target pass in train mode, 0.8300), already on disk.
Weight 30 is the best weight of D2 and the Deep CORAL row of Table 4.6. Expected cost about 2.2 h an arm. Run E1 first.

## Prediction and decision rules, fixed now

Paired over 40 subjects: Wilcoxon, dz, BCa 95% (10,000 resamples), subjects lower, Holm across the contrasts below.

1. E1 minus lambda-0-train, macro-F1. **H confirmed** if E1 is at least 3.0 pp lower, Holm p < 0.05, and E1's mean is within
   1.5 pp of 0.772. **H rejected** if E1 is within 1.5 pp of 0.8300: then the harness differs from the 0.772 run for another
   reason; stop, find it, report. Anything else: **partial**; report the numbers, no framing.
2. E2 minus E1, macro-F1 (only if E2 runs). The CORAL loss **contributes on its own** if E2 is higher by more than 0.5 pp
   (the nondeterminism band), Holm p < 0.05. Otherwise it does not measurably contribute.
3. Descriptive: domain probe and target class probe for E1 and E2 against lambda-0-train. Report, no rule.

## Constraints

Run from `06_Code/` in the project `.venv`. Do not touch any existing `results_*` directory or any thesis file. Do not admit
anything to the BH family (v9 waits for Enam's framing). Report unfinished arms as unfinished; do not fabricate numbers.

## Report back, in this order

The CPU check; E1 mean F1 and its offset from 0.772 and 0.8300; contrast 1 with the verdict in one sentence; E2 and contrast 2
if run; the probe descriptives; anything unexpected in the training logs.
