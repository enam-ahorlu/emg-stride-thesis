# Run order: the channel-dropout mechanism programme

**Written 2 September 2026. Total about 6.3 hours GPU, serial.**

The order below is not arbitrary. Follow it.

| # | Stage | Cost | Code state | Plan |
|---|---|---|---|---|
| 1 | **G3** occlusion baseline | 26 min | **unchanged** | `EXPERIMENT_PLAN_G3_OCCLUSION.md` |
| 2 | code changes + both inertness assertions | 0 | — | §2 of the W-4 and W-3 plans |
| 3 | **W-4** gain-jitter control | 50 min | changed | `EXPERIMENT_PLAN_GAIN_JITTER.md` |
| 4 | **W-3** skip-connection ablation | 1.3 h | changed | `EXPERIMENT_PLAN_RESIDUAL_ABLATION.md` |
| 5 | **G2** dropout-rate sweep | 3.7 h | changed | `EXPERIMENT_PLAN_CHANNEL_DROPOUT.md` §4.1, §4.3 |

**Why G3 first, before touching any code.** G3 compares a new run against
`results_cd_resnet_nose_chandrop`, which was produced on the current code. Run it now and that
comparison needs no inertness argument at all. Run it after the edits and it depends on an
assertion holding.

**Why both code changes land together at step 2.** `use_residual` (W-3) and the `gainjitter` mode
(W-4) both touch shared files. Making both, then asserting both, then running three stages against
one stable code state, means every result from step 3 onward is era-internal with every other.
Assert each change separately even so, so a failure names its own cause.

**Why this order among 3, 4, 5.** Steps 3 and 4 are short and each answers a distinct mechanism
question; step 5 is the long pole and is the one most likely to be confirmatory ("p = 0.2 stands").
Front-loading the short ones means an interruption costs the least information.

---

## The two inertness assertions, both mandatory

Neither is optional and neither can be replaced by reproducing a number: the pipeline's run-to-run
SD is 0.47 pp (R-1), so any tolerance loose enough to pass is too loose to detect a real change.

1. **W-4 §2.2.** For all five existing augmentation modes, in **both** `train_cnn_loso.py` and
   `run_cnn_arch_loso.py`: the output tensor bytes and the post-call torch RNG state must be
   identical before and after adding the `gainjitter` branch.
2. **W-3 §2.1.** For `arch="resnet"` and `arch="resnet_se"`: the post-construction torch RNG state
   and the concatenated initial parameter bytes must be identical before and after adding
   `use_residual`.

Capture the "before" values on the current code **first**, into a file, then edit. If either fails,
stop: every prior run becomes incomparable and nothing downstream is interpretable.

---

## Standing rules for all five stages

- **Do not edit `MSc Thesis.docx` or any chapter file.** Wording is Enam's once numbers are in.
- **Report the pre-registered outcome letter first**, then the gates, then the numbers. Do not
  narrate a direction that failed to clear its threshold as if it were a finding.
- **A null is a result.** G3 Outcome N, W-3 Outcome D, W-4 Outcome P and "p = 0.2 stands" are all
  outcomes the plans were written to accept. Report them plainly; do not go looking for a reading
  that rescues a hypothesis.
- **If G2 §4.3 produces a challenger to p = 0.2, STOP AND ESCALATE.** Adopt nothing. Eight
  downstream result sets consume p = 0.2; see parent plan §9.
- **Check every `--resume` for duplicated subject rows** before trusting a summary. That is the
  failure mode a resume produces, and every stats script asserts against it.
- Record wall-clock per run. Measured references: 25.6 min for a no-augmentation `resnet` 40-fold
  run, 52.5 min for the channel-dropout one (`_run_logs/w2_g1.log`).

---

## What each stage buys the thesis

| Stage | Question | Analogue in the per-subject normalization treatment |
|---|---|---|
| G3 | Does channel dropout flatten electrode reliance, and does the profile transfer across people? | §4.13.1's direct measurement of the mechanism quantity |
| W-4 | Is it removal, or per-channel variation in general? | §4.13's head-to-head against CORAL, Deep CORAL and AdaBN |
| G2 | Does more enforced invariance eventually hurt? | §4.13.2's over-alignment result |
| W-3 | Which architectural property does the gain need? | no analogue; this one is specific to the augmentation |

Together these are what brings channel dropout to something near the depth the thesis gives
per-subject normalization, which is the point of the programme.
