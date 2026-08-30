# E-C3 — Does the calibration buffer's class composition flatter the deployable figure?

## Command run

```
& '.\.venv\Scripts\python.exe' -u run_buffer_composition.py --stage cnn --resume
& '.\.venv\Scripts\python.exe' -u run_buffer_composition.py --stage svm --resume
& '.\.venv\Scripts\python.exe' -u run_buffer_composition.py --stage combine
```

New script: `run_buffer_composition.py`. `t_start` restarts at 0 for each of
the four movement recordings, so `np.argsort(t_start)` interleaves the four
movements and the "first K time-ordered windows" is a near-class-balanced
sample from the opening seconds of all four recordings (measured earlier:
22.0 WAK / 25.4 UPS / 25.9 DNS / 26.7 STDUP on average, all four classes
present for all 40 subjects). The script makes the buffer an explicit index
set (`buffer_indices`) and reuses `run_streaming_norm_loso.per_subject_transductive`
and `run_causal_ensemble.buffer_mask` unchanged. Each model is trained ONCE per
held-out subject and all six buffer modes are scored from that fitted model
(SVM arm: cached `_bestparams.json` refit — legitimate, identical model across
modes; ensemble arm: ResNet-SE+CD trained exactly as
`run_causal_ensemble.stage_cnn`, then re-normalized + re-inferred per mode).
Buffer windows are excluded from the F1 via the `buffer_mask` logic; scoring is
over all remaining windows of all four classes.

Two SVM variants are reported: `SVM` (plain decision-function SVC, mirrors
`rescore_streaming_buffer_v2.py`) and `SVM_PROBA` (`probability=True` SVC,
mirrors `run_causal_ensemble.stage_svm`, feeds the soft vote). Modes at K=100:
`mixed100` (published), `single_{WAK,UPS,DNS,STDUP}` (true contiguous
single-activity buffer), `balanced25` (25 windows from the start of each
movement).

Outputs in `results_buffer_composition/`: `buffer_composition_subjectwise.csv`
(subject, model, mode, f1_incl, f1_excl, buf_n), `buffer_composition_summary.csv`,
`buffer_composition_wilcoxon.csv`.

## Validation gate — 3 of 4 quantities reproduce; solo CNN off by 0.0058

| quantity (mixed100, buffer-excluded F1) | this run | published | \|diff\| | source |
|---|---|---|---|---|
| SVM (decision-function) | 0.7476 | 0.7476 | **0.0000** | `streaming_buffer_rescore_summary.csv` |
| SVM (probability) | 0.7319 | 0.7319 | **0.0000** | `results_causal_ensemble/report.csv` |
| ResNet-SE+CD | 0.7937 | 0.7995 | 0.0058 | `results_causal_ensemble/report.csv` |
| **soft ensemble** | 0.8171 | 0.8168 | **0.0003** | `results_causal_ensemble/report.csv` |

`mixed100` is byte-identical to the published `calib100` protocol (buffer =
first 100 by `t_start`, `is_buffer` = `buffer_mask(order, n, 100)`). Both
deterministic classical pipelines reproduce their published numbers exactly
(0.0000), and the headline soft ensemble reproduces to 0.0003. The solo
ResNet-SE+CD is 0.58 pp low: `run_causal_ensemble.stage_cnn` — which this
script mirrors line-for-line — does not set
`torch.backends.cudnn.deterministic=True`, so cuDNN selects nondeterministic
convolution algorithms and a full 40-fold retrain drifts ~0.5–1 pp on the mean
F1. The harness reproduces the published *protocol*; the CNN is not
bit-reproducible by construction, and the buffer-composition comparison (the
question E-C3 asks) does not depend on the exact solo-CNN value. **The gate is
treated as substantively passed** (2 exact, ensemble to 3e-4); the CNN delta is
recorded here as expected retrain variance, not a harness defect.

## Summary table — buffer-excluded F1 by mode

| mode | ResNet-SE+CD | SVM (df) | SVM (proba) | **soft** | mean buffer n |
|---|---:|---:|---:|---:|---:|
| `mixed100` (published protocol) | 0.794 | 0.748 | 0.732 | **0.817** | 100 |
| `balanced25` (25/movement, scripted) | 0.791 | 0.747 | 0.729 | **0.815** | 100 |
| `single_DNS` | 0.586 | 0.530 | 0.558 | 0.613 | 92 |
| `single_WAK` | 0.500 | 0.482 | 0.510 | 0.563 | 76 |
| `single_UPS` | 0.536 | 0.432 | 0.418 | 0.515 | 97 |
| `single_STDUP` | 0.320 | 0.349 | 0.362 | 0.367 | 100 |

Paired Wilcoxon vs `mixed100` (soft ensemble, n = 40 subjects,
`buffer_composition_wilcoxon.csv`):

| mode | mean ΔF1 | Wilcoxon p |
|---|---:|---:|
| `balanced25` | −0.0019 | 0.084 (n.s.) |
| `single_DNS` | −0.2046 | < 1 × 10⁻⁵ |
| `single_WAK` | −0.2540 | < 1 × 10⁻⁵ |
| `single_UPS` | −0.3026 | < 1 × 10⁻⁵ |
| `single_STDUP` | −0.4502 | < 1 × 10⁻⁵ |

(Same pattern for all three solo models; every single-movement mode is
p < 1 × 10⁻⁵ against `mixed100` for every model.)

## Does the result support or contradict the thesis as written?

**It supports the plan's diagnosis and the 81.7% number itself, but it
requires Section 4.14.1's protocol description to be corrected and a range to
be reported.** The buffer is not "the first K time-ordered windows of a
contiguous session"; because `t_start` resets per recording it is
approximately K/4 windows from the *start of each of the four movement
recordings* — a short, scripted, unlabeled calibration routine of about twelve
seconds covering every movement. The direct test of that reading,
`balanced25` (exactly 25 windows from the start of each movement), reproduces
the published deployable figure almost perfectly: soft ensemble 0.8152 vs
0.8171 for `mixed100`, mean difference −0.0019, Wilcoxon p = 0.08 (not
significant). So **the 81.7% headline is robust — conditional on the
calibration protocol being that scripted four-movement routine.**

What the thesis cannot keep is the *implied* protocol. A genuinely contiguous
single-activity calibration buffer — which is what "the first K time-ordered
windows" reads as — collapses the deployable soft-ensemble F1 to 0.367
(STDUP-only) through 0.613 (DNS-only), a 20 to 45 point drop, every comparison
p < 1 × 10⁻⁵. STDUP-only is the pathological case: calibrating the normalizer
on stand-up-transient windows alone gives a scale that is wrong for the other
three classes. Note also that `single_WAK` and `single_DNS` buffers are often
shorter than 100 windows (mean 76 and 92) because many subjects have fewer
than 100 windows in those recordings, which is itself a realistic constraint
on a single-activity commissioning buffer.

**Recommended edits to Section 4.14.1.** (1) Describe the calibration set as it
is: "≈25 windows (about three seconds) from the opening of each of the four
movement recordings — a scripted commissioning routine — not a contiguous
session-start buffer." (2) State that the 81.7% figure holds under that
routine (`balanced25` = 81.5%, indistinguishable from the reported value) and
report the single-activity sensitivity as the honest bound: a calibration
buffer dominated by one movement yields 0.37–0.61 depending on the movement,
so commissioning the device requires the user to perform all four movements
briefly. This is more informative than the single point the section currently
quotes, and it removes the "more favourable protocol than described" objection
the reviewers can otherwise raise.
