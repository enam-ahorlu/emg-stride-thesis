# B8 extension: the ENABL3S subject-dependent tier, movement-blocked

Run 4 to 5 September 2026, on Enam's instruction, as item 1.2 of the writing phase. It closes the
defect recorded at section 2.2 of `WRITING_PHASE_DIAGNOSIS_ADDENDUM.md`: section 4.12 quoted ENABL3S
subject-dependent baselines from `results_ext_sd/`, a pooled shuffled 5-fold over 50%-overlapping
windows, which is the protocol section 4.1.3 corrects. The SIAT side had been corrected and the
external side had not, so the "roughly double the SIAT gap" claim compared a corrected figure against
an uncorrected one.

## The trap, caught before the run

`b8_movement_blocked_sd.py` computed its guard band as `window_ms / 1000`, in seconds. **SIAT's meta
writes `t_start` in seconds; the ENABL3S windower writes it in sample indices.** A guard band of 0.25
in a sample-indexed axis is a quarter of one sample: the script would have run, reported a guard
fraction near zero, and produced a "movement-blocked" number that still leaked, silently.

Two guarded additions, both defaulted to the previous behaviour:

- `--time-units {seconds,samples}`, default `seconds`. On `samples` the guard is
  `window_ms / 1000 * median(fs)`.
- `--min-guard-frac`, default 0.0. Aborts without writing if the guard drops less than the given
  fraction, so a units error on a new dataset fails loudly instead of quietly.

Measured effect of the fix on ENABL3S: guard fraction 0.35% with the wrong units against 1.00% with
the right ones. The 1.00% is correct for this data shape rather than a residual symptom: an ENABL3S
subject-by-movement group spans about 1.1 million samples across roughly fifty concatenated circuits,
against a single 14-second trial on SIAT, so proportionally far fewer windows sit at one of the four
chunk seams.

## Inertness control

The published SIAT `freq72_w250` config was re-run through the same environment before the ENABL3S
arm was trusted, because the run happened on scikit-learn 1.8.0 rather than the project venv.

| | SVM old | SVM new | RF old | RF new | LDA old | LDA new | guard |
|---|---|---|---|---|---|---|---|
| published `b8_all_configs.csv` | 93.67 | 88.83 | 91.98 | 87.99 | 89.85 | 84.57 | 8.2% |
| container re-run | 93.67 | 88.83 | 91.98 | 87.99 | 89.85 | 84.57 | 8.18% |

Every figure reproduces. **PASS.** Log: `b8_INERTNESS_siat_freq72_w250.log`.

## Result

ENABL3S, Freq-56, 250 ms, 10 subjects, SVM and RF, guard fraction 1.00%, no outcome-X flags (every
fold keeps test windows and all four classes in training).

| model | SD old (pooled random) | SD new (movement-blocked) | delta | Wilcoxon p | d |
|---|---|---|---|---|---|
| SVM | 86.04 | **82.14** | -3.90 pp | 0.001953 | -2.43 |
| RF | 85.61 | **79.70** | -5.91 pp | 0.001953 | -2.99 |

**Two findings.**

1. **The corrected gaps.** Against the published ENABL3S LOSO figures (per-subject SVM 0.657, RF
   0.636) the movement-blocked baselines put the subject-dependent-to-LOSO gaps at **16.4 pp (SVM)
   and 16.1 pp (RF)**, against the 11.1 and 10.7 pp that section 4.1.3 measures the same way on SIAT.
   The ENABL3S gap is wider, but not double, which is what section 4.12 previously claimed on the
   uncorrected numbers.
2. **The leak replicates externally.** The pooled-versus-blocked cost is 3.9 pp for the SVM and 5.9 pp
   for the Random Forest, both inside the 3.7 to 6.1 pp range section 4.1.3 measures across six SIAT
   configurations. Section 4.1.3's methodological finding is therefore not specific to SIAT-LLMD, and
   that strengthens it.

## Reproduction caveat, same as B8's own

The pooled-random figures here (86.04, 85.61) sit above `results_ext_sd`'s 84.30 and 81.16 because
this harness uses fixed LOSO-consistent hyperparameters rather than the per-fold grid the original
subject-dependent driver ran. The controlled claims are the pooled-versus-blocked delta and the
blocked levels, exactly as in the SIAT B8 run, where the re-run's pooled SVM was 93.7 against Table
4.1's 87.4.

## New paired tests for the Benjamini-Hochberg family

Two, both raw p = 0.001953: the pooled-versus-blocked contrast for the SVM and for the Random Forest
on ENABL3S. Section 4.17 not edited; these join the recompute at the end of the writing phase.

## Written into the thesis

Section 4.12, one paragraph, in `MSc Thesis.docx` and `Results_Chapter.docx`. `results_ext_sd/` is no
longer cited anywhere in the thesis.
