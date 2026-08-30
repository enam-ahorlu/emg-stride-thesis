# E-C2 — Is rung 4 a fair whitening, or an artifact of its regularizer?

## Command run

```
& '.\.venv\Scripts\python.exe' -u check_rung4_robustness.py --resume
```

New script: `check_rung4_robustness.py`. Reuses `rung0_global_z`,
`rung3_mean_scale`, `rung4_full_whiten_recolor`, `sym_sqrt`, `sym_invsqrt`,
`stratified_subsample`, `rbf_mmd2`, `median_heuristic_gamma` from
`analyze_between_subject_variance.py` and `per_subject_zscore` from
`train_classical_loso.py`. No transform or metric code was reimplemented; the
per-(subject,class) window subsample (cap 100, `np.random.default_rng(42)` in
the same loop order) and the shared silhouette subsample (`stratified_subsample`)
are the identical point sets `part_b_ladder` uses.

CSV: `results_variance_decomposition/rung4_robustness.csv` (new file alongside
the published `alignment_ladder.csv`, which was not touched); copy in
`results_EC2/rung4_robustness.csv`. Runtime 59 s, CPU.

## Validation gate — PASS

`raw_lam1` is the published rung-4 transform (`rung4_full_whiten_recolor`,
lam=1.0) rerun through this harness. It reproduces the published rung-4 row of
`alignment_ladder.csv` exactly:

| quantity | published rung 4 | raw_lam1 (this run) | abs diff |
|---|---|---|---|
| MMD removed % | 72.95770 | 72.95770 | 0.00e+00 |
| subject probe bal. acc. | 0.0118718 | 0.0118718 | 0.00e+00 |
| silhouette by class | −0.0060605 | −0.0060605 | 0.00e+00 |

Byte-identical (same code, same seeds), so the harness reproduces the old
number before any new condition is read.

## Summary table

Feature-variance spread on the raw matrix, as the plan states:
min 3.36e-05, max 5.82e+03 — 8.2 orders of magnitude; 18 of 72 features have
variance below 0.01, for which the fixed λ=1 ridge is >100× their own variance.

| variant | MMD removed % | W1 removed % | subject probe bal. acc. | silhouette by class |
|---|---|---|---|---|
| `raw_lam1` (published) | 72.96 | 75.25 | 0.01187 | **−0.00606** |
| `prez_lam1` (faithful CORAL analogue) | 68.74 | 64.81 | 0.01662 | **+0.01658** |
| `prez_scalefree_a1` (λ = tr(C)/p) | 69.63 | 63.89 | 0.01476 | **+0.01542** |
| `prez_scalefree_a01` (λ = 0.1·tr(C)/p) | 72.83 | 70.39 | 0.00932 | **+0.00954** |

Reference rungs from the published ladder: rung 3 (`mean_scale`) silhouette
0.02296, probe 0.02419; rung 0 (`global_z`) silhouette 0.00798, probe 0.77713.

Ordering test — for every one of the four variants, at rung 4:
- silhouette by class is **below** rung 3's peak of 0.02296 (−0.0061, +0.0166, +0.0154, +0.0095); and
- the subject-identity probe is **below** rung 3's 0.02419 (0.0119, 0.0166, 0.0148, 0.0093).

The load-bearing ordering — full second-order whitening removes still more
subject structure than diagonal rung-3 alignment, while class separability
drops back below the rung-3 peak — **holds under all four regularizer
variants**, including the scale-free ridge.

## Does the result support or contradict the thesis as written?

**Supports the mechanism, with one framing caveat that belongs in the
footnote.** By the plan's own criterion the ordering did not flip under a
scale-free ridge, so this is the "a footnote settles it" outcome rather than
the "Section 4.13.2 needs rewriting" outcome: the over-alignment trade-off
(class structure peaks at rung 3, then degrades when alignment is pushed to
full covariance whitening) is not an artifact of rung 4's unit-dependent
regularizer. However, the *sign* of the rung-4 silhouette is not robust. The
published `raw_lam1` value is negative (−0.0061), and any wording in
Section 4.13.2 that leans on "silhouette goes negative" / "whitening actively
destroys class structure" overstates it: the faithful CORAL analogue
(`prez_lam1`, standardize first as `run_coral_loso.py` does) and both
scale-free variants give a small *positive* silhouette (0.0095–0.0166). The
defensible claim is the relative one — rung 4 sits below the rung-3 peak on
both class separability and the subject probe — not the absolute sign. The
footnote should state that rung 4's negative silhouette is specific to the
raw-feature fixed ridge and that the ordering, not the sign, is what is
robust.
