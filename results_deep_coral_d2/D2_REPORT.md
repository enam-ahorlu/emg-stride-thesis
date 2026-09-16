# D2 lambda sweep — final report

Run 15-16 September 2026, per the 15-Sep AMENDMENT to `docs/EXPERIMENT_PLAN_DEEPCORAL.md`.
GPU: NVIDIA RTX 4050 Laptop (6.44 GB). Reported in the order the amendment specifies.

## 1. Reproduction gate

Re-ran lambda = 1.0 into `results_deep_coral_lam1p0_repro` (fresh directory, published
`results_deep_coral_chandrop` untouched). **Mean = 0.832924** (n=40), vs published
**0.825712**. Offset **+0.7212 pp** — well inside the 1.5 pp bound. **PASS.** The sweep is
comparable to the published arm.

## 2. Single-fold timing and projection

`--heldout 1` into a scratch directory: **443 s** (run concurrently with the reproduction
gate on the same GPU, so likely inflated by contention on top of the plan's own caveat
that `--epochs 40` is a ceiling with patience-7 early stopping, making a single fold an
upper bound, not an average).

Projection at the time: one arm ≈ 443 × 40 = 17,720 s ≈ **4.92 h**; Stage A (3 arms) ≈
**14.77 h** — over the plan's ~8 h stop line. Reported and paused per the plan's explicit
rule ("that is Enam's call to make, not the runner's"). **Enam's decision: proceed with
the full 3-arm Stage A anyway.** In practice the reproduction gate itself (uncontended
after the timing job finished) ran at roughly 200 s/fold, i.e. the single-fold timing was
indeed an overestimate, as anticipated — the realised cost of the full 6-arm sweep (240
folds total) was far under the original 14.77 h projection.

## 3. Comparator check

`results_cnn_aug_resnet_se_chandrop/cnn_arch_subjectwise.csv`: mean f1_macro =
**0.839490** to six decimals, 40 unique subjects, no duplicates. Confirmed this is NOT
`results_cnn_aug_resnet_se_chandrop_proba` (0.841566) before any contrast was computed.
Verification is in `PREDICTION.md`, written before the first run started.

## 4. Sweep table

| lambda | mean f1_macro | SD | n |
|---|---|---|---|
| 0.1 | 0.832075 | 0.060996 | 40 |
| 1.0 (reproduction) | 0.832924 | 0.065280 | 40 |
| 3.0 | 0.833654 | 0.064816 | 40 |
| 10.0 | 0.831717 | 0.066439 | 40 |
| 30.0 | 0.834703 | 0.062852 | 40 |
| 100.0 | 0.834530 | 0.068523 | 40 |

References: global-norm baseline ≈ 0.772 (not re-run; lambda=0 degenerate case, structural
per the method, not measured fresh in this sweep), per-subject normalization 0.839490,
AdaBN 0.818204 (`results_adabn_chandrop/adabn_subjectwise.csv`).

Every arm's 40 subjectwise `f1_macro` vectors were confirmed pairwise non-identical before
any test was admitted to the BH family (no two arms are bit-identical — unlike the FILTER
chain's envelope arms).

## 5. Stage B: triggered

Trigger rule: "if lambda = 10 lands within 0.5 pp of lambda = 1, or above it, the peak is
not bracketed." Lambda=10 (0.831717) sits **0.121 pp below** lambda=1 (0.832924) — well
inside the 0.5 pp band, so the peak was **not bracketed** by Stage A. Ran lambda=3 and
lambda=30 (Stage B) per the pre-registered rule.

## 6. Paired contrasts (Holm within family of 11; BCa 95%; n_improved out of 40)

| contrast | mean delta (pp) | Cohen's dz | BCa95 (pp) | n improved | raw p | Holm p | sig | x 0.5pp | clears band |
|---|---|---|---|---|---|---|---|---|---|
| λ=0.1 vs per-subject | −0.741 | −0.186 | [−1.94, 0.50] | 16 | 0.265 | 1.000 | No | −1.48 | No |
| λ=3.0 vs per-subject | −0.584 | −0.130 | [−1.96, 0.78] | 15 | 0.452 | 1.000 | No | −1.17 | No |
| λ=10.0 vs per-subject | −0.777 | −0.158 | [−2.27, 0.80] | 17 | 0.314 | 1.000 | No | −1.55 | No |
| λ=30.0 vs per-subject | −0.479 | −0.091 | [−2.30, 0.97] | 19 | 0.545 | 1.000 | No | −0.96 | No |
| λ=100.0 vs per-subject | −0.496 | −0.085 | [−2.41, 1.12] | 16 | 0.501 | 1.000 | No | −0.99 | No |
| λ=0.1 vs λ=1 | −0.085 | −0.031 | [−0.86, 0.81] | 19 | 0.581 | 1.000 | No | −0.17 | No |
| λ=3.0 vs λ=1 | +0.073 | +0.022 | [−0.89, 1.16] | 18 | 0.910 | 1.000 | No | +0.15 | No |
| λ=10.0 vs λ=1 | −0.121 | −0.032 | [−1.17, 1.10] | 19 | 0.858 | 1.000 | No | −0.24 | No |
| λ=30.0 vs λ=1 | +0.178 | +0.042 | [−1.44, 1.24] | 18 | 0.900 | 1.000 | No | +0.36 | No |
| λ=100.0 vs λ=1 | +0.161 | +0.040 | [−1.57, 1.11] | 23 | 0.259 | 1.000 | No | +0.32 | No |
| **best λ=30 vs AdaBN** | **+1.650** | **+0.373** | **[0.20, 2.92]** | 27 | **0.00558** | **0.0613** | No | **+3.30** | No |

Every contrast against per-subject normalization and against lambda=1 is small,
non-significant, and its BCa interval straddles zero (does not clear the 0.5 pp
nondeterminism band in either direction). No lambda is distinguishable from the published
fixed setting or from per-subject normalization on this sweep. The one contrast with a
notable raw p (best lambda vs AdaBN, p=0.0056) does not survive Holm correction within
this 11-test family (p_holm=0.061).

## 7. Interior-maximum prediction

**Did not hold as specified.** The predicted shape was a rise from the global-norm
baseline (~0.772) to lambda=1 (confirmed structurally: 0.833 ≫ 0.772) followed by a
**fall** as lambda increases further, from over-alignment. Instead, macro-F1 is
**essentially flat across three orders of magnitude** (0.1 to 100): every arm sits within
about 0.3 pp of every other, well inside run-to-run noise (SD ≈ 0.06-0.07 per arm). The
nominal peak (λ=30, mean 0.834703) sits *above* lambda=1, but by an amount indistinguishable
from noise (contrast λ=30 vs λ=1: +0.178 pp, p=0.90). **The honest reading is not "the peak
is above lambda=1" but "no peak is detectable at all in this range" — the over-alignment
account predicts a fall that this sweep does not show, at least not within [0.1, 100].**
This is itself worth stating plainly: it does not confirm the over-alignment story on this
backbone (unlike the classical CORAL sweep and the alignment ladder), but it also does not
contradict it, since Deep CORAL never approaches the global-norm floor either — the whole
tested range sits in a broad, flat, statistically indistinguishable plateau.

## 8. Outcome

**OUTCOME B.** The best lambda (30) closes the gap to per-subject normalization to −0.479
pp — inside the 0.5 pp band, Holm p = 1.000, BCa interval straddles zero. **Per-subject
normalization and a tuned Deep CORAL are statistically indistinguishable on this backbone.**
Per the amendment: report and stop; do not edit any chapter. The abstract's "edges its deep
variant by about a point" and the corresponding §4.7/§5.8/§6.2 sentences need to come out
in the write-up wave — not done here.

Framing, as instructed: this narrows one sentence about one backbone. It does not touch
the normalization finding's spine — consistency across three model families, the classical
CORAL sweep (four orders of magnitude), the alignment ladder, ENABL3S, and the active-only
control all stand untouched by this result.

## 9. v8 correction family

`recompute_unified_fdr_v8.py`, run from v7's 218 tests / 153 survivors (A_all_reported
scope). **11 new tests admitted** (every λ-vs-per-subject and λ-vs-λ1 contrast for the 5
new lambdas, plus best-λ-vs-AdaBN), none excluded — all six sweep arms confirmed pairwise
non-identical before admission. **v8: 229 tests, 154 survivors** (A_all_reported);
226 tests, 153 survivors (B_bearing).

- **Nothing changed side.** Every test that survived BH in v7 still survives in v8; every
  test that failed still fails.
- Of the 11 new D2 tests, **only one survives BH**: best λ=30 vs AdaBN (raw p=0.0056,
  BH=0.0094). All ten λ-vs-per-subject and λ-vs-λ1 contrasts are clearly non-significant
  (BH p ranging 0.33-0.93).
- **The two boundary contrasts the amendment named by name** (RF's ENABL3S replication,
  raw p=0.0371; AdaBN's pre-to-post contrast on the same cohort, raw p=0.0371) both moved
  from p_BH=0.0522 (v7, NS) to p_BH=0.0545 (v8, NS) — **further from significance, not
  closer**, exactly as the amendment predicted growing the family would do. Neither
  crosses 0.05 in either direction.
- The three original D1 contrasts (per-subject vs Deep CORAL λ=1, per-subject vs AdaBN,
  Deep CORAL λ=1 vs AdaBN) were confirmed already present in v7 under family "D-1 CNN-side
  adaptation (4.7)" before this script ran, and were **not** re-added.

Outputs: `report_figs/new_experiments/unified_fdr_family_v8_{A_all_reported,B_bearing}.csv`.
