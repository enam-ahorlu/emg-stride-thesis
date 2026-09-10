# Channel-dropout mechanism programme — consolidated report

**Run 1–2 September 2026 by Claude Code, on Enam's machine. All local, no network.**
Master run order: `RUN_ORDER_CD_MECHANISM.md`. Five stages plus the R-1 precursor.
**No `MSc Thesis.docx` or chapter file was edited.** Every wording change implied below is Enam's.

---

## Executive summary

| Stage | Question | Outcome | One line |
|---|---|---|---|
| **R-1** | Is the CNN pipeline run-to-run reproducible? | run-to-run SD **0.47 pp** | Not deterministic per fold; stable at the 40-fold mean. 0.787 vs 0.772 → canonical **0.772**. |
| **G3** | Does channel dropout flatten electrode reliance and make it transfer? | **N** (neither) | Both mechanism quantities move *significantly against* the §5.7 hypothesis. §5.7 must be softened. |
| **W-4** | Removal, or per-channel variation in general? | **P** (perturbation) | Gain jitter (never zero) matches channel dropout: −0.79 pp, p = 0.259. §5.7's "liftoff/re-siting" framing is too specific. |
| **W-3** | Which architectural property does the gain need? | **no letter** (M/D boundary) | CD works without skip connections (+4.55 pp) but is reliably ~2 pp larger with them (interaction p = 0.011). |
| **G2** | Does more enforced invariance eventually hurt? | **p = 0.2 stands** | No rate displaces 0.2. p = 0.5 is not clearly worse — the over-regularization parallel is **not** supported. |

**Net for the thesis:** channel dropout's benefit is real and large (+6.5 pp on the plain residual net), but none of the four mechanism hypotheses the programme tested survives intact — not SE (W-2 G1), not a transferable reliance profile (G3), not zeroing specifically (W-4), not the skip connection alone (W-3), and not an over-regularization shape (G2). The augmentation works; the *story* in §4.8 and §5.7 needs to become a narrower, measured claim about robustness to per-channel sensor variability on a deep-enough backbone.

---

## R-1 — CNN reproducibility precursor

**Run-to-run SD (40-fold mean), `resnet_se` / global norm: 0.47 pp.** Three identical runs: 0.7723 / 0.7667 / 0.7760.

- **Step 0 (config archaeology):** the 22 Jul AdaBN run (`results_adabn_chandrop`, `f1_pre_adabn_mean` 0.7874) and the W-1 fresh run (`results_win250_cnn_global`, 0.7723) are configured identically on every checkable flag, and the two drivers' fold code is line-for-line identical. `--augmentation none` escape does **not** apply.
- **Step 1 (pairing):** wide per-subject spread (SD 4.57 pp), balanced signs (24/16), r = 0.87 → nondeterminism signature.
- **Step 2 (three repeats):** SD 0.47 pp, range 0.93 pp; **0.787 sits ~3σ outside** the reproduced cluster.
- **Git + environment:** `run_cnn_arch_loso.py` and `cnn_architectures.py` byte-identical since the May–Jul squash; `train_cnn_loso.py` changed only in unused `main()` reporting; torch 2.10.0+cu126 / cuDNN 91002 unchanged since 14 Apr. No recoverable pre-squash refs. **Code drift is not the explanation.** The W-3 fresh plain-ResNet baseline later reproduced 0.7563 → 0.7600 (+0.37 pp), corroborating this.
- **Recommendation:** canonical = **0.772**; footnote the reproducibility bound (40-fold-mean run-to-run SD ≈ 0.5 pp; per-subject cuDNN spread up to ~15 pp). The defect is the thesis carrying *both* 0.787 and 0.772 for one quantity.
- **`torch.backends.cudnn.deterministic`** left `False` (per the plan) — noted as a recommendation for the reproducibility section only; enabling it would break comparability with every published number.

Full detail: `R1_CNN_REPRODUCIBILITY_REPORT.md`.

---

## G3 — occlusion: Outcome **N**

§5.7's mechanism paragraph is an interpretation the data does not reach and must be softened.

**Gates (4/4 pass):** `occlusion.csv` 360 rows each (40×9); `f1_full` matched per-fold `f1_macro` every fold (0 warnings); instrumented baseline **0.7684 vs 0.7600 = +0.84 pp**, inside the 1.5 pp R-1 gate; G0 RNG assertions never fired.

**Numbers** (`d = chandrop − no-aug`, per subject, paired; `--arch resnet`, SE-free, matching G1):

| quantity | Δ | 95% BCa | p (Holm) | d |
|---|---|---|---|---|
| between-subject consistency, Spearman | **−0.168** | [−0.220, −0.102] | 2.3e-6 | −0.89 |
| between-subject consistency, Pearson | −0.249 | [−0.309, −0.180] | 3.1e-8 | −1.19 |
| within-subject concentration, normalized (lower = flatter) | **+0.034** | [+0.018, +0.051] | 2.3e-4 | +0.64 |
| concentration, raw pp (scale-confounded — not the verdict metric) | −4.72 | [−5.53, −4.00] | — | −1.91 |

Randomization test (10k subject-level swaps, seed 42) agrees on consistency: p = 0.0005.

**Reading:** both mechanism quantities are significant but **in the opposite direction to the hypothesis** — channel dropout makes electrode reliance *less* consistent across people and *less* flat within a subject. Caveat: the chandrop profiles have **93/360 (26%)** negative `drop_pp` entries clipped to zero when normalizing (baseline 14/360), which per §3.1 makes the normalized concentration measure shaky — so the reversal is solid on consistency, weaker on concentration. **Does not establish** that channel dropout fails to help (it clearly does, +6.5 pp) — only that its benefit is not a transferable electrode-reliance profile.

---

## W-4 — gain jitter: Outcome **P** (perturbation)

Any channel-structured multiplicative perturbation does the work; zeroing is not special. §5.7's "electrode liftoff and re-siting" framing is too specific and should be reworded to per-channel sensor variability.

**Gates:** run has 40 rows, no duplicates; §2.2 inertness passed (all 5 existing aug modes byte-identical in both files); all three arms era-internal (current code).

| arm | mean | gain vs no-aug | 95% BCa | p | d |
|---|---|---|---|---|---|
| no augmentation | 0.7600 | — | — | — | — |
| channel dropout p=0.2 | 0.8251 | +6.51 pp | [+5.00, +7.98] | 2.3e-9 | +1.35 |
| gain jitter sd=0.4 | 0.8330 | +7.31 pp | [+6.02, +8.64] | 1.8e-11 | +1.71 |
| **chandrop − gainjitter** | | **−0.79 pp** | **[−2.16, +0.30]** | **0.259** | −0.20 |

Family of one, no Holm. §6 noise floor: two no-aug `resnet` repeats 0.7600 / 0.7684, SD 0.59 pp (consistent with R-1's 0.47 pp).

---

## W-3 — skip-connection ablation: **no pre-registered letter (M/D boundary)**

**§4 trainability check (first): PASS.** No-skip no-aug **0.7718 vs plain residual 0.7600 = +1.18 pp**, within 3.0 pp — both networks trained, comparison clean. The 6-block no-skip net scored slightly *higher*; no degradation problem at this depth.

**Gates:** both new runs 40 rows, no dups; subject index sets identical across all four runs; G1 arms re-read from disk; §2.1 inertness passed on `resnet` and `resnet_se` (RNG state + initial parameter bytes byte-identical). Param counts: `resnet` 546,020 · `resnet_se` 557,276 · `resnet_nores` 535,396 (1.95% / 10,624 smaller).

**§5 interaction** (per subject, paired, family of one):

| contrast | Δ | 95% BCa | p | d |
|---|---|---|---|---|
| CD on plain residual (G1) | +6.51 pp | [+5.00, +7.98] | 2.3e-9 | +1.35 |
| CD on no-skip | **+4.55 pp** | [+3.29, +5.95] | 5.1e-8 | +1.04 |
| interaction (res − no-skip) | **+1.96 pp** | [+0.22, +3.49] | **0.011** | +0.36 |

Arm means: plain-res no-aug 0.7600 · plain-res+CD 0.8251 · no-skip no-aug 0.7718 · no-skip+CD 0.8172.

**Why no letter:** `delta_nores` = +4.55 pp clears the 4.0 pp cutoff (so not **M**), but the interaction is significant with a CI excluding zero (so not **D**); not **K** (needs `delta_nores` < 2.0). Two partial readings, neither forced:
- *D-leaning:* channel dropout delivers +4.55 pp **without** skips — the skip is not *required*. SE and skips both ruled out as necessary; depth and capacity remain.
- *M-leaning:* the gain is reliably ~2 pp larger with skips (p = 0.011) — the skip contributes a real minority share.

§4.8.1's confound paragraph narrows either way, from three entangled candidates to "depth + capacity, with skips a partial contributor."

---

## G2 — dropout-rate sweep: **p = 0.2 stands**

**§4.1 reproduction gate: PASS** — fresh p=0.2 = **0.8376 vs published 0.8395, −0.19 pp** (gate ±1.5 pp; ±0.002 would fail ~67% by chance at SD 0.47 pp).

**Gates:** all 4 arms 40 rows, 0 dups; `occlusion.csv` 360 rows / 40 subjects each; 0 RNG/f1-match failures; `--arch resnet_se` (model of record); p=0.2 first as the era-internal comparator.

| rate | mean | vs p=0.2 | 95% BCa | Holm p | d |
|---|---|---|---|---|---|
| p = 0.1 | 0.8311 | −0.65 pp | [−1.46, +0.12] | 0.263 | −0.25 |
| p = 0.2 | 0.8376 | — | — | — | — |
| p = 0.3 | 0.8373 | −0.03 pp | [−0.71, +0.62] | 0.858 | −0.01 |
| p = 0.5 | 0.8205 | −1.71 pp | [−2.53, −0.74] | 0.001 | −0.59 |

**§4.3 verdict:** no rate displaces 0.2 — each either fails Holm significance (0.1, 0.3) or falls short of the 3.0 pp margin (0.5 is significant but only −1.71 pp). **No challenger, no escalation, no downstream artifact touched.**

**Mechanism shape:** the sequence 0.8311 → 0.8376 → 0.8373 → 0.8205 is **not** monotone-then-declining. p = 0.5 is not clearly worse than p = 0.2, so the over-regularization parallel to §4.13.2 is **not supported** — report the sweep as flat within noise, do not claim a shape.

---

## Consolidated implications for the thesis (Enam's wording)

1. **§4.8 / §4.8.1 — the "complements SE" clause** (W-2 G1, Outcome R): wrong. Channel dropout works essentially independent of SE (interaction −0.78 pp, p = 0.221). Correct before the viva.
2. **§4.8.1 — the three-candidate confound** (W-3): narrows to depth + capacity, with skip connections a partial contributor (~2 pp of ~6.5). SE and "skips alone" are ruled out as *necessary*.
3. **§5.7 — the mechanism paragraph** (G3 + W-4): must be softened and re-scoped. It is not a transferable electrode-reliance profile (G3: consistency moves the wrong way, significantly), and it is not about zeroing (W-4: gain jitter matches it). The defensible claim is robustness to per-channel multiplicative sensor variability on a residual backbone.
4. **§4.13.2 parallel** (G2): do **not** cite a channel-dropout over-regularization shape alongside the over-alignment result — the rate sweep is flat.
5. **R-1 — 0.787 vs 0.772**: canonicalise **0.772**, footnote the run-to-run bound.
6. **Reproducibility section**: recommend enabling `cudnn.deterministic` / `benchmark=False` for future work (not retroactively).

None of this touches the ensemble, the causal chain, the external validation or the 85.8 % headline.

---

## Wall-clock vs estimates (`RUN_ORDER` said ~6.3 h GPU total)

| Stage | Estimate | Actual |
|---|---|---|
| G3 (smoke + 1 instrumented 40-fold + analysis) | 26 min | ~30 min |
| W-4 (smoke + 1 40-fold + analysis) | 50 min | ~50 min |
| W-3 (smoke + 2 40-fold + analysis) | 1.3 h | ~1.3 h (25 + 49 min) |
| G2 (4 instrumented 40-fold + analysis) | 3.7 h | ~4.05 h (53–64 min each) |
| **Total** | **~6.3 h** | **~6.7 h** |

Plus R-1 Step 2 (~2.6 h, run earlier) and the W-2 G1 pair (~1.3 h).

---

## Artifact index

`results_g3_noaug_instr/`, `results_w4_gainjitter/` + `w4_gainjitter_pairs.csv`, `results_w3_nores_noaug/` + `results_w3_nores_chandrop/` + `w3_residual_pairs.csv`, `results_cd_rate_p{0.1,0.2,0.3,0.5}/` + `g2_rate_pairs.csv` + `g2_rate_tests.csv`, `results_w2_g1/`, `results_repro_250global_r{2,3}/`. Stats scripts: `g3_occlusion_stats.py`, `w4_gainjitter_stats.py`, `w3_residual_stats.py`, `g2_rate_stats.py`, `w2_g1_stats.py`. Inertness: `w3_inertness.py` + `w3_fp_before.json`, `w4_inertness.py` + `w4_fp_before.json`. Code changes (uncommitted): `cnn_architectures.py` (`use_residual`, `resnet_nores`), `run_cnn_arch_loso.py` + `train_cnn_loso.py` (`gainjitter` mode, `--instrument`).
