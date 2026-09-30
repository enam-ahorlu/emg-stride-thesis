# EXPERIMENT PLAN: preprocessing-audit follow-ups (S-1, S-2)

Raised by audit item **B10** in `04_Reviews_and_QA/METHODOLOGY_AUDIT_3SEP.md`. Queue after the currently running
work. Read that section first; this plan assumes it.

---

## STATUS (updated 2026-09-03)

- **S-2 (fs 1920 vs 2000 mismatch): COMPLETE. PASS, no escalation.**
  Step 2 column check: MNF and MDF columns of the `--fs 1920` file equal the old ones times 0.96 (= 1 / 1.041667);
  every other column byte-identical (MAV/RMS/WL/ZC/WAMP identical, SpectralPower ignores `--fs`).
  Step 3 SVM LOSO per-subject on the new features vs `results_loso_freq_persubj/*SVM*subjectwise.csv`:
  **all 40 subject macro-F1 values matched; largest absolute difference 0.000e+00 (byte-identical, not just FP noise).**
  Pre-registered expectation met. B10.2's invariance argument holds; no thesis edit, no FDR family touched.
  Artifacts: `results_s2_fs1920_svm_persubj/`, `results_locus/s2_outcome.json`.
- **S-1 (active-only STDUP re-run): COMPLETE (classical + deep). GRID OUTCOME H. No escalation.**
  - Windows: 13,643 total; DNS 4069 / STDUP 1982 (14.5%) / UPS 4550 / WAK 3042. Min subject-by-class cell 32,
    no cell < 30, so no count-check outcome-X.
  - Per-class F1, active-only, per-subject norm (published Freq-72 LOSO in parentheses):
    SVM  DNS 0.717 (0.677) | STDUP 0.844 (0.961) | UPS 0.762 (0.763) | WAK 0.733 (0.702).
    RF   DNS 0.718 (0.662) | STDUP 0.831 (0.960) | UPS 0.768 (0.754) | WAK 0.732 (0.709).
    Hierarchy STDUP > UPS > WAK > DNS holds identically for both models. STDUP drops ~0.12 and its margin over
    UPS compresses from ~0.20 to ~0.08, but it stays clearly top; the other three classes each improve slightly.
    => **Outcome H** (the plan's less-expected, thesis-favourable outcome): the biomechanical reading is not a
    rest-window artefact; ">>" should become ">".
  - Normalization delta (independent escalation gate): SVM per-subject 0.7667 vs global 0.6713, delta +9.54 pp
    (BCa [+7.73, +11.43], p = 3.5e-11, d = +1.56); RF 0.7656 vs 0.6990, delta +6.66 pp (BCa [+5.25, +8.53],
    p = 2.5e-10, d = +1.26). Published +6.90 / +5.10; drift +2.64 / +1.56 pp, both within the 3.0 pp tolerance,
    both still strongly positive. **Finding A is not class-definition-dependent. No escalation.**
  - Deep arm (step 4, `resnet_se` + channel dropout, per-subject, 40-fold LOSO on the active-only windows):
    mean macro-F1 0.8229 (n=40) - above both classical models (0.767 / 0.766) and the SimpleEMGCNN headline
    0.754, and close to the full-set model-of-record 0.840. Pooled per-class F1: DNS 0.804 | STDUP 0.875 |
    UPS 0.810 | WAK 0.782. Hierarchy STDUP > UPS > DNS > WAK - **STDUP top**, so H holds on the model-of-record
    family too; the deep model is barely affected by the class-definition change. (WAK/DNS tail order differs
    from the classical models, which is not part of the claim.)
  - Artifacts: `results_aonly_persubj/`, `results_aonly_global/`, `results_aonly_resnet_se_cd_persubj/`,
    `results_locus/s1_outcome.json`.
  - Watchdog note: the classical run was babysat by `s1_memory_watchdog.sh` (proactive kill + `--resume` on low
    RAM / sustained hard page-faults) after an operator error left two pipelines contending on a 16 GB box; ~39
    proactive restarts, all four runs finished `exit 0` at 40/40.

## Rules for whoever runs this

- **Do not edit any file in `01_Thesis/`.** Report numbers back; the write-up is handled separately.
- **Do not touch §4.17 or the unified FDR family.** New tests here are additions to be folded in later, not edits.
- **Do not change the deep model of record** (`resnet_se` + channel dropout, 0.840).
- Write every result under a new `results_*` folder. Do not overwrite an existing one.
- Log the exact command line for every run into the output folder. The external CNN run failed to do this and it
  is now an open reproducibility gap; do not repeat it.

---

## S-2 first, because it is 20 minutes and it de-risks S-1

`extract_features.py` was run on SIAT with its default `--fs 2000.0`. The true rate is **1920 Hz**. The argument
reaches only `feat_mean_freq`, `feat_median_freq` and `feat_spectral_power`; the third ignores it, and the first
two are exactly linear in fs. The claim to confirm is that **every classical result is unchanged**, because a
uniform positive scale on a feature is cancelled by the StandardScaler and by per-subject z-scoring, and RF is
scale-invariant.

1. Re-run `extract_features.py` on the existing 250 ms windows NPZ with `--fs 1920`, writing to a new tag
   (suggest `..._fs1920`). Do not overwrite `features_out/`.
2. Confirm numerically that the MNF and MDF columns of the new file equal the old ones divided by 1.041667, to
   float tolerance, and that every other column is identical.
3. Run `train_classical_loso.py` for **SVM only, per-subject norm**, on the new features. Compare the 40 subject
   macro-F1 values against `results_loso_freq_persubj/*SVM*subjectwise.csv`.

**Pre-registered expectation:** identical to floating-point noise. If they are not identical, stop and report,
because that would mean something other than a scale factor changed and the invariance argument in B10.2 is wrong.

**Deliverable:** one line stating whether the 40 values matched, and the largest absolute difference.

---

## S-1: the active-only STDUP re-run

### Why

STDUP as constituted is **86.5% rest windows**, while WAK, UPS and DNS contain only gait-phase-labeled locomotion.
Mean envelope amplitude alone separates STDUP from the other three at **AUC 0.90**; restricted to active windows
that falls to **0.47, which is chance**. The reported class hierarchy STDUP >> UPS > WAK > DNS is therefore shaped
by the class definition, and the thesis currently explains it biomechanically. The disclosure is already written
into §3.2.4, §4.3 and §5.3. This experiment is what would let the thesis report a hierarchy that does not depend
on that decision.

### What to run

**Step 1. Rebuild the windows with active-only STDUP.**
`preprocess_emg.py` already has the switch: `PreprocessConfig.keep_only_active_stdup`, default `False`. Set it
`True` (or expose it as a CLI flag, which is cleaner). Everything else identical: 20 to 450 Hz 4th-order
zero-phase, 50 ms envelope, **250 ms** windows, 50% overlap, 0.60 purity, same four movements, all 40 subjects.
Write to a new tag, suggest `..._w250_ov50_conf60_Aonly`.

Expected shape, from the current metadata: **13,643 windows**, of which 1,982 STDUP (14.5%). Report the actual
per-subject class counts and flag any subject-by-class cell below 30 windows, since the LOSO fold for that
subject would then be evaluating a class on very little test data.

**Step 2. Extract features.** Same feature set (`freq`, 8 per channel, 72 dimensions). **Pass `--fs 1920`.**

**Step 3. Classical LOSO, four runs.** `train_classical_loso.py`, unchanged protocol: strict 40-fold LOSO, nested
`GridSearchCV` with the 5-fold `GroupKFold` inner loop, same grids.

| run | model | norm | out folder |
|---|---|---|---|
| 1 | SVM | per-subject | `results_aonly_persubj` |
| 2 | RF  | per-subject | `results_aonly_persubj` |
| 3 | SVM | global      | `results_aonly_global` |
| 4 | RF  | global      | `results_aonly_global` |

This is roughly half the windows of the main run, so it should be cheaper than the original. CPU only.

**Step 4. Deep arm, one run, only if step 3 completes cleanly.** `resnet_se` + channel dropout, per-subject norm,
40-fold LOSO, on the same active-only features/windows. Output `results_aonly_resnet_se_cd_persubj`. One GPU run.
If time is short this step can be dropped; the classical arm answers the question on its own.

### What to report

1. **Per-class F1** for SVM and RF, per-subject norm, active-only, next to the current published per-class F1.
   This is the primary output. State the resulting hierarchy explicitly.
2. **Macro-F1** for all four runs, and the **per-subject minus global delta** for each model, with paired Wilcoxon,
   BCa 95% CI and paired Cohen's d, matching how the thesis reports every other delta.
3. Per-subject class counts from step 1, and any cell flagged in step 1.

### Pre-registered outcome grid, decided before the runs

- **Outcome C, composition.** STDUP is no longer top of the hierarchy under the active-only definition. The
  corrected wording now in §4.3 and §5.3 stands as written, and Chapter 4 gains a short subsection reporting both
  hierarchies. **This is the expected outcome given AUC 0.47.**
- **Outcome H, hierarchy holds.** STDUP stays top even when restricted to active windows. Then the biomechanical
  reading was right after all, the current correction is over-stated, and §4.3 and §5.3 get a scoped version of the
  original claim plus this evidence. This would be a genuinely good result for the thesis and should be reported
  as prominently as Outcome C.
- **Outcome M, mixed.** STDUP falls but stays above one of UPS, WAK or DNS. Report the ordering as measured and
  make no causal claim beyond it.

**Independently of C, H or M**, report the normalization delta. The expectation is that per-subject normalization
still beats global by a similar margin, because Finding A does not depend on the class definition. **If that
expectation fails, stop and escalate**, because it would mean the headline finding is partly carried by the rest
windows, which nothing in the audit currently suggests.

### What this experiment cannot do

It does not re-run ENABL3S, the ensemble, the causal-buffer analysis, the alignment ladder, the channel-dropout
programme or the window-length ablation. It is a class-definition sensitivity check on the classical LOSO tier and
should be reported as one. Do not restate the 85.8% ensemble headline on this basis, and do not re-run it.
