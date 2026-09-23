# Plan: kill-critic 23 September, deployment and benchmark stages (KC-S1 to KC-S3)

**Phase 0/1 status header (23 September 2026, this session; plan text below unedited):**
- **KC-S2 F0 feasibility (adapter extension only):** the `adapt_external_dataset.py` code change (circuit + per-circuit t_start, gated behind `--with-circuit-meta`) is done and inertness-proven byte-identical (npz hash match, meta `DataFrame.equals` True). The transition-count gate itself (F-OK/F-MARGINAL/F-X) has NOT been computed -- `kc23_s2_f0_feasibility.py` (the gate script) does not exist yet; queued.
- **KC-S1/KC-S3:** not run. `run_scripted_supervised.py` (S1) and the S3 inventory script do not exist yet -- new scripts per the plan, out of Phase-1 scope, queued with a "not yet runnable" note in `kc23_jobs_gpu.csv`/`kc23_jobs_cpu.csv`.

**Status:** ready to run. Written 23 September 2026 from `04_Reviews_and_QA/KILL_CRITIC_2026-09-23.md`.
**Execution:** local, through Claude Code. Dispatcher: `RUN_ORDER_KC23.md`.
**Cost:** about 8 to 10 GPU hours and about 5 CPU hours.
**Owner decision points:** S1 outcome D-S (a claim change); S2 F0 feasibility (if marginal).

| Stage | Kill-critic item | Question |
|---|---|---|
| KC-S1 | M3 | Once the commissioning buffer has to be scripted, do its labels beat the label-free pipeline on the same windows? |
| KC-S2 | moderate 4 and 5, limitations in 5.3 | How does the locked pipeline behave at real mode transitions on ENABL3S: decision delay, transition errors, smoothing, and the 250 against 400 ms trade? |
| KC-S3 | moderate 3 | A complete active-only benchmark table, all families, both normalizations |

The rules of Section 0 in `EXPERIMENT_PLAN_KC23_CLASSICAL.md` and `EXPERIMENT_PLAN_KC23_DEEP.md` apply here unchanged: interpreter, no thesis edits, no v9 changes, model of record fixed, new directories, command logs, `run_config.json`, inertness before new code runs, outcome letter first, and the halt protocol.

---

## KC-S1. Scripted buffer: label-free against supervised on the same windows (M3). About 3 h GPU, about 2 h CPU.

### S1.1 Problem

The deployable 81.7% needs a buffer covering all four movements. A single-activity buffer collapses the ensemble to between 36.7% and 61.3% (Section 4.4.3), and Section 3.2 describes the buffer as a scripted commissioning step. If the device prescribes the movements, the labels of those windows are known. The fair deployable competitor is supervised use of the same windows, and it was never run.

### S1.2 Protocol, fixed now

For each held-out subject, the scripted buffer B_K is the first K windows of each movement's own recording (the `balanced25` arm of `run_buffer_composition.py` at K = 25), for K ∈ {5, 10, 25}.

Every arm uses the same two rules:

- it normalizes the held-out subject from B_K only (causal);
- it is scored on the same set, **all windows not in B_25**, so that every K is scored on identical windows.

### S1.3 Arms

| Arm | Labels used | Model |
|---|---|---|
| L0 | none | label-free soft vote: SVM plus ResNet-SE+CD (reproduces `balanced25`) |
| L1 | none | label-free ResNet-SE+CD alone |
| L2 | none | label-free SVM (probability route) |
| S-pool | B_K labels | SVM trained on source plus B_K, using the regime-C pooling protocol of `run_within_subject_baseline.py` unchanged (weighting as published there) |
| S-only | B_K labels | SVM on B_K alone. Report it, but expect it to be weak at small K |
| S-ft | B_K labels | ResNet-SE+CD fine-tuned on B_K with the regularized three-epoch schedule of `run_cnn_calibration_multidraw.py` (`--ft-epochs 3`, `--ft-lr 5e-4`), **starting from the same base model as L1** |
| S-ens1 | B_K labels | soft vote of S-ft and S-pool |
| S-ens2 | B_K labels | soft vote of S-ft and L2 |

Pairing S-ft with L1 inside each fold puts the fine-tune gain inside one training realization, so run variance cannot contaminate it. The base training runs at 3 realizations (seeds 42 re-run, 7, 123) so that the cross-arm contrasts can also be read against KC-D1's variance.

### S1.4 Code

New script `run_scripted_supervised.py`, reusing `run_buffer_composition.py` and `run_cnn_calibration_multidraw.py` functions (import, never copy).

**Reproduction gate:** L0 at K = 25 reproduces the published `balanced25` per-subject F1 exactly for the SVM part, and within the KC-D1 run band for the ensemble. Fail means stop.

### S1.5 Endpoint and outcomes

**Primary endpoint:** at K = 25, the better of S-ens1 and S-ens2 (chosen on the realization average, and reported as such) against L0. Paired over 40 subjects, realization-averaged.

| Letter | Condition | Reading |
|---|---|---|
| **D-S** | Supervised ahead by ≥ 1.0 pt, significant, on ≥ 25 of 40 subjects | **ESCALATE (claim change).** "Without labeled calibration" holds offline only. Online, a scripted commissioning session should use its labels. Affects the abstract, Sections 1.4.1, 4.4.3, 4.7 ("Plan the calibration step"), 5.1 (O4), 5.5 and Figure 4.6. Report the recommended wording for Enam's text block |
| **D-T** | Within ±1.0 pt | Labels add nothing once the scripted buffer exists, which is a new and useful result for the pipeline. Section 4.4.3 gains a sentence |
| **D-L** | Label-free ahead by ≥ 1.0 pt | Report; it strengthens the label-free case |

**Secondary:** the K curve (the smallest K at which the best supervised arm reaches L0), and S-ft against L1 at each K (the pure fine-tune gain under causal normalization).

**Output:** `results_kc23_s1_scripted/` and `S1_VERDICT.md`.

---

## KC-S2. Real transitions on ENABL3S: errors, decision delay, smoothing, and the window trade. About 3 to 4 h mixed.

### S2.1 Problem

SIAT-LLMD records one sustained trial per movement. The thesis therefore could not measure behaviour at real transitions, could not validate the five-window vote (Section 4.4.3), and argues the 250 against 400 ms choice from an upper-limb latency figure (Section 4.4.6).

ENABL3S circuits are continuous recordings with a per-sample `Mode` label (`adapt_external_dataset.py`), so transitions between level walking and the stairs exist in the data.

### S2.2 F0 feasibility gate, first

1. The published window metadata has no circuit identifier. Extend `adapt_external_dataset.py` **additively** to write `circuit` and a per-circuit `t_start` into a new meta file under a new tag.
2. Prove the windows npz byte-identical to the published one, and the shared meta columns identical.
3. Count, per subject, the retained transitions of each type after the published drops (ramps, standing, sitting and stand-to-sit are dropped): WAK→UPS, UPS→WAK, WAK→DNS and DNS→WAK.

| Letter | Condition | Reading |
|---|---|---|
| **F-OK** | At least 5 of each type for at least 8 of 10 subjects | Run S2 |
| **F-MARGINAL** | Fewer | Report the counts and **ask Enam** before running |
| **F-X** | Transitions are not recoverable (for example, the ramp drop removes every stair entry) | Stop S2. The limitation stays in Section 5.3 |

Define a transition as the sample where the `Mode` label changes between two retained classes, with no dropped mode in between. A change that passes through a dropped mode is not a transition for this analysis.

### S2.3 Predictions needed

Per-window LOSO predictions, with circuit and time, for the locked SVM, the locked ResNet-SE+CD and the locked soft vote on ENABL3S, under:

- transductive per-subject normalization;
- the causal 100-window buffer. Also score the scripted `balanced25` buffer, for parity with KC-S1.

Re-run the ENABL3S LOSO with `--save-preds` and `--save-proba` if the published runs did not save them. The models and flags are exactly the published ones.

### S2.4 Measures

| Measure | Definition |
|---|---|
| Steady-state error | windows more than 2 s from any transition |
| Transition-zone error | windows whose end time lies within ±1 s of a transition, by transition type |
| Critical errors at transitions | DNS→WAK confusions inside transition zones against steady state |
| Decision delay | from the true change to the end time of the first of 3 consecutive correct predictions. Window end time is the reference, so the window's fill time is included. Median and IQR per transition type |
| Causal five-window vote at real transitions | change in steady-state error, change in transition-zone error, and added decision delay |
| **S2b, the window trade** | build ENABL3S windows at 400 ms (the same adapter, a new tag), train the locked ResNet-SE+CD and SVM at 400 ms, and report accuracy and decision delay at 250 against 400 ms. This measures the latency cost Section 4.4.6 argues from the literature |

### S2.5 Pre-registered readings (no halting letters; S2 is descriptive)

- If the vote cuts steady-state error and adds under 250 ms median delay, Section 4.4.3's smoothing paragraph can be stated for real transitions, with numbers. Otherwise, the limitation stays and gains a number.
- If 400 ms adds at least 100 ms median delay for less than a 2-point accuracy gain, Section 4.4.6's trade is supported by a lower-limb measurement. Otherwise, it is reported as measured.

**Output:** `results_kc23_s2_transitions/` and `S2_VERDICT.md`.

---

## KC-S3. The active-only benchmark, completed (moderate item 3). About 3 h GPU, about 2 h CPU.

### S3.1 Problem

If the thesis offers "the first reproducible LOSO benchmark" on SIAT-LLMD, the benchmark others reuse should not rest on a class that is 86.5% rest windows. Active-only numbers exist for some cells only (`s1_active_only_stats.py`).

### S3.2 Design

1. Inventory the existing active-only results from `RUN_MANIFEST.csv` and the S-1 outputs, as a table of family × normalization × present or missing.
2. Run only the missing cells, on `windows_..._w250_ov50_conf60_Aonly.npz` and the matching `freq_fs1920_..._Aonly_features_*`, from this set:
   - classical: LDA, SVM and RF (plus SVM-X and HGB if KC-C3 lands P2 or P3);
   - deep: SimpleEMGCNN, ResNet-SE and ResNet-SE+CD, under global and per-subject normalization;
   - the ensemble.
3. Include the DNS→WAK critical-error rate for each cell.

### S3.3 Output

A single benchmark table (`results_kc23_s3_active_benchmark/benchmark_active_only.csv`) and `S3_VERDICT.md`, reporting any change in the class hierarchy for families not previously run. There are no halting letters.

---

## Outputs and report-back

As in the other two plans. New paired tests go to `kc23_new_tests.csv`. Add a status header per stage to the top of this file, leaving the plan below unedited.
