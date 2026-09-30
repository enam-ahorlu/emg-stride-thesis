# RUN QUEUE: everything still to be executed, in order

**This is the dispatcher, not the designs.** Each item points at the plan file that specifies it. Do not run
anything from a design not listed here, and do not reorder without saying so in the report.

Last updated: 3 September 2026.

---

## Rules that apply to every item below

- **Do not edit any file in `01_Thesis/`.** Report numbers; the write-up is handled separately.
- **Do not touch §4.17 or the unified Benjamini-Hochberg family.** Every new test here is an addition to be folded
  in later, not an edit to the existing family.
- **Do not change the deep model of record** (`resnet_se` + channel dropout, 0.840).
- **Every run writes to a NEW `results_*` folder.** Never overwrite an existing one.
- **Every run logs its exact command line into its own output folder.** The external ENABL3S CNN run did not, and
  that is now a standing reproducibility gap. Do not add another.
- Report against each item's pre-registered grid. A null or negative result is a result; say so plainly rather
  than reaching for a favourable reading.

---

## Status

**ALL ITEMS COMPLETE - 4 September 2026.** Consolidated results in `RUN_QUEUE_REPORT.md`.

| item | state | grid outcome |
|---|---|---|
| P-9 (4 arms) | done | backbone **S**, transfer **U** (both backbones) |
| P-10 (upward sweep) | done | **B** (boundary near SD 0.80; SD 1.00 arm = X) |
| B1 (manifests + cross-check) | done | 142/143 dirs carry an UNKNOWN core field; one verifiable contradiction = the known backbone defect |
| B2 (literature band) | done | "band changes / fails to populate"; **section 2.5 open for Enam** |
| B3 (5-point correlations) | done | section 3.2 fix works (40x5 paired); **section 3.4 open for Enam** |
| B4 / B5 (result files + hygiene) | done except 400 ms ladder | **section 4.5 open for Enam** (recommendation: delete the 4 stale columns) |
| B8 (subject-dependent protocol) | done | **RANKING FLIPS + CONTAINED (marginal)** |
| S-2 (sampling-rate invariance) | done | 40/40 byte-identical; no escalation |
| S-1 (active-only STDUP) | done (classical + deep arm) | **H** (hierarchy holds); normalization delta within tolerance, no escalation |

Three decision points remain for Enam: **B2 section 2.5, B3 section 3.4, B4 section 4.5.** No escalation triggers
fired.

---

## Block 1. Finish the mechanism programme. GPU.

### 1.1 P-9 arm 4 completion
Let it finish. It needs its summary CSV written; the other three arms have one and it does not yet.
Then report all four arms together against the P-9 grid.
**Plan:** `EXPERIMENT_PLAN_LOCUS.md` §3, and note §3.5a, the reproduction trap, which must be settled before
reading the numbers rather than after.

### 1.2 P-10, the upward sweep
3 runs, roughly 2.5 hours. Mean-preserving channel dropout, **not** gain jitter; §4.2 of the plan explains why.
**Plan:** `EXPERIMENT_PLAN_LOCUS.md` §4.

**After 1.2 the mechanism programme is closed.** Everything below is audit remediation.

---

## Block 2. Cheap audit items. CPU only, no GPU, no model runs.

Do these before Block 3, because B1 gives every later run a manifest and the later runs are the ones that most
need one.

### 2.1 B1: run manifests
Build `build_run_manifest.py`, then run the cross-check. This is the root cause of the backbone
misattribution, so the cross-check is the point of the exercise, not the script.
**Read §1.3 carefully:** three confidence states, never a guess.
**Plan:** `EXPERIMENT_PLAN_AUDIT_REMEDIATION.md` §1.

### 2.2 B2: the literature comparison band
Enumerate the band properly and then answer the question honestly. §2.5 is a decision point for Enam, so stop
there and report rather than deciding it.
**Plan:** `EXPERIMENT_PLAN_AUDIT_REMEDIATION.md` §2.

### 2.3 B3: the five-point correlations in §4.13.2
Try the substantive fix in §3.2 first; it is free. Fall back to §3.3 only if that does not work. §3.4 is a
decision point for Enam.
**Plan:** `EXPERIMENT_PLAN_AUDIT_REMEDIATION.md` §3.

### 2.4 B4 and B5: incomplete result files in the public repo, plus hygiene
Covers the alignment ladder, the known-and-still-unfixed `results_win400_ladder` defect, the two code robustness
notes, and, added since the plan was written, **two more hygiene items from the later audit passes**:
- the bare `except Exception` in `build_full_dataset` (`preprocess_emg.py`), which swallows a failed
  subject-by-movement cell into a printed message. No cell was actually lost, but it should raise or record.
- the external CNN runs saved no configuration at all, so epochs, learning rate, batch and seed exist only in
  command history. Fold this into the same pass as the manifest work in 2.1.
§4.5 is a decision point for Enam.
**Plan:** `EXPERIMENT_PLAN_AUDIT_REMEDIATION.md` §4, and §4.3 to §4.4 for B5.

---

## Block 3. The re-runs.

### 3.1 B8 fullest: the subject-dependent protocol
Enam chose the fullest option. **Read §4A.1 before writing any splitting code**: `t_start` restarts for every
movement, so contiguous blocking on a subject's t_start order collapses whole folds onto one class. The design is
per-movement blocking with a guard band.
No LOSO is involved. Classical is CPU seconds per cell; the CNN arm is well under one LOSO run.
**Plan:** `EXPERIMENT_PLAN_AUDIT_REMEDIATION.md` §4A.

### 3.2 S-2: confirm the sampling-rate invariance
20 minutes, CPU. Feature extraction ran at `--fs 2000` when the true rate is 1920 Hz. The argument that no result
changed is provable; this confirms it empirically because an examiner may ask.
**If the 40 values do not match, stop and escalate**, because that would mean the invariance argument is wrong.
**Plan:** `EXPERIMENT_PLAN_PREPROC_AUDIT.md`, S-2.

### 3.3 S-1: the active-only STDUP re-run
Do S-2 first. Four classical LOSO runs on CPU, plus one optional GPU run. STDUP is currently 86.5% rest windows
while the three locomotion classes are locomotion only, so the published class hierarchy is shaped by the class
definition. This is the run that gives a hierarchy independent of it.
**Report the normalization delta regardless of what happens to the hierarchy**, and escalate if that delta moves.
**Plan:** `EXPERIMENT_PLAN_PREPROC_AUDIT.md`, S-1.

---

## Not in this queue, and not for the executor

These are write-up items handled in the thesis directly. Listed so nobody picks them up by mistake.

- The backbone attribution fix in §4.8.2, §5.7 and §6.2.
- Recomputing the unified FDR family from 145 to its final count once every experiment above has landed.
- Exhibits for W-5, P-1, P-3 and the parity and locus stages.
- The voice pass on the prose written since 30 August.
- The 59 em dashes in the public repo's markdown documents.
