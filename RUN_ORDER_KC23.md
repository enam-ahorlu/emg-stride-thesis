# Run order: the kill-critic programme of 23 September 2026

**This is the dispatcher, not the designs.** Each item points at the plan that specifies it. Do not run anything that is not listed here, and do not reorder without saying so in the report.

**Designs:**

- `EXPERIMENT_PLAN_KC23_CLASSICAL.md` (KC-C1 to C7)
- `EXPERIMENT_PLAN_KC23_DEEP.md` (KC-D0 to D5)
- `EXPERIMENT_PLAN_KC23_DEPLOYMENT.md` (KC-S1 to S3)

The consolidation stages (KC-F1 to F4) are specified below.

**Source critique:** `04_Reviews_and_QA/KILL_CRITIC_2026-09-23.md`. **Master checklist:** `KC23_TRACKLIST.md` (project root). Tick experiment items there as they close.

**Total:** about 80 to 90 GPU hours serial for the original stages, plus up to about 72 GPU hours for KC-D6 (staged; a failed manipulation check stops its later seeds), and about 55 to 65 CPU hours, much of it concurrent with the GPU queue. The order front-loads the answers that carry the most weight, so an interruption costs the least.

---

## 1. Standing rules for every item

- **Interpreter:** `06_Code/.venv/Scripts/python.exe`. The `jobs_*.txt` files are stale.
- **No thesis edits.** Nothing in `01_Thesis/` changes. Report numbers; wording is Enam's, through the text blocks.
- **The v9 family and the model of record are frozen.** New tests go to `kc23_new_tests.csv`; the family is handled once, at KC-F1.
- **New directories only** (`results_kc23_*`), each with its command line logged and `run_config.json` written.
- **Inertness before new code runs.** Byte-identical on existing paths, asserted. Failure stops everything downstream of that code.
- **Outcome letter first.** A null is a result. Do not look for a reading that rescues a hypothesis.
- **Decision rules are encoded in each stage's stats script** and printed, so they cannot drift after the numbers are seen.
- **Halt protocol.** An ESCALATE outcome writes `06_Code/KC23_HALT.md` (stage, letter, numbers, affected thesis passages) and appends a line to `KC23_STATUS.md`. The queue runner stops the dependent stages and continues independent ones, per the dependency table in Section 3.
- **Resume hygiene.** Check every `--resume` for duplicated subject rows before trusting a summary.
- **Wall-clock.** Record it for every run; the estimates in the plans get replaced by measurements.
- **Git.** Commit code changes and verdict files in logical units. **Commits are authored by Enam alone, with no Co-Authored-By or Claude-Session trailers.** Do not commit raw results directories that `.gitignore` excludes. Do not push without Enam.
- **Repository documents carry no em dashes.** This is Enam's standing rule, and these files are public.
- **Machine care.** Laptop GPU with 6 GB: one GPU job at a time. CPU jobs run beside it with worker counts capped so two logical cores stay free. Run GPU jobs detached, with a PID file, so a closed terminal does not kill them.
- **Data on OneDrive.** The raw datasets sit inside OneDrive (49 GB). Do not move them. If file access is slow, report it rather than relocating anything.

---

## 2. The queue runner

Before Phase 2, build `kc23_queue.py`, a small resumable runner:

- It reads `kc23_jobs_gpu.csv` and `kc23_jobs_cpu.csv`. Columns: job_id, stage, seed, command, out_dir, depends_on, gate_script.
- It runs one GPU job and at most one CPU job at a time, detached, logging to `_run_logs/kc23/<job_id>.log`.
- It writes `KC23_STATUS.md` after every job: done, running and queued, wall-clock, and last outcome letters.
- After a job whose `gate_script` is set, it runs that script. The script exits 0 (continue), 10 (report, continue) or 20 (ESCALATE: halt dependents).
- It skips any job whose `out_dir` already holds a complete summary, and verifies completeness, so a restart is safe.
- It never deletes anything.

Test the runner on two smoke jobs (`--heldout 1`) before loading the real queues.

---

## 3. The order

| # | Phase | Item | Resource | Cost | Depends on | Gate |
|---|---|---|---|---|---|---|
| 1 | 0 | **KC-C7** verifications (V1 to V7) | CPU | < 1 h | none | none |
| 2 | 0 | **KC-C1** nested selection audit | CPU | minutes | none | C1 letter (E escalates) |
| 3 | 1 | **KC-D0** deep code changes and inertness, including D0.5 for KC-D6 (`run_adv_align_loso.py`, `--coral-normalize`) | CPU/GPU smoke | 2 to 3 h | none | inertness (fail stops all deep stages) |
| 4 | 1 | Classical code changes for C2, C3, C5 and S2's adapter, each with its inertness proof | CPU | 1 to 2 h | none | inertness per script |
| 5 | 1 | `kc23_queue.py` built and smoke-tested | CPU | < 1 h | 3, 4 | none |
| 6 | 2 | **KC-D1 Tier A, seed 42 re-run** (R1 to R12) | GPU | about 8.3 h | 3, 5 | **D1 reproduction gate** (fail stops all deep stages) |
| 7 | 2 | **KC-S1** scripted buffer (3 realizations) | GPU plus CPU | about 3 h plus 2 h | 6 | S1 letter (D-S escalates as a claim change, not a halt) |
| 8 | 2 | **KC-D5** ENABL3S deep (5 realizations) | GPU | about 2 to 3 h | 6 | none |
| 9 | 3 | **KC-D1 Tier A, seeds 7, 123, 1001** | GPU | about 25 h | 6 | D1 headline gate after the last seed |
| 10 | 3 | **KC-D2** reliance analysis | CPU | < 1 h | 9 | D2 letter |
| 11 | 4 | **KC-D4** channel-axis invariance sweep | GPU | about 19 h | 9 | D4 letter |
| 11b | 4 | **KC-D6 Stage 1**: learned alignment axis, every family at seed 42 re-run | GPU | about 24 h | 3, 6 | D6 sanity gate (fail escalates), then the per-family manipulation gate |
| 11c | 4 | **KC-D6 Stage 2**: seeds 7 and 123, passing families only | GPU | up to about 48 h | 11b | D6 outcome letters and the combined through-line matrix |
| 12 | 4 | **KC-D3** axis against magnitude | GPU | about 11 h | 9 | D3 letters |
| 13 | 4 | **KC-D1 Tier B** (R13 to R17, 3 realizations) | GPU | about 14 h | 6 | none (analysed with Tier A at #9's stats rerun) |
| 14 | 5 | **KC-S3** active-only benchmark | GPU plus CPU | about 3 h plus 2 h | 6; KC-C3 result for optional cells | none |
| 15 | 5 | **KC-S2** ENABL3S transitions and window trade | GPU plus CPU | about 3 to 4 h | F0 gate | F0 (MARGINAL asks Enam) |
| C-a | 2 to 5 | **KC-C2** whitening, 250 and 400 ms | CPU, beside the GPU | about 8 h | 4 | C2 letters (W3 escalates) |
| C-b | 2 to 5 | **KC-C3** classical tuning parity | CPU, beside the GPU | about 30 to 40 h | 4 | C3 letters (P2, P3, N2, E2 escalate) |
| C-c | 2 to 5 | **KC-C4** richer features | CPU | about 6 to 10 h | C-b (uses SVM-X) | C4 letter (F-C escalates) |
| C-d | 2 to 5 | **KC-C5** leak decomposition | CPU plus about 1 h GPU slot | about 4 h | 4 | C5 letter (L3 escalates) |
| C-e | 2 to 5 | **KC-C6** ENABL3S ladder | CPU | about 1 h | C-a | C6 letter |
| 16 | 6 | **KC-F1** to **KC-F4** consolidation | CPU | about 3 h | everything above | F1 decision point for Enam |

**Why this order.** Items 1 and 2 are free, and they settle framing questions (the route of Table 4.11, the size of the selection optimism) that shape how later numbers are read. The D1 seed-42 re-run comes first on the GPU because it is the reproduction gate: nothing deep is interpretable if the code has drifted. S1 and D5 follow immediately. They are short, and they answer the two claim-level questions a viva is most likely to press (label-free deployment, and what replicates on ENABL3S). The remaining D1 seeds then complete the variance picture that every later deep contrast is read against. D4 and D6 come before D3 and Tier B because the through-line is the thesis's most prominent claim. D6 follows D4 because its verdict ends with the combined matrix across all three axes, which needs C2 and D4 in hand. D6 Stage 2 runs only for the families whose knob passed the manipulation gate, so a knob that fails the way Deep CORAL's did costs one seed, not three. The CPU stages run beside the GPU from Phase 2 onward.

**Dependencies of a halt.**

- A D1 reproduction-gate failure halts every deep stage (#7 to #15 including 11b and 11c, and C-d's CNN arm). A D6 sanity-gate failure halts 11c only. The CPU stages continue.
- A C2 W3 halts C-e only. C3's P3 halts nothing, but the text blocks must wait for Enam's decision.
- Every other ESCALATE is a claim change: the queue continues and the item waits in `KC23_HALT.md` for Enam.

---

## 4. Consolidation (KC-F1 to KC-F4)

### KC-F1. The statistical family. Owner decision point.

Write `recompute_unified_fdr_v10.py`, leaving v9 and its outputs untouched, producing two versions:

- **v10-add:** the 234 v9 tests plus every row of `kc23_new_tests.csv` that meets the thesis's admission rule (Section 3.4.5; Appendix D.2 exclusions applied the same way).
- **v10-replace:** as v10-add, except that each realization-averaged KC-D1 contrast replaces its single-run v9 counterpart. Give an explicit mapping table (v9 row → KC-D1 contrast).

Report survivors under both, and every comparison whose status differs between v9, v10-add and v10-replace. **Enam chooses** which version the thesis publishes. Recommend v10-replace if D1 lands cleanly, since it is the more accurate evidence, and give the reason in one paragraph.

### KC-F2. The numbers delta

Produce `06_Code/KC23_NUMBERS_DELTA.md`:

- every number in the thesis that any KC23 result changes, with old value, new value, source directory and the family-dependent p-value;
- every passage that states it, found by searching `pdftotext -layout` of `01_Thesis/MSc Project Final Draft Restructured..pdf`, with page and section. Search for each number in all its printed forms (0.858, 85.8, 85.8%).

This file drives text blocks TB1 to TB16. It must be complete: a number restated in the Discussion or Conclusion and missed here is the classic drift defect.

### KC-F3. Repository hygiene

- Re-run `build_run_manifest.py`. Every `results_kc23_*` directory must come back `verified` on its core fields.
- Add every KC23 command to `REPRODUCE.md` (exact commands, in run order), and index the new results in `README.md`.
- Draft release notes for v1.3.0 in `RELEASE_NOTES_v1.3.0_DRAFT.md`. **Do not tag, push or mint a DOI.** That is Enam's step.

### KC-F4. The programme report

Write `06_Code/KC23_PROGRAMME_REPORT.md`:

- a letter table for every stage;
- every ESCALATE with the decision needed;
- the measured run-variance band that replaces 0.5;
- the delta summary;
- a closing list of "claims now stronger", "claims narrowed" and "claims withdrawn", each with its source stage.

Then update `KC23_TRACKLIST.md` Part E ticks, and stop.

---

## 5. What to report back at the end

1. The KC23 letter table.
2. Every ESCALATE, with its numbers and the passages it touches.
3. The F1 family comparison and the recommendation.
4. Confirmation that no thesis file was edited, the model of record and v9 were untouched, and no push or tag happened.
