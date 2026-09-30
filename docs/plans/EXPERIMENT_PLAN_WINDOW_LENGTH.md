# Experiment plan: window-length ablation (W-1)

**Status:** ready to run. Written 31 August 2026.
**Execution:** on Enam's machine. Everything here runs locally; nothing needs the network.
**Owner decision point:** one, defined in §1.3. Do not resolve it yourself. Report and stop.

---

## 0. Read this first

This plan is self-contained. You do not need the conversation that produced it.

Three things that will otherwise cost you a round trip each:

1. **The Python interpreter is `06_Code/.venv/Scripts/python.exe`.** Every existing `jobs_*.txt` in `06_Code/` points at `MSc Python Project/.venv/Scripts/python.exe`. That folder was removed in the August restructure. Those files are stale. Do not copy their interpreter path.
2. **You are writing almost no new modelling code.** `train_classical_loso.py` and `run_cnn_arch_loso.py` are already parameterized on the feature file and on `--norm-mode`, and both already support `--resume`. A different window length is a different `--features` / `--npz` argument and nothing else. Resist the urge to write a sweep driver that re-implements them.
3. **`--n-jobs 1` always, on every job, without exception.** See §5.1. This is not a performance preference. It is the fix for a deadlock.

---

## 1. The experiment

### 1.1 Why it exists

The thesis fixes a 250 ms analysis window and never varies it under LOSO. Two admitted weaknesses hang off that choice:

- **§3.3 confesses a spectral-resolution limit.** At a 250 ms window the discrete spectrum resolves to about 4 Hz, so MNF and MDF are noisy, and the text concedes that reliable spectral estimation would want one to two seconds. Every Freq-72 result in the thesis rests on features the thesis itself calls noisy. Nothing currently tests what that costs.
- **§5 argues the window is the binding constraint on control latency**, not the classifier. The thesis says this and then never quantifies the trade-off. A shorter window is the only real lever on responsiveness, and its price in accuracy is unmeasured.

So this is not a sweep looking for the best window. It is a robustness check on the thesis's central claim, and a latency-accuracy trade-off measurement that belongs beside the causal-buffer work in §3.13 and §4.14.

### 1.2 The primary endpoint is the gap, not the accuracy

This matters and it is easy to get wrong.

The thesis does not claim 250 ms is optimal. It claims **per-subject normalization beats global normalization** under cross-subject evaluation. So the quantity that has to survive the ablation is the **normalization gap**, per-subject macro-F1 minus global macro-F1, computed per subject.

- **Primary endpoint:** the normalization gap at each window, for the SVM and for the ResNet-SE+CD. If the gap holds at 150, 250 and 400 ms, the finding is robust and the ablation is a clean positive result regardless of which window scores highest.
- **Secondary endpoint:** absolute per-subject LOSO macro-F1 under per-subject normalization, 400 ms versus 250 ms. This is the one that can force a decision.

Report both. Do not collapse them.

### 1.3 Pre-registered decision rule

Fix this before running. Enam's instruction was to report if 400 ms wins by a margin of real statistical value, so "real statistical value" is defined here, in advance, and not negotiated after seeing the numbers.

On the **secondary** endpoint, 400 ms versus 250 ms, per-subject normalization, paired across the 40 subjects:

| Outcome | Condition | Action |
|---|---|---|
| **A. No material difference** | Holm-corrected p ≥ 0.05, **or** \|Δ\| < 1.0 pp with \|d\| < 0.3 | 250 ms stands. Write it up as a robustness result. No escalation. |
| **B. Material but tolerable** | corrected p < 0.05 **and** 1.0 ≤ \|Δ\| < 2.5 pp **and** \|d\| < 0.8 | 250 ms stands, reported with the trade-off stated honestly. Flag to Enam in the summary, but do not stop. |
| **C. Material and large** | corrected p < 0.05 **and** \|Δ\| ≥ 2.5 pp **and** \|d\| ≥ 0.8 | **Stop and escalate to Enam.** Do not begin rewriting anything. |

Δ is the mean paired difference in macro-F1, in percentage points. d is the paired Cohen's d. Holm correction runs across the family defined in §7.2.

On the **primary** endpoint, escalate if the normalization gap fails to reach significance at any window, since that would be a threat to the thesis's central claim rather than a question about window choice.

### 1.4 Windows, and why these three

| Window | Status | Why it is in |
|---|---|---|
| **150 ms** | features already extracted | Already has a subject-dependent result in the thesis. Tests the short end, where the latency argument lives. |
| **250 ms** | the incumbent, all results exist | The baseline. Reused, not re-run. |
| **400 ms** | must be built | Resolves the spectrum to about 2.5 Hz instead of 4 Hz, which is the direct test of the §3.3 limitation. |

100 ms was considered and cut. It is roughly half the total bill on its own, because the SVM cost scales as n² and the window count scales as 1/W, and it answers a question about the latency floor rather than about the finding.

---

## 2. What already exists and must be reused, not re-run

Do not regenerate any of these. Read them.

| Artifact | Path | Use |
|---|---|---|
| 250 ms windows | `06_Code/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz` | CNN input, 250 ms arm |
| 150 ms windows | `06_Code/windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR.npz` | CNN input, 150 ms arm |
| 250 ms Freq-72 | `features_out/freq_windows_..._w250_..._features_ext.npz` + `_meta.csv` | SVM, 250 ms arm |
| 150 ms Freq-72 | `features_out/freq_windows_..._w150_..._features_ext.npz` + `_meta.csv` | SVM, 150 ms arm |
| SVM 250 global | `results_loso_freq/` | baseline arm, reused |
| SVM 250 per-subject | `results_loso_freq_persubj/` | baseline arm, reused |
| ResNet-SE+CD 250 per-subject | `results_cnn_aug_resnet_se_chandrop/` | baseline arm, reused (Stage 0 fix, 31 Aug: original path held a SimpleEMGCNN run at 0.757, not the deep model of record at 0.8395) |
| ResNet-SE+CD 250 global | `results_win250_cnn_global/` | **generated fresh in Stage 3**, no published gate value (Stage 0 fix, 31 Aug: no stored ResNet-SE+CD global-norm 250 ms run existed; original path held a SimpleEMGCNN run at 0.681) |
| Alignment ladder 250 | `results_alignment_ladder_loso/` | baseline arm, reused |

**Verify each of these exists and parses before launching anything.** If a 250 ms arm is missing, say so and stop; do not silently re-run it, because a re-run costs hours and the published numbers are the gate (§6).

---

## 3. What must be built

### 3.1 The 400 ms windows and features

This is the only preprocessing work. All flags below were read off the scripts and are correct.

**Stage 0 first: reproduce the 250 ms features before building anything new.** Run the extraction command against the existing 250 ms windows npz, writing into a scratch directory, and confirm the resulting feature matrix matches `features_out/freq_windows_..._w250_..._features_ext.npz` exactly. If it does, the command is right and can be trusted for 400 ms. If it does not, the flags are wrong and every 400 ms number would be incomparable with the rest of the thesis. This costs minutes and removes all guesswork:

```
"<PY>" -u extract_features.py --npz windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz --meta <the preprocess meta for w250> --out-dir features_out_gatecheck --prefix freq --use raw --freq --no-wavelet --fs 2000.0
```

then

```python
import numpy as np
a = np.load("features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz")["X"]
b = np.load("features_out_gatecheck/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz")["X"]
assert a.shape == b.shape and np.allclose(a, b, rtol=1e-5, atol=1e-6), "extraction command does not reproduce the published features"
```

**Then build 400 ms.** Preprocess, holding every parameter at the values the existing artifacts used and changing only `--win-ms`:

```
"<PY>" -u preprocess_emg.py --subjects 1-40 --movements WAK,UPS,DNS,STDUP --win-ms 400 --overlap 0.5 --min-conf 0.60 --envelope-ms 50 --low 20 --high 450 --order 4 --out-npz windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR.npz --out-meta windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_meta.csv
```

Check the remaining `preprocess_emg.py` flags against `--help` before running, since a few (the data root, the status filter that produces the `AorR` tag, and the alignment tolerance) were not captured here and the defaults may not match what produced the existing files. **Compare your 400 ms meta against the 250 ms meta column by column; every column except the window-derived ones must have the same semantics.**

Then extract, with the identical feature flags Stage 0 validated:

```
"<PY>" -u extract_features.py --npz windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR.npz --meta windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_meta.csv --out-dir features_out --prefix freq --use raw --freq --no-wavelet --fs 2000.0
```

**`--fs 2000.0` is deliberate and must not be "corrected" to 1920. See §10.1.** Changing it would confound the window comparison with a sampling-rate change.

Expected outputs, matching the existing naming so the downstream drivers find them:

```
06_Code/windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR.npz
features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_ext.npz
features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_meta.csv
```

**Sanity checks on the 400 ms output before you use it:**

- Window count near **16,500**. The 1/W relationship holds tightly between the existing two: 150 ms gives 45,382 and 250 ms gives 26,347. Anything outside 15,000 to 18,000 means something other than the window length changed.
- The windows npz must carry both `X_raw` and `X_env`, each `(n, 9, 768)`. 768 is `round(0.400 × 1920)`; if `fs` estimates differently per trial the count may vary slightly, which is fine, but 9 channels is not negotiable.
- All 40 subjects present; four classes present; STDUP still the majority at roughly 55%.
- Feature matrix `(n, 72)`, float32, no NaN and no inf.
- `_features_cfg.json` must be identical to the 250 ms one, including `"sampling_rate": 2000.0`.

### 3.2 Job queues

Three files in `06_Code/`, written in the style of the existing `jobs_*.txt` but with the **correct** interpreter path. One full command per line. See §4 for contents.

### 3.3 The statistics and reporting script

`06_Code/window_ablation_stats.py`. See §7.

---

## 4. Run order and commands

Run the stages in order. Stage 2 and Stage 3 can overlap: Stage 2 is CPU-bound and Stage 3 is GPU-bound, so running both at once is close to free. Do not overlap Stage 1 with anything.

Interpreter, used throughout, quoted because the path contains spaces:

```
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
```

### Stage 1 — build the 400 ms artifacts

Serial, single process. Reads the 25 GB dataset, so expect it to be I/O-bound and do not run anything else against the disk at the same time. Runtime unmeasured; budget one to three hours and check in rather than assuming it has hung.

Finish with the §3.1 sanity checks. Do not proceed on a failed check.

### Stage 2 — classical SVM, CPU, via the memory guard

`jobs_window_svm.txt`, four jobs:

```
"<PY>" -u train_classical_loso.py --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_ext.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_meta.csv --out results_win150_global  --models SVM --norm-mode global      --inner-splits 5 --cv-scheme loso --n-jobs 1 --seed 42 --save-preds --flush-preds --resume
"<PY>" -u train_classical_loso.py --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_ext.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_meta.csv --out results_win150_persubj --models SVM --norm-mode per_subject --inner-splits 5 --cv-scheme loso --n-jobs 1 --seed 42 --save-preds --flush-preds --resume
"<PY>" -u train_classical_loso.py --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_ext.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_meta.csv --out results_win400_global  --models SVM --norm-mode global      --inner-splits 5 --cv-scheme loso --n-jobs 1 --seed 42 --save-preds --flush-preds --resume
"<PY>" -u train_classical_loso.py --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_ext.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_meta.csv --out results_win400_persubj --models SVM --norm-mode per_subject --inner-splits 5 --cv-scheme loso --n-jobs 1 --seed 42 --save-preds --flush-preds --resume
```

Launch:

```
"<PY>" -u run_multi_guard.py --jobs-file jobs_window_svm.txt --max-concurrent 4 --max-mem-percent 88 --min-free-gb 2.0 --kill-mem-percent 95 --kill-min-free-gb 0.7 --stagger 8
```

`--max-concurrent 4` rather than the default 10 deliberately. The 150 ms jobs hold a 45,382 × 72 matrix each and fit an RBF SVM over it, and the two 150 ms jobs are the memory-heaviest in the whole plan. If the guard reports emergency kills more than twice, drop to 2 and let it run longer.

### Stage 3 — ResNet-SE+CD, GPU, serial

The GPU serializes anyway, so no guard and no concurrency. `jobs_window_cnn.txt`, run one at a time, or just run them in sequence:

```
"<PY>" -u run_cnn_arch_loso.py --npz windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_meta.csv --arch resnet_se --augmentation chandrop --aug-chandrop-p 0.2 --norm-mode per_subject --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --out results_win150_cnn_persubj --resume
"<PY>" -u run_cnn_arch_loso.py --npz windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_meta.csv --arch resnet_se --augmentation chandrop --aug-chandrop-p 0.2 --norm-mode global      --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --out results_win150_cnn_global  --resume
"<PY>" -u run_cnn_arch_loso.py --npz windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_meta.csv --arch resnet_se --augmentation chandrop --aug-chandrop-p 0.2 --norm-mode per_subject --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --out results_win400_cnn_persubj --resume
"<PY>" -u run_cnn_arch_loso.py --npz windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_meta.csv --arch resnet_se --augmentation chandrop --aug-chandrop-p 0.2 --norm-mode global      --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --out results_win400_cnn_global  --resume
"<PY>" -u run_cnn_arch_loso.py --npz windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv --arch resnet_se --augmentation chandrop --aug-chandrop-p 0.2 --norm-mode global      --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --out results_win250_cnn_global  --resume
```

The fifth job is the Stage 0 fix (added 31 Aug): the ResNet-SE+CD global-norm 250 ms arm, which had no stored artifact. Its per-subject 250 ms partner is the reused `results_cnn_aug_resnet_se_chandrop/` (0.8395). Adds ~1.3 h to Stage 3.

**Check the meta path pairs with the npz.** The CNN driver takes the windows npz for X and the feature meta CSV for labels and subject ids, and they must be row-aligned. They are, for the existing 150 and 250 artifacts. Verify it holds for 400 by asserting `len(meta) == X.shape[0]` before the first real run.

**Smoke test first.** Run one fold of one configuration with `--heldout 1` and confirm it completes and writes a row. That costs two minutes and catches a wrong path before it costs you an evening.

### Stage 4 — alignment ladder at 400 ms (optional, recommended)

**`run_alignment_ladder_loso.py` cannot be pointed at a different feature file as it stands.** It takes only `--out`, `--rungs`, `--seed`, `--inner-splits`, `--resume`, `--verbose` and `--subjects`. Its data comes from `analyze_between_subject_variance.load_data()`, which reads two module-level constants hardcoded to the 250 ms paths:

```python
# analyze_between_subject_variance.py, lines 42-43
FEAT = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
META = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
```

Patch it in the least invasive way: make those two constants read an environment variable and fall back to the current literal, so every existing caller keeps working unchanged and the published 250 ms results stay reproducible.

```python
import os
FEAT = Path(os.environ.get("LADDER_FEAT", ROOT / "features_out" / "freq_windows_..._w250_..._features_ext.npz"))
META = Path(os.environ.get("LADDER_META", ROOT / "features_out" / "freq_windows_..._w250_..._features_meta.csv"))
```

**Then re-run the 250 ms ladder gate before trusting the 400 ms one.** With no environment variable set, rung 3 must still reproduce 0.7767 within 0.002 and rung 0 must still land near 0.708; the script already asserts this itself. That confirms the patch changed nothing. Then set the two variables to the 400 ms paths and run:

```
"<PY>" -u run_alignment_ladder_loso.py --out results_win400_ladder --rungs 3,0,1,2,4 --inner-splits 5 --seed 42 --resume
```

The gate values are hardcoded to the 250 ms figures, so the 400 ms run will trip them. Add a `--no-gate` flag, or pass the expected values in, rather than deleting the gate; it is load-bearing for every other caller.

This is the mechanism arm: it asks whether the over-alignment result, that per-subject standardization beats full whitening, is a property of the feature geometry or of the window. It is worth 5 hours. **The ladder at 150 ms is not in this plan** because it is SVM-based and therefore scales as n², putting it at roughly 38 hours alone. If the 400 ms ladder shows the rung ordering changing, that is the moment to reconsider paying for it.

### Stage 5 — statistics and report

```
"<PY>" -u window_ablation_stats.py --out results_window_ablation
```

---

## 5. Traps, and the remedies already found for them

Every item here cost time on a previous experiment in this thesis. They are not hypothetical.

### 5.1 Never use in-script joblib or loky parallelism

Repeated `GridSearchCV` under detached or nested processes deadlocks on Windows. The fix, already implemented across this project, is to parallelise at the OS-process level with `run_multi_guard.py` and give every child `--n-jobs 1`. `train_classical_loso.py` defaults to `--n-jobs 1` for exactly this reason; do not raise it. The one exception is `--rf-n-jobs`, which is safe because it parallelises inside a single fitted forest rather than across grid points, and it is irrelevant here since the core plan has no RF.

### 5.2 Never kill a job on soft memory pressure

An SVM fold is minutes of committed work. `run_multi_guard.py` distinguishes a **soft gate**, which only stops launching new children, from a **hard gate** at 95% used or 0.7 GB free, which is the only condition that kills. Keep that distinction. Raising the soft gate to make jobs launch faster is how you turn a slow run into a thrashing one.

### 5.3 Kill the newest, not the oldest

When the guard must kill, it kills the **newest** running child, because that is the one with the least invested work, and requeues it. This is already implemented. Do not "improve" it to kill the largest.

### 5.4 `--resume` and `--flush-preds` on every single job

Both drivers checkpoint per held-out subject and skip completed subjects on restart. `--flush-preds` writes each subject's predictions as the fold finishes rather than at the end. Together these make a kill cost one fold instead of a whole run. There is no reason to omit them, ever, and the guard's requeue behaviour depends on them.

### 5.5 Stagger the launches

The guard defaults to 4 seconds; this plan uses 8. Simultaneous cold loads of the same feature matrix by four children is a RAM spike that trips the hard gate for no reason.

### 5.6 Reproduce a known number before trusting a new one

`run_alignment_ladder_loso.py` already does this: it gates on rung 3 reproducing the published per-subject figure of 0.7767 within 0.002, and rung 0 landing near the published global baseline of 0.708. Apply the same discipline here, per §6.

### 5.7 A partially-failed batch script needs the whole batch re-verified

If a multi-edit script aborts without saving, the edits that had already succeeded are discarded too. Verify the whole batch, not just the step that failed.

### 5.8 Word holds a share lock on an open `.docx`

Writes to any chapter file or to `MSc Thesis.docx` fail silently or hard while that file is open in Word. Close it before the reporting phase. On one occasion the machine needed a full reboot rather than an app restart.

### 5.9 Every chapter must start on a new page

If this experiment leads to any edit of `MSc Thesis.docx`, the eight division headings (Chapters 1 to 6, References, Appendix A) carry `w:pageBreakBefore`, and that is what enforces it. Do not add padding blank paragraphs at the end of a chapter; with a real page break in front of them they produce a blank page. Re-check the render for blank pages after any content edit.

### 5.10 Disk

The 400 ms windows npz will be roughly 270 MB, on top of the two 430 MB files already present. Check free space before Stage 1. The dataset itself is 25 GB and lives in OneDrive; if OneDrive decides to sync mid-run it will compete for I/O.

---

## 6. Reproduction gates

Before any new number is reported, confirm the reused 250 ms arms still read back at their published values. These are read from existing result files, not re-run.

| Arm | Source | Expected LOSO macro-F1 | Tolerance |
|---|---|---|---|
| SVM, global, 250 ms | `results_loso_freq/` | 0.708 | ±0.002 |
| SVM, per-subject, 250 ms | `results_loso_freq_persubj/` | 0.777 | ±0.002 |
| ResNet-SE+CD, per-subject, 250 ms | `results_cnn_aug_resnet_se_chandrop/` | 0.840 | ±0.002 |
| ResNet-SE+CD, global, 250 ms | `results_win250_cnn_global/` | no published value — generated in Stage 3 | n/a |

**If a gate fails, stop and report it.** A failed gate means either the wrong directory is being read or something in the pipeline has drifted, and in both cases every new number is untrustworthy. Do not proceed and do not adjust the expected value to match what you found.

Second gate, on the new runs: **all four classical arms and all four CNN arms must have exactly 40 rows, one per subject, with no duplicates.** A resumed run that double-appends a subject is a real failure mode of append-mode checkpointing. Check it explicitly rather than trusting the row count.

---

## 7. Statistics

Follow the thesis convention exactly, which is set in §3.6 and used throughout Chapter 4.

### 7.1 Per comparison

- **Paired Wilcoxon signed-rank** across the 40 subjects. The pairing is by subject; each subject contributes one macro-F1 per arm.
- **Paired Cohen's d**, mean difference over the standard deviation of the differences.
- **BCa bootstrap 95% CI**, 10,000 resamples, on the mean paired difference. Seed 42.
- Report macro-F1 to three decimals in tables, and as percentages to one decimal in prose. Effect sizes, p-values and correlations stay decimal everywhere. This exception is a thesis convention and a blanket regex will corrupt it.

### 7.2 The correction family

Holm correction is applied within, not across, these two families:

- **Family 1, primary:** the normalization gap at each of the three windows, for each of the two models. Six tests.
- **Family 2, secondary:** window comparisons under per-subject normalization, 150 vs 250, 400 vs 250, 150 vs 400, for each of the two models. Six tests.

Do not pool the two families; they answer different questions and pooling them dilutes the primary result.

### 7.3 Output

`results_window_ablation/` containing:

- `window_ablation_summary.csv` with one row per arm: window, model, norm mode, n, mean F1, SD, and the 95% CI.
- `window_ablation_tests.csv` with one row per test: family, comparison, Δ in pp, raw p, Holm-corrected p, d, CI low, CI high.
- `window_ablation_verdict.md`, a short plain-language statement that names which of outcomes A, B or C in §1.3 was reached, and gives the numbers that decided it.
- `fig_window_ablation.png`, black and white, square-cornered, plain black arrows and no colour, matching the treatment applied to Figures 3.1 and 5.2. Generators for those live in `06_Code/figures_rework/` and are the style reference. Point sizes must be scaled by `FIG_W / PAGE_W` so the figure is legible at page width; see `make_fig5_2.py`.

---

## 8. Cost

Anchored on `fit_time_sec`, which the classical drivers already record, and on scaling exponents measured on the project's own Freq-72 matrix: **SVM fit time scales as n^2.03 and Random Forest as n^1.04.** Window count goes as 1/W, so 150 ms carries 1.67× the windows of 250 ms and 400 ms carries 0.63×.

| Stage | Hours | Basis |
|---|---|---|
| 1. Build 400 ms artifacts | 1 to 3 | unmeasured, I/O-bound |
| 2. SVM, 150 and 400 ms, both norms | 3.4 | measured |
| 3. ResNet-SE+CD, 150 and 400 ms, both norms | 5.2 | **estimated**, no CNN run in this project ever recorded timing |
| 4. Alignment ladder, 400 ms | 5.2 | measured |
| 5. Statistics and reporting | under 1 | |
| **Total** | **roughly 16 to 18 hours** | |

Stages 2 and 3 overlap, so the wall clock is closer to 12 hours than 18.

The CNN figure is the soft one. If Stage 3's first configuration takes materially longer than 1.3 hours for its 40 folds, say so rather than letting it run silently past the estimate.

**Optional additions, priced:** Random Forest at 400 ms is 7.3 hours and at 150 ms is 20.3 hours. The alignment ladder at 150 ms is 37.7 hours. None are in the core plan. Do not add them without asking.

---

## 9. What to report back

A single summary containing, in this order:

1. Which of outcomes A, B or C in §1.3 was reached, stated in the first line.
2. The gate results from §6, pass or fail, with the numbers.
3. The primary endpoint: the normalization gap at each window, per model, with corrected p and d. State plainly whether the thesis's central claim survives at all three windows.
4. The secondary endpoint: the window comparison table.
5. Anything that did not run, and why.
6. Actual wall-clock times against the §8 estimates, so the next estimate is better.

Do not begin drafting thesis text. Do not edit `MSc Thesis.docx` or any chapter file. The write-up is a separate decision that depends on which outcome came back.

---

## 10. Open items that need Enam, not you

### 10.1 The sampling-rate discrepancy, now characterized

This was investigated on 31 August and is understood. It is a text error, not a results error. Do not act on it; report it and let Enam decide the wording.

**What is true.** `preprocess_emg.py` estimates the sampling rate from each trial's Time column and gets **1920.0001 Hz**, which it uses correctly for the bandpass design, the envelope window and the window sample count, and writes into the meta. A 250 ms window is therefore 480 samples, which the windows npz confirms.

`extract_features.py` does **not** read that rate. It takes `--fs`, defaulting to 2000.0, and the saved `_features_cfg.json` records `"sampling_rate": 2000.0`. So the Freq-72 features were computed on a frequency axis scaled by 2000 / 1920 = **1.041667**.

**What it affects.** Only MNF and MDF, and both by exactly that constant. `feat_spectral_power` sums the whole spectrum and never uses `fs`, so it is untouched. The time-domain features never see it.

**What it does not affect: any reported result.** MNF and MDF are scaled by a single global constant, and every evaluation condition standardizes the features, either globally or per subject. Standardization removes a constant scale factor exactly, since if x becomes cx then the mean becomes c times the mean and the standard deviation c times the standard deviation, leaving z unchanged. The Random Forest is scale-invariant regardless. The one condition where it could bite is the `none` normalization row of the ablation, where a 4.2% rescaling of 18 of the 72 features slightly changes their relative weight, and that is a minor ablation row rather than a headline. The thesis quotes no absolute MNF or MDF value in Hz; the ICCs it does quote, 0.456 and 0.415, are correlations and are scale-invariant.

**What is actually wrong.** One clause in §3.3: "At a 250 ms window sampled at 2000 Hz (500 samples)". The true figures are 1920 Hz and 480 samples. The conclusion drawn in that same sentence, a spectral resolution of about 4 Hz, is correct either way, because 1920/480 and 2000/500 both give 4 Hz. So the argument survives intact and only the two numbers in the parenthetical are wrong.

**For this experiment: keep `--fs 2000.0`.** Consistency with the existing 150 ms and 250 ms features is what makes the window comparison valid. Correcting the rate for the 400 ms arm alone would confound window length with sampling rate and invalidate the whole ablation.

### 10.2 The decision if outcome C comes back

Not yours to make. The realistic options are to re-baseline the thesis at 400 ms, which is not affordable this close to submission, or to reframe 250 ms as a latency-constrained choice and report 400 ms as the accuracy-optimal one. Enam decides.
