# Check: is the CNN pipeline reproducible? (R-1)

**Status:** ready to run. Written 1 September 2026, after W-1.
**Cost:** Steps 0 and 1 are free. Step 2 is 2.6 h of GPU and runs only if Steps 0 and 1 say it should.
**Blocks:** W-2 Stage G2. Do not run the dropout-rate sweep until this closes.

---

## 1. What went wrong

W-1 generated a fresh ResNet-SE+CD run at 250 ms under train-fold global normalization and got **0.7723**. The thesis publishes **0.787** for what appears to be the same quantity, in the §4.13 CNN-adaptation figure, sourced from `f1_pre_adabn_mean = 0.7874` in `results_adabn_chandrop/adabn_summary.csv`.

They ought to agree. `run_adabn_cnn_loso.py` imports `train_fold` directly from `run_cnn_arch_loso.py` and applies the same `compute_train_norm(X[tr])` / `apply_norm` train-fold z-score, with the same epochs, patience, val-frac and seed. **1.5 pp apart is four times the 0.38 pp seed SD recorded in `results_seed_stability/`.**

Two things hang on the answer:

- **The thesis carries two different numbers for one quantity.** Whichever way this resolves, one sentence has to change or a footnote has to appear.
- **W-2's revised decision rule depends on the noise floor.** §4.3 of the channel-dropout plan now requires a margin of 3.0 pp *or twice the measured run-to-run SD*. That SD is what Step 2 measures. Without it the dropout sweep cannot be read.

---

## 2. Step 0 — config archaeology (free, minutes)

**Before assuming nondeterminism, establish that the two runs really were configured identically.** The cheapest explanation is that they were not.

Find the exact flags the AdaBN run used. `results_adabn_chandrop` is a *directory name*, not proof that `--augmentation chandrop --aug-chandrop-p 0.2` was passed. Look in:

- `logs/` and `_run_logs/` for the invocation
- `REPRODUCE.md`, which documents the command chain
- `git log` on the jobs files and on `run_adabn_cnn_loso.py`
- any argv echo in the run's own stdout capture

Specifically confirm, for both runs: `--arch`, `--augmentation` and `--aug-chandrop-p`, `--norm-mode`, `--epochs`, `--patience`, `--batch`, `--lr`, `--val-frac`, `--seed`, and the input npz.

**If the AdaBN run used `--augmentation none`, the question dissolves immediately.** 0.787 would then be ResNet-SE *without* channel dropout under global normalization, and 0.7723 would be ResNet-SE+CD under global normalization. Those are different quantities, both correct, and the fix is a label change in the thesis rather than anything numerical. Report that and stop.

---

## 3. Step 1 — pair the two runs per subject (free, minutes)

```
"<PY>" -u check_cnn_reproducibility.py
```

Defaults are already set to `results_adabn_chandrop` (`f1_pre_adabn`) against `results_win250_cnn_global` (`f1_macro`). It pairs the 40 subjects and prints the mean difference, its spread, the sign split, the correlation, a paired Wilcoxon, and the six largest disagreements. It writes `cnn_reproducibility_pairs.csv`.

**How to read it.** Two signatures discriminate, and the script scores only those:

| | Nondeterminism | Config difference |
|---|---|---|
| Sign split | both signs well represented | most subjects shift the same way |
| Spread of differences | wide, several pp | narrow |

Correlation is **not** diagnostic. It is high under both explanations, because the same 40 subjects drive it either way. The script treats a *low* correlation as a warning that the pairing itself is wrong, not as evidence for either hypothesis.

If the script says CONFIG DIFFERENCE, go back to Step 0 and find it. Step 2 would only measure noise around the wrong baseline.

---

## 4. Step 2 — measure the noise floor (2.6 h GPU)

Only if Steps 0 and 1 point at nondeterminism.

Re-run the W-1 250 ms global arm **twice more, changing nothing at all**, into fresh directories. Same seed, same flags, same input:

```
"<PY>" -u run_cnn_arch_loso.py --npz windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv --arch resnet_se --augmentation chandrop --aug-chandrop-p 0.2 --norm-mode global --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --out results_repro_250global_r2 --resume
```

and the same again into `results_repro_250global_r3`.

**Use exactly the flags that produced `results_win250_cnn_global`, whatever Step 0 found those to be.** If they differ from the line above, use the real ones and say so; the point is three runs of one identical command, not three runs of a command I guessed at.

With `results_win250_cnn_global` that gives three runs at identical settings. Report the mean of each, the SD across the three, and the full range. Then run `check_cnn_reproducibility.py` between each pair of repeats to see whether the within-repeat differences look like the AdaBN-versus-W-1 difference.

---

## 5. How to read the outcome

| Finding | What it means | What follows |
|---|---|---|
| **Step 0 finds a real config difference** | The two numbers measure different things | Label the thesis figure correctly. No further runs. W-2's noise floor still unmeasured, so Step 2 runs anyway before G2. |
| **Run-to-run SD ≥ 1.0 pp and the 1.5 pp gap sits inside the observed range** | cuDNN nondeterminism, as suspected | Pick one number as canonical, footnote the reproducibility bound, and set W-2's threshold from the measured SD |
| **Run-to-run SD < 0.5 pp and none of the repeats approach 0.787** | Neither explanation holds | Stop and report. Something is different that Step 0 missed, and it needs finding before anything else runs |

**Whatever the outcome, record the SD.** It is the number W-2 §4.3 now depends on, and it is worth a sentence in the thesis regardless: a deep pipeline whose run-to-run spread is a third of its headline effect is a fact a reader is entitled to.

---

## 6. The fix worth making either way

`run_cnn_arch_loso.py` and `train_cnn_loso.py` seed NumPy, torch and CUDA but never set:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Do not add these as part of this check.** Changing them now would make new runs incomparable with every published number in the thesis. Note it as a recommendation for the reproducibility section and for any future work, and leave the code alone.

---

## 7. What to report

1. The Step 0 finding: were the two runs configured identically, yes or no, with the evidence.
2. The Step 1 table and which signature the pattern matched.
3. If Step 2 ran: three means, the SD, the range, and whether 1.5 pp sits inside it.
4. The number to use for W-2 §4.3's threshold.
5. A one-line recommendation on which of 0.787 or 0.772 should be canonical in the thesis, and why.

**Do not edit `MSc Thesis.docx` or any chapter file.** The wording decision is Enam's once the numbers are in.
