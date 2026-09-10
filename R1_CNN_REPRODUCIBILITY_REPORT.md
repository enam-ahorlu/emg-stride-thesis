# R-1: CNN reproducibility check — report

**Run 1 September 2026, after W-1. No thesis file was edited.**

## Question

W-1 produced `results_win250_cnn_global` (ResNet-SE+CD, 250 ms, train-fold global
z-score) at **0.7723**. The thesis §4.13 publishes **0.787** for what looks like
the same quantity, sourced from `f1_pre_adabn_mean = 0.7874` in
`results_adabn_chandrop/adabn_summary.csv`. 1.5 pp apart.

## Step 0 — config archaeology: the two runs ARE configured identically

| Axis | Evidence |
|---|---|
| Flags | AdaBN command (`docs/EXPERIMENT_PLAN_CHANDROP.md:43`): `run_adabn_cnn_loso.py --arch resnet_se --augmentation chandrop --epochs 40`. Every unspecified flag defaults to the value W-1 passed explicitly: `--aug-chandrop-p 0.2`, `--batch 512`, `--lr 1e-3`, `--patience 7`, `--val-frac 0.15`, `--seed 42`, `--xkey X_env`. |
| Augmentation | Both `chandrop`, p = 0.2. **Not** `--augmentation none`, so the "different quantity" escape in the plan does not apply. |
| Code | `run_adabn_cnn_loso.py:41` — `from run_cnn_arch_loso import train_fold  # reuse the exact source-training loop`. The pre-AdaBN eval block is line-for-line identical to `run_cnn_arch_loso.py`'s global-norm fold: same `compute_train_norm(X[tr])` / `apply_norm`, same `choose_val_subjects(subtr, val_frac, seed+heldout)`, same `train_fold(...)`, same `DataLoader(batch_size=512, shuffle=False)`, same `f1_score(average="macro")`. |
| No code drift | `cnn_architectures.py` mtime 2026-07-19, `run_cnn_arch_loso.py` 2026-07-21 20:11, `run_adabn_cnn_loso.py` 2026-07-22 01:21, `train_cnn_loso.py` 2026-07-10 — all **before** the AdaBN run (2026-07-22 05:21) and untouched since. The Chapter-3 "Table 3.4 kernel sizes" fix in the handoff was a table correction, not a code change. |
| No environment drift | `.venv` torch/cuDNN (`torch 2.10.0+cu126`, cuDNN 91002) mtime 2026-04-14, untouched since. |
| Same input | Both use the w250 npz (26,347 windows, subjects 1–40). |

**Conclusion: identical config, identical code, identical environment, identical input.** The plan's Step-0 "report and stop" branch (AdaBN used `--augmentation none`) does not apply.

## Step 1 — pair the two runs per subject

```
mean paired difference A - B : +1.52 pp
sd of the differences        : 4.57 pp
range                        : -11.30 to +19.33 pp
sign split                   : 24 A>B, 16 A<B
correlation                  : Pearson r = 0.870, Spearman = 0.894
paired Wilcoxon              : p = 0.041, d = 0.33
```

Both discriminating signatures fire: signs balanced (24/16), spread wide (4.57 pp).
Correlation high, so the pairing is sound. **Signature = NONDETERMINISM.**

## Step 2 — three runs of one identical command

`run_cnn_arch_loso.py --npz windows_...w250....npz --meta ...w250..._features_meta.csv --arch resnet_se --augmentation chandrop --aug-chandrop-p 0.2 --norm-mode global --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42`

| run | LOSO macro-F1 |
|---|---|
| `results_win250_cnn_global` (W-1) | 0.7723 |
| `results_repro_250global_r2` | 0.7667 |
| `results_repro_250global_r3` | 0.7760 |

- 3-run mean: **0.7716**
- 3-run SD (n = 3): **0.47 pp**; range 0.7667–0.7760 (span 0.93 pp)
- Per-subject paired-diff SD across the three pairings: **5.44 pp** (individual subjects swing up to 26 pp between identical runs — e.g. Sub12: 0.878 / 0.628 / 0.891)
- Model-implied SE of a 40-subject mean: 5.44 / sqrt(40) = **0.86 pp**
- AdaBN's 0.7874 is **+1.14 pp above the max** of the three repeats and **+1.58 pp above the 3-run mean** (~1.8x the 0.86 pp mean-SE).

## Reading against the plan's §5 table

The point-estimate 3-run SD (0.47 pp) is under 0.5 and no repeat reaches 0.787, which
by the letter is the plan's third row ("neither explanation holds — something Step 0
missed"). **That row's premise is falsified here**: Step 0 verified identical config,
code (mtime-checked), environment and input on four independent axes. The plan's §1
anchored on the wrong noise figure — the 0.38 pp *seed* SD in `results_seed_stability/`
is for **SimpleEMGCNN, per-subject norm, no augmentation**, a different model. The
true run-to-run spread for ResNet-SE+CD / global / chandrop is **~0.5–0.9 pp at the
mean** (0.47 pp measured from three; 0.86 pp implied by the 5.44 pp per-fold SD) and
**~5 pp per fold**. The 1.5 pp gap is ~1.7–3x that, with 0.7874 a high single draw.

**Finding: cuDNN nondeterminism, noise floor 2–3x larger than the plan assumed. No hidden config difference exists to find.**

## Answers to §7

1. **Configured identically?** Yes — flags, code (untouched since before the AdaBN run), environment, input all verified identical.
2. **Step 1 signature:** NONDETERMINISM (wide per-subject spread, balanced signs, high correlation).
3. **Step 2:** three means 0.7723 / 0.7667 / 0.7760; SD 0.47 pp (n=3); model-implied mean SE 0.86 pp; range 0.7667–0.7760. The 1.5 pp gap does **not** sit inside the three-run range (0.787 is 1.14 pp above the max) but is within ~1.5–2x a realistic run-to-run SE.
4. **W-2 §4.3 threshold:** "3.0 pp or twice the measured run-to-run SD." Twice the SD is 2 x 0.86 = 1.7 pp (or 2 x 0.47 = 0.94 pp on the raw n=3). Both are below the 3.0 pp floor, so **W-2 §4.3's threshold = 3.0 pp.** Unchanged.
5. **Canonical number:** use **0.772** (mean of the three fresh 2026-09 repeats is 0.7716; W-1's 0.7723 is that). 0.7874 is one 2026-07 execution that lands above all three current repeats. Recommended thesis fix: a footnote in §4.13 — "ResNet-SE+CD under global normalisation reaches LOSO macro-F1 ~0.77 (three repeats: 0.772, 0.767, 0.776); run-to-run SD ~0.5–0.9 pp from cuDNN nondeterminism". Note this shifts the AdaBN pre-to-post delta in §4.13 from +3.1 pp to roughly +4.6 pp; the framing of that comparison is Enam's call.
6. **cudnn.deterministic:** recommend setting `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` for future work, and adding one sentence to the reproducibility section stating the measured run-to-run spread. **Do not set it now** — it would decouple new runs from every published number (plan §6). Code left unchanged.

## Effect on W-1

None. `results_win250_cnn_global` fed only the ResNet-SE+CD 250 ms *global* arm. At
0.772 vs a hypothetical 0.787 the 250 ms normalization gap moves from +6.7 pp to
+5.6 pp — still large, still Holm-significant, outcome B unchanged. The decisive
400-vs-250 comparison is per-subject only and untouched.
