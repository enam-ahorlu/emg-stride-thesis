# 06_Code — why this folder is still flat

Renamed from `MSc Python Project` on 24 August 2026. The three docstrings and eight `.md` files that named the old folder were updated in the same pass, so nothing here reads stale.

## What moved
- The eight planning/summary notes → `docs/`. No script opens a `.md`; the ones that mention a plan file do so in a comment only.
- `logs_e3_cnn.log`, `logs_e3_svm.log`, `logs_e4.log`, `logs_e5_calibrate.log` → `logs/`. No script reads them by name.
- `__pycache__` → `../_to_delete/06_Code__pycache__`. Regenerable; safe to delete.
- `Codebase_Documentation.html` moved in from the project root — it documents this code, so it belongs beside it.

## What deliberately did NOT move, and why

**The 99 `.py` files stay in one flat directory.** Twelve of them import `train_classical_loso` as a sibling module, and Python puts the *script's own* directory on `sys.path`, not the working directory. Filing them into `src/analysis/`, `src/figures/` and so on breaks every one of those imports the moment a script is run from anywhere but its own folder.

**The ~90 `results_*` directories stay at this level.** They are named by hardcoded relative path in the scripts — **79 distinct ones**, e.g. `"results_ensemble_v2_chandrop"`, `"results_variance_decomposition"`. Moving them under a `results/` parent silently breaks every read and write in the pipeline, and the failure mode is a script that runs and quietly writes to the wrong place.

The same applies to `features_out/` (31 scripts), `report_figs/` (41), `plots/` (40), `reports/` (5), `logs/` (3), the `windows_*.npz` window caches (35), and `jobs_*.txt` (2).

So the flat layout here is not untidiness — it is the interface. `REPRODUCE.md` depends on it. Anyone tempted to tidy it further should first read that file and count the paths they would have to rewrite.

## The one thing that IS worth acting on: 49 GB of raw dataset

| Folder | Size | What it is |
|---|---|---|
| `SIAT_LLMD20230404/` | **25 GB** | the extracted primary dataset |
| `5362627/` | **24 GB** | the extracted ENABL3S external-validation dataset |
| `../05_Sources/SIAT_LLMD20230404.rar` | **8.3 GB** | the *source archive* of the 25 GB folder above — a redundant second copy |

That is roughly **57 GB inside a OneDrive-synced folder**, and it is almost certainly why file operations here are slow.

Both datasets are public and re-downloadable, and the pipeline no longer needs them: feature extraction is done, and everything downstream reads `features_out/` (53 MB) and the `windows_*.npz` caches. Only five scripts touch the raw folders at all, and all five are the already-completed extraction step.

Recommended, in order of return:
1. **Delete `../05_Sources/SIAT_LLMD20230404.rar`** — 8.3 GB back, and it is a pure duplicate of the extracted folder. Nothing reads it.
2. **Move `SIAT_LLMD20230404/` and `5362627/` out of OneDrive** to a local drive or external disk. 49 GB back, and the five extraction scripts still work if you point them at the new location with their `--root` argument.

Neither was done for you: 49 GB is a decision with sync and backup consequences, and it should be yours.
