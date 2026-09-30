# Repository layout

On 30 September 2026 the repository moved from a flat root (about 620 entries) into the folders below.
Every script, job file and shell runner was updated to the new paths in the same commit, and the KC23
test suite passes against them. The archived releases (v1.0.0, v1.1.0 and v1.2.0, the version of record
cited in the thesis) keep the old flat layout, so the table at the end maps one onto the other.

| Folder | Holds |
|---|---|
| `src/` | every pipeline script: preprocessing, feature extraction, trainers, experiment drivers, statistics, figure builders, and the KC23 queue. Kept flat on purpose, because many scripts import one another as siblings. |
| `data/raw/` | the SIAT-LLMD and ENABL3S recordings. Not tracked; see `.gitignore`. |
| `data/windows/` | windowed arrays (`.npz`, not tracked) and their tracked metadata, config and summary files |
| `data/features_out*/` | derived feature matrices (SIAT, ENABL3S, the filter experiment, the gate check) |
| `results/` | one folder per experimental arm, plus the run manifest, `_bestparams.json` and the small pair and profile tables the statistics scripts write |
| `results/smoke/` | one-fold smoke runs that preceded the full chains |
| `figures/report_figs/` | thesis figures and their CSV summary tables |
| `figures/rework/` | the rebuilt Chapter 3 to 5 figures and their generators |
| `jobs/` | KC23 job CSVs, the published-run table, and the older `jobs_*.txt` lists |
| `scripts/` | shell runners for the experiment chains, the KC23 Task Scheduler script and the power-setting restore |
| `tests/kc23/` | the KC23 test suite (`python -m pytest tests/kc23` from the repository root) |
| `docs/plans/` | pre-registered experiment plans and run orders, unchanged since they were written |
| `docs/reports/` | outcome and reproducibility reports |
| `docs/kc23/` | KC23 status, halt record, conformance and verification notes |
| `docs/notes/` | summaries and session notes |
| `logs/` | run logs and the KC23 queue state (`logs/kc23/`) |
| `archive/` | superseded scripts (FDR families v2 to v7, scratch checks) and the old `_STRUCTURE.md` |

## Running things

Run every script from the repository root, for example `python src/train_classical_loso.py ...`.
Scripts locate the root themselves (`Path(__file__).resolve().parents[1]`), so paths inside them are
written relative to the root: `results/<arm>`, `data/features_out/...`, `data/windows/...`.

The KC23 queue runs as `.venv\Scripts\pythonw.exe -u src/kc23_queue.py --redirect-output` from the
root, which is what the `KC23Queue` scheduled task and `scripts/kc23_queue_task.ps1` do.

## Old path to new path

| Before 30 September 2026 | Now |
|---|---|
| `<script>.py` | `src/<script>.py` |
| `results_<name>/` | `results/<name>/` |
| `features_out*/` | `data/features_out*/` |
| `windows_*` | `data/windows/windows_*` |
| `SIAT_LLMD20230404/`, `5362627/` | `data/raw/SIAT_LLMD20230404/`, `data/raw/5362627/` |
| `report_figs/` | `figures/report_figs/` |
| `figures_rework/` | `figures/rework/` |
| `plots/` | `figures/plots/` |
| `reports/` | `results/reports/` |
| `_run_logs/` | `logs/` |
| `_w3_smoke/` and the other smoke folders | `results/smoke/w3_smoke/` and so on |
| `tests_kc23/` | `tests/kc23/` |
| `kc23_jobs_*.csv`, `jobs_*.txt`, `kc23_d1_published_runs.csv` | `jobs/` |
| `*.sh`, `kc23_queue_task.ps1`, `kc23_restore_power.cmd` | `scripts/` |
| `EXPERIMENT_PLAN_*.md`, `RUN_ORDER_*.md`, `RUN_QUEUE.md`, `docs/EXPERIMENT_PLAN_*.md` | `docs/plans/` |
| `*_REPORT.md`, `CHECK_CNN_REPRODUCIBILITY.md` | `docs/reports/` |
| `KC23_*.md` | `docs/kc23/` |
| other `docs/*.md` | `docs/notes/` |
| `_bestparams.json`, `RUN_MANIFEST.csv`, `analysis_results.json` and the root-level pair and profile CSVs | `results/` |
| `.venv_stats_requirements.txt` | `requirements-stats.txt` |
| `recompute_unified_fdr_v2.py` to `v7.py`, `scratch_test_gc.py`, `e3_check.py` | `archive/scripts/` |

Plans, reports and notes written before the move still name files by their old paths. They are left
as written because several of them are pre-registration records; read their paths through the table above.
