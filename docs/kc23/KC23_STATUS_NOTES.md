Conformance commits (26 September 2026, authored by Enam, no trailers):

- `d613a06` pre-registration conformance: gates aligned to the 23 September plan text before their data exists
- `0696a4b` KC23 aggregators, instrumented D6 runners and queue wiring for the conformant gates

Table: KC23_PREREG_CONFORMANCE.md. Open plan items (no data yet): C3 edge-rule rerun generator, D1 MixedLM secondary model
(statsmodels not installed), D6 ADV-C / CDAN / mechanism contrasts and the divergence retry, S3 SVM-X and HGB cells (only if C3
lands P2 or P3).

Statistics environment (26 September 2026, ruling C): `06_Code/.venv_stats`, used only by the D1 MixedLM secondary model
(`kc23_d1_mixedlm.py`); `06_Code/.venv` is frozen for reproduction and has no statsmodels. Python 3.14.0, statsmodels 0.15.0,
numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, patsy 1.0.3, formulaic 1.2.2 (full list in `.venv_stats_requirements.txt`; the directory is
git-ignored).

PAUSED 26 September 2026 (Enam's request, laptop hibernation). Everything was stopped: the runner (pythonw), the running jobs
d1_r10_s1001 and s2b_predictions_400 (their entries are still in _run_logs/kc23/running.json, so a new runner reports them as no
longer running and re-queues them; both resume from their saved folds or subjects, losing only the fold or subject in progress), and
both S2 inertness runs (results_kc23_s2_inertness_check/: the new-script run has subject 156 on disk and resumes at 185; the
pre-change control restarts). The KC23Queue scheduled task is DISABLED (state Disabled) so nothing starts by itself on wake.
No failed jobs, no halted stages (queue_state.json). To resume: Enable-ScheduledTask -TaskName KC23Queue, then Start-ScheduledTask
-TaskName KC23Queue (the runner takes the stale queue.lock); restart the two S2 inertness runs by hand if wanted.
