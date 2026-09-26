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
