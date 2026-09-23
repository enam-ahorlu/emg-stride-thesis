# KC23 status

Updated: 2026-09-23T13:12:56.789248

## Pre-registration record

Commit `0ba3575` ("KC23 pre-registration: every stage gate/stats script and
runner, written and tested before any KC23 result exists", 24 September
2026): every stage gate/stats script (C2, C3, C4, C5, C6, D1, D2, D3, D4,
D5, D6, S1, S2, S3) and the two new runners (run_scripted_supervised.py,
kc23_c4_extract_rich.py) were written and their 101 synthetic tests
(tests_kc23/, 17 files) all passed GREEN before this commit, and before any
real KC23 stage result exists. This is the pre-registration record: every
outcome grid below was fixed in code at this commit, ahead of the data that
would decide it.

## GPU queue

| job_id | stage | seed | status | wallclock_s | gate |
|---|---|---|---|---|---|
| smoke_gpu1 | SMOKE | 42 | done | 24.0 | 0 |

## CPU queue

| job_id | stage | seed | status | wallclock_s | gate |
|---|---|---|---|---|---|
| smoke_cpu1 | SMOKE | 42 | done | 189.0 | 0 |

